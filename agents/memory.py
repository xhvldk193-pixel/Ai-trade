# =============================================
# Financial Situation Memory (BM25 기반)
# =============================================
# 원본: TradingAgents/tradingagents/agents/utils/memory.py
# 적용:
#   - OpenAI embeddings 의존 제거 → BM25 (rank_bm25) 로 대체
#     * 순수 로컬/오프라인, 외부 API 호출 없음
#     * BTC 분석 맥락에서 "지지 이탈 + 펀딩 과열" 류 키워드 매칭에 충분
#   - JSONL 파일 기반 영속화 (/data/memory/*.jsonl)
#   - 분석 단계마다 get_memories(current_situation, top_k) 로 회상
#   - 분석 사후 add_situation(situation, advice, outcome) 로 경험 축적 (reflection)
# =============================================
from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except Exception:
    BM25Okapi = None  # type: ignore
    _BM25_AVAILABLE = False


# ── 저장 위치 ──────────────────────────────────────
# crypto_analyzer/ 와 같은 레벨에 data/memory/ 생성.
# 이 모듈 파일 기준 부모의 부모.
_THIS_DIR = Path(__file__).resolve().parent         # .../crypto_analyzer/agents
_PROJECT_DIR = _THIS_DIR.parent                     # .../crypto_analyzer
DEFAULT_MEMORY_DIR = Path(
    os.getenv("MEMORY_DIR", str(_PROJECT_DIR / "data" / "memory"))
)


# ── 토크나이저 ────────────────────────────────────
# 한국어 + 영문 + 숫자 혼재 → 공백/특수문자로 단순 분리 후 소문자화.
# 한글 형태소 분석기를 쓰면 더 좋지만 의존성 무거워져서 생략.
_TOKEN_RE = re.compile(r"[A-Za-z가-힣0-9%]+")
# 5자리 이상 순수 숫자(절대가격)는 BM25에서 노이즈 — 제외
# "81412", "82829" 같은 가격 숫자가 다르면 유사 상황도 0점 처리되는 문제 방지
_PRICE_NUM_RE = re.compile(r"^\d{5,}$")


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    # 절대가격 숫자 제거 — 범주형 태그·짧은 숫자(24h, 3%)는 유지
    return [t for t in tokens if not _PRICE_NUM_RE.match(t)]


@dataclass
class MemoryRecord:
    """한 건의 경험 — 상황 + 당시 조언 + (선택) 결과."""
    timestamp: str       # ISO8601
    situation: str       # 당시 시장 상황 요약 (쿼리 대상)
    advice: str          # 당시 애널리스트 조언/판단
    outcome: str = ""    # 사후 결과 (reflection 단계에서 채움)
    meta: dict = None    # type: ignore # 자유 메타 (심볼·PnL 등)

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.meta is None:
            d["meta"] = {}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryRecord":
        return cls(
            timestamp=d.get("timestamp", ""),
            situation=d.get("situation", ""),
            advice=d.get("advice", ""),
            outcome=d.get("outcome", ""),
            meta=d.get("meta") or {},
        )


class FinancialSituationMemory:
    """
    시장 상황-조언 페어를 저장하고 BM25 로 유사 상황을 회상한다.

    Usage
    -----
    mem = FinancialSituationMemory("bull_bear")
    mem.add_situation(situation_text, advice_text, outcome="")
    hits = mem.get_memories(current_situation_text, top_k=3)
      # hits: [{"record": MemoryRecord.to_dict(), "score": float}, ...]
    """

    def __init__(
        self,
        name: str,
        memory_dir: Optional[Path] = None,
    ):
        self.name = name
        self.memory_dir = Path(memory_dir) if memory_dir else DEFAULT_MEMORY_DIR
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.memory_dir / f"{name}.jsonl"

        self._lock = threading.Lock()
        self._records: list[MemoryRecord] = []
        self._load()

    # ── 영속화 ────────────────────────────────────
    def _load(self) -> None:
        self._records = []
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._records.append(MemoryRecord.from_dict(obj))
        except OSError:
            self._records = []

    def _append_to_disk(self, record: MemoryRecord) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def _rewrite_disk(self) -> None:
        tmp = self.path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in self._records:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        tmp.replace(self.path)

    # ── 내부 dedup 헬퍼 ────────────────────────────
    # ── API ───────────────────────────────────────
    def add_situation(
        self,
        situation: str,
        advice: str,
        outcome: str = "",
        meta: Optional[dict] = None,
        dedup_threshold: float = 0.85,
        dedup_window: int = 3,
    ) -> Optional[MemoryRecord]:
        """
        이번 판단을 메모리에 추가한다.

        dedup:
          최근 `dedup_window` 개 기록의 situation 토큰과 Jaccard >= threshold 이면
          '실질적으로 같은 구조' 로 보고 저장을 건너뛴다 (무한 누적 방지).
          건너뛴 경우 반환값은 None.

          BTC 분석처럼 지표가 비슷한 상황이 반복될 때 0.92는 너무 엄격해서
          대부분이 dedup 처리됨 → 0.85 / window=3 으로 완화.
          기본값을 0 (off) 에서 0.85 (on) 로 변경 — 메모리 무한 누적 방지.
          호출처에서 명시적으로 0 을 주면 dedup 비활성화 가능.
        """
        new_tokens = _tokenize(situation)
        with self._lock:
            # 최근 N개와 비교
            recent = self._records[-dedup_window:] if dedup_window > 0 else []
            for r in recent:
                sim = self._jaccard(new_tokens, _tokenize(r.situation))
                if dedup_threshold > 0 and sim >= dedup_threshold:
                    return None

            rec = MemoryRecord(
                # tz-aware UTC ISO8601 (Z 표기) — datetime.utcnow() 는 deprecated
                timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                situation=situation.strip(),
                advice=advice.strip(),
                outcome=outcome.strip(),
                meta=meta or {},
            )
            self._records.append(rec)
            self._append_to_disk(rec)
        return rec

    @staticmethod
    def _norm_ts(ts: str) -> str:
        """타임스탬프를 비교용 정규형으로 통일 (초 단위 UTC, Z 표기)."""
        if not ts:
            return ""
        try:
            _ts = ts.strip().replace(" ", "T")
            if _ts.endswith("Z"):
                _ts = _ts[:-1] + "+00:00"
            dt = datetime.fromisoformat(_ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            return ts.strip()

    def update_outcome(self, timestamp: str, outcome: str) -> bool:
        """
        특정 timestamp 의 기록에 사후 결과를 덧붙인다.
        Reflection 에서 사용.
        Z / +00:00 / 공백 등 형식 차이를 정규화해 비교 — 미기록 고착 버그 수정.
        """
        norm_target = self._norm_ts(timestamp)
        with self._lock:
            # 정규화 비교 (주 경로)
            for r in self._records:
                if self._norm_ts(r.timestamp) == norm_target:
                    r.outcome = (r.outcome + "\n" + outcome).strip() if r.outcome else outcome
                    self._rewrite_disk()
                    return True
            # fallback: 원본 문자열 완전 일치
            for r in self._records:
                if r.timestamp == timestamp:
                    r.outcome = (r.outcome + "\n" + outcome).strip() if r.outcome else outcome
                    self._rewrite_disk()
                    return True
        return False

    @staticmethod
    def _jaccard(a: list[str], b: list[str]) -> float:
        """토큰 집합 간 Jaccard 유사도 (0~1)."""
        if not a or not b:
            return 0.0
        sa, sb = set(a), set(b)
        inter = len(sa & sb)
        union = len(sa | sb)
        return inter / union if union else 0.0

    def _jaccard_rank(
        self,
        query_tokens: list[str],
        corpus_tokens: list[list[str]],
        records: list,
        top_k: int,
    ) -> list[dict]:
        """
        Jaccard 유사도 기반 랭킹.
        BM25 IDF 붕괴(소규모 균질 코퍼스) 시 대체 스코어링으로 사용.
        """
        scored = [
            (r, self._jaccard(query_tokens, doc_tokens))
            for r, doc_tokens in zip(records, corpus_tokens)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            {"record": r.to_dict(), "score": round(float(s), 4)}
            for r, s in scored[:top_k]
        ]

    def get_memories(
        self,
        query: str,
        top_k: int = 4,
        require_outcome: bool = True,
    ) -> list[dict]:
        """
        query 와 가장 유사한 상황 top_k 개를 반환.
        반환: [{"record": {...}, "score": float}, ...]

        전략:
          1) BM25 시도 → 유효한 양수 점수 있으면 BM25 랭킹 반환
          2) BM25 전부 0점 (소규모 균질 코퍼스 → IDF 붕괴) → Jaccard 랭킹으로 대체
          3) rank_bm25 미설치 / 쿼리 토큰 없음 → Jaccard 랭킹

        require_outcome=True(기본값):
          outcome(리플렉션 결과)이 있는 기록을 우선 반환.
          outcome 없는 기록은 슬롯이 남을 때만 보충.
          학습 기여 없는 기록이 유사도 슬롯을 낭비하는 문제 방지.
        """
        with self._lock:
            records = list(self._records)

        if not records:
            return []

        if require_outcome:
            records_with    = [r for r in records if r.outcome]
            records_without = [r for r in records if not r.outcome]
        else:
            records_with    = records
            records_without = []

        def _rank(recs, k):
            if not recs:
                return []
            corpus_tokens = [_tokenize(r.situation) for r in recs]
            query_tokens  = _tokenize(query)
            if not query_tokens:
                return self._jaccard_rank(query_tokens, corpus_tokens, recs, k)
            if _BM25_AVAILABLE:
                try:
                    bm25   = BM25Okapi(corpus_tokens)
                    scores = bm25.get_scores(query_tokens)
                    ranked = sorted(zip(recs, scores), key=lambda x: x[1], reverse=True)[:k]
                    ranked_hit = [(r, s) for r, s in ranked if s > 0]
                    if ranked_hit:
                        return [{"record": r.to_dict(), "score": float(s)} for r, s in ranked_hit]
                except Exception:
                    pass
            return self._jaccard_rank(query_tokens, corpus_tokens, recs, k)

        results = _rank(records_with, top_k)
        if len(results) < top_k and records_without:
            results += _rank(records_without, top_k - len(results))
        return results

    def __len__(self) -> int:
        return len(self._records)

    def size(self) -> int:
        return len(self._records)

    # ── 공개 열람 API (server.py 등 외부에서 _records 직접 접근 대신 사용) ──
    def list_records(self) -> list[MemoryRecord]:
        """전체 기록의 shallow copy 를 반환."""
        with self._lock:
            return list(self._records)

    def list_pending_reflections(
        self,
        min_age_seconds: float = 300.0,   # 5분 (기존 30분 → 완화)
        limit: int = 10,                  # 5 → 10: 200건 적체 해소 가속
    ) -> list[MemoryRecord]:
        """
        Reflection 대상 — outcome 이 비어 있고 충분히 시간이 지난 기록만.
        오래된 것 우선 (FIFO), 최대 `limit` 개.
        """
        import datetime as _dt
        now = _dt.datetime.now(_dt.timezone.utc)
        out: list[MemoryRecord] = []

        with self._lock:
            for rec in self._records:
                if rec.outcome:
                    continue
                try:
                    _raw_ts = rec.timestamp.strip().replace(" ", "T")
                    if _raw_ts.endswith("Z"):
                        _raw_ts = _raw_ts[:-1] + "+00:00"
                    ts = _dt.datetime.fromisoformat(_raw_ts)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=_dt.timezone.utc)
                except Exception:
                    continue
                elapsed = (now - ts).total_seconds()
                if elapsed < min_age_seconds:
                    continue
                out.append(rec)
                if len(out) >= limit:
                    break
        return out


# ── 프롬프트 주입용 포매터 ──────────────────────────
def format_memory_block(memories: list[dict]) -> str:
    """
    get_memories() 결과를 최종 프롬프트 주입용 블록으로 변환.
    체크리스트/반복실수 항목을 최상단에 강조 배치 — LLM이 현재 분석 전에 반드시 확인하도록.
    비어 있으면 빈 문자열 반환.
    """
    if not memories:
        return ""

    # 먼저 모든 사례의 체크리스트/반복실수를 집계해 최상단에 배치
    # LLM 이 현재 분석 전에 누적 패턴을 먼저 인식하도록
    all_checklists: list[str] = []
    all_repeats: list[str] = []
    for _mem in memories:
        _rec = _mem.get("record", {}) if isinstance(_mem, dict) else {}
        _out = (_rec.get("outcome") or "").strip()
        for _line in _out.splitlines():
            _s = _line.strip()
            if _s.startswith("다음 체크리스트:") and _s not in all_checklists:
                all_checklists.append(_s)
            if _s.startswith("반복 실수 금지:") and _s not in all_repeats:
                all_repeats.append(_s)

    lines = ["[과거 유사 상황 — 자기학습 블록]"]
    if all_checklists or all_repeats:
        lines.append("━━ 누적 학습 체크리스트 (현재 데이터로 먼저 확인하세요) ━━")
        for _cl in all_checklists:
            lines.append(f"  ⚑ {_cl}")
        for _rp in all_repeats:
            lines.append(f"  ⚠ {_rp}")
        lines.append("")
    lines.append(
        "  아래 사례들은 유사 지표 조합에서의 과거 판단 기록입니다. "
        "답습하지 말고 현재 조건과 차이를 대조해 교훈만 추출하세요."
    )

    for i, item in enumerate(memories, start=1):
        rec = item.get("record", {}) if isinstance(item, dict) else {}
        ts = rec.get("timestamp", "?")
        score = item.get("score", 0.0)
        situation = (rec.get("situation") or "").strip()
        advice = (rec.get("advice") or "").strip()
        outcome = (rec.get("outcome") or "").strip()

        lines.append(f"\n— 사례 {i} · {ts} · 유사도 {score:.2f} —")
        if situation:
            # 프롬프트 주입용 — 300자로 압축 (UI는 state.memories[] 직접 읽음)
            snippet = situation if len(situation) <= 300 else situation[:300] + " …"
            lines.append(f"  상황: {snippet}")
        if advice:
            # 핵심 논거는 앞부분에 집중 — 200자로 압축
            snippet = advice if len(advice) <= 200 else advice[:200] + " …"
            lines.append(f"  당시 조언: {snippet}")
        if outcome:
            # 체크리스트 + 반복 실수 금지 두 줄 모두 추출해 상단 강조
            checklist_line = ""
            repeat_line = ""
            for _cl in outcome.splitlines():
                _s = _cl.strip()
                if _s.startswith("다음 체크리스트:") and not checklist_line:
                    checklist_line = _s
                if _s.startswith("반복 실수 금지:") and not repeat_line:
                    repeat_line = _s
            if checklist_line:
                lines.append(f"  ⚑ {checklist_line}")
            if repeat_line:
                lines.append(f"  ⚠ {repeat_line}")
            # 체크리스트/반복실수는 위에서 별도 강조 — 여기선 나머지 맥락만
            snippet = outcome if len(outcome) <= 400 else outcome[:400] + " …"
            lines.append(f"  실제 결과: {snippet}")
        else:
            lines.append("  실제 결과: (아직 리플렉션 미기록)")

    return "\n".join(lines)


# ── 편의 팩토리 ────────────────────────────────────
_MEMORIES: dict[str, FinancialSituationMemory] = {}
_FACTORY_LOCK = threading.Lock()


def get_memory(name: str = "analyst") -> FinancialSituationMemory:
    """모듈 전역 싱글턴 — 동일 name 은 같은 인스턴스를 공유."""
    with _FACTORY_LOCK:
        mem = _MEMORIES.get(name)
        if mem is None:
            mem = FinancialSituationMemory(name=name)
            _MEMORIES[name] = mem
        return mem


# ══════════════════════════════════════════════════════
# AgentMemories — 역할별 독립 메모리 집합
# ══════════════════════════════════════════════════════
# TradingAgents 의 bull_memory / bear_memory / trader_memory 패턴을
# BTC 선물 에이전트 구조에 맞게 확장.
# 각 역할이 자신의 과거 판단 이력을 분리해 학습하므로,
# Bull 의 상승 편향 실수는 Bull 메모리에, Bear 의 하락 과신은 Bear 메모리에 쌓인다.

AGENT_ROLES = ("bull", "bear", "judge", "aggressive", "conservative", "neutral", "analyst")

_AGENT_MEMORIES_INSTANCE: Optional["AgentMemories"] = None
_AGENT_MEMORIES_LOCK = threading.Lock()


class AgentMemories:
    """
    역할별 FinancialSituationMemory 를 한 곳에서 관리하는 컨테이너.

    사용 예:
        am = get_agent_memories()
        past = am.recall("bull", situation_query)   # 프롬프트 삽입용 문자열
        am.get("bull").add_situation(...)           # 직접 저장
    """

    def __init__(self, memory_dir: Optional[Path] = None):
        _dir = Path(memory_dir) if memory_dir else DEFAULT_MEMORY_DIR
        self._stores: dict[str, FinancialSituationMemory] = {
            role: FinancialSituationMemory(role, _dir)
            for role in AGENT_ROLES
        }

    def get(self, role: str) -> FinancialSituationMemory:
        """역할별 메모리 인스턴스 반환. 알 수 없는 role 도 자동 생성."""
        if role not in self._stores:
            self._stores[role] = FinancialSituationMemory(role)
        return self._stores[role]

    def recall(self, role: str, situation: str, top_k: int = 4) -> str:
        """
        역할별 과거 기억을 회상해 프롬프트 삽입용 텍스트로 반환.
        기억이 없거나 BM25 매칭 실패 시 빈 문자열.
        """
        mems = self.get(role).get_memories(situation, top_k=top_k)
        if not mems:
            return ""
        lines = ["[과거 유사 상황에서의 내 판단 이력]"]
        for i, item in enumerate(mems, 1):
            rec = item.get("record", {}) if isinstance(item, dict) else {}
            score = item.get("score", 0.0)
            advice = (rec.get("advice") or "").strip()
            outcome = (rec.get("outcome") or "").strip()
            ts = rec.get("timestamp", "?")
            # 역할별 회상 — outcome 에 리플렉션 교훈이 담기므로 충분히 전달
            advice_snippet  = advice[:400]  + " …" if len(advice)  > 400  else advice
            outcome_snippet = outcome[:600] + " …" if len(outcome) > 600 else outcome

            # "다음 체크리스트:" 항목을 별도 추출해 회상 상단에 강조
            checklist_line = ""
            repeat_line = ""
            if outcome:
                for _cl in outcome.splitlines():
                    _s = _cl.strip()
                    if _s.startswith("다음 체크리스트:") and not checklist_line:
                        checklist_line = _s
                    if _s.startswith("반복 실수 금지:") and not repeat_line:
                        repeat_line = _s

            lines.append(f"\n— 사례 {i} · {ts} · 유사도 {score:.2f} —")
            if checklist_line:
                lines.append(f"  ⚑ {checklist_line}")
            if repeat_line:
                lines.append(f"  ⚠ {repeat_line}")
            if advice_snippet:
                lines.append(f"  당시 주장: {advice_snippet}")
            if outcome_snippet:
                lines.append(f"  이후 결과: {outcome_snippet}")
            else:
                lines.append("  이후 결과: (미기록 — reflection 대기)")
        return "\n".join(lines)

    def all_roles(self) -> list[str]:
        return list(self._stores.keys())


def get_agent_memories() -> "AgentMemories":
    """프로세스 전역 싱글턴 AgentMemories."""
    global _AGENT_MEMORIES_INSTANCE
    with _AGENT_MEMORIES_LOCK:
        if _AGENT_MEMORIES_INSTANCE is None:
            _AGENT_MEMORIES_INSTANCE = AgentMemories()
        return _AGENT_MEMORIES_INSTANCE
