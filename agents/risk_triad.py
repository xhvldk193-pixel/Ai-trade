# =============================================
# Risk Triad Debate Runner (Aggressive / Conservative / Neutral)
# =============================================
# 원본: TradingAgents/tradingagents/graph/conditional_logic.py + risk_debators
# 적용:
#   - Bull/Bear 토론 결과 + 공통 데이터를 입력으로
#   - 3자(공격/보수/중립)가 리스크 관점에서 추가 토론
#   - 결과는 최종 analyze_with_claude() 의 [사전 토론] 블록에 추가 주입
# =============================================
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import anthropic

from config import CLAUDE_API_KEY
from .risk_prompts import (
    AGGRESSIVE_SYSTEM,
    CONSERVATIVE_SYSTEM,
    NEUTRAL_SYSTEM,
    RISK_USER_TEMPLATE,
    risk_opponent_block,
)

# AgentMemories 는 선택적 의존
try:
    from .memory import AgentMemories
except Exception:
    AgentMemories = None  # type: ignore


# ── 설정 ──────────────────────────────────────────
# Risk Triad 모델. 필요 시 env 로 오버라이드.
RISK_MODEL = os.getenv("RISK_MODEL", "claude-haiku-4-5-20251001")

# 한 라운드 = Aggressive → Conservative → Neutral 순서로 1발언씩.
# 기본 1라운드 (총 3회 호출). 2라운드면 6회 — 토론이 길어진다.
RISK_MAX_ROUNDS = int(os.getenv("RISK_MAX_ROUNDS", "1"))

# Risk Triad 자체를 끄고 싶을 때: RISK_ENABLED=0
RISK_ENABLED = os.getenv("RISK_ENABLED", "1") not in ("0", "false", "False", "")

# 출력 목표: 400~700자 ≈ 200~350 tokens. 안전 마진 포함 1000으로 제한.
# 기존 7500은 낭비. Risk 에이전트는 리스크 규모 서술이라 조금 더 여유 허용.
RISK_MAX_OUTPUT_TOKENS = int(os.getenv("RISK_MAX_OUTPUT_TOKENS", "1000"))


# 발언 순서 — Aggressive 가 먼저 치고 나가면 Conservative/Neutral 이 반박/중재하는 구조.
# 단, 마지막 발언자가 final analyst 의 종합에 더 큰 영향을 주므로 (recency bias 회피)
# 라운드마다 순환시킴 — _resolve_speaking_order 가 처리.
SPEAKING_ORDER = ("aggressive", "conservative", "neutral")
SIDE_META = {
    "aggressive":    ("Aggressive", "⚔️", AGGRESSIVE_SYSTEM),
    "conservative":  ("Conservative", "🛡️", CONSERVATIVE_SYSTEM),
    "neutral":       ("Neutral", "⚖️", NEUTRAL_SYSTEM),
}


def _resolve_speaking_order(round_index: int, judge_verdict: str = "") -> tuple[str, ...]:
    """라운드별 발언 순서를 결정.

    - 1라운드: 시그널 방향과 반대편이 마지막 발언 (최종 결정에 가장 영향력 있음).
      Bull 우위면 Conservative 가 마지막 (낙관 견제), Bear 우위면 Aggressive 가
      마지막 (비관 견제), 중립이면 기본 순서.
    - 2라운드 이후: 라운드 번호로 순환 (recency bias 분산).
    """
    base = ("aggressive", "conservative", "neutral")
    if round_index == 0:
        if "상방" in judge_verdict:
            # Bull 우위 → Conservative 가 마지막에 견제
            return ("aggressive", "neutral", "conservative")
        elif "하방" in judge_verdict:
            # Bear 우위 → Aggressive 가 마지막에 견제
            return ("conservative", "neutral", "aggressive")
        else:
            return base
    # 2라운드 이후: 회전
    rotation = round_index % 3
    return base[rotation:] + base[:rotation]


@dataclass
class RiskTurn:
    """한 에이전트의 한 발언."""
    side: str          # "aggressive" | "conservative" | "neutral"
    round_index: int   # 0부터 시작
    content: str
    model: str
    elapsed_s: float


@dataclass
class RiskTriadResult:
    """Risk Triad 토론 전체 결과."""
    enabled: bool
    rounds: int
    turns: list[RiskTurn] = field(default_factory=list)
    final_aggressive: str = ""
    final_conservative: str = ""
    final_neutral: str = ""
    error: Optional[str] = None

    def to_payload(self) -> dict:
        return {
            "enabled": self.enabled,
            "rounds": self.rounds,
            "turns": [asdict(t) for t in self.turns],
            "final_aggressive": self.final_aggressive,
            "final_conservative": self.final_conservative,
            "final_neutral": self.final_neutral,
            "error": self.error,
        }


ProgressCallback = Callable[[str, str], None]


def _call_llm(
    client: anthropic.Anthropic,
    system: str,
    cacheable_user: str,
    variable_user: str = "",
) -> str:
    """Risk 에이전트 단일 호출. 429/529 백오프 포함.

    prompt caching: cacheable_user (시장 요약 + Bull/Bear final + Judge) 는
    Risk 3턴 동안 동일하므로 cache_control 마크 → 첫 턴은 write, 이후 2턴은 read.
    variable_user (opponent_block, past_memories) 만 턴마다 변경.
    """
    max_retries = 3
    wait = 8
    content_blocks: list[dict] = [
        {
            "type": "text",
            "text": cacheable_user,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    if variable_user:
        content_blocks.append({"type": "text", "text": variable_user})

    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=RISK_MODEL,
                max_tokens=RISK_MAX_OUTPUT_TOKENS,
                system=system,
                messages=[{"role": "user", "content": content_blocks}],
            )
            if not hasattr(msg, "content") or not isinstance(msg.content, list):
                raise RuntimeError(
                    f"API 응답 형식 오류 — {type(msg).__name__}: {msg!r:.200}"
                )
            text = next((b.text for b in msg.content if b.type == "text"), "")
            return text.strip()
        except anthropic.APIStatusError as e:
            if e.status_code in (429, 529) and attempt < max_retries - 1:
                time.sleep(wait)
                wait *= 2
                continue
            raise


def _make_context_summary(context_blob: str) -> str:
    """context_blob에서 핵심 라인만 추출 (Risk 에이전트용 요약)."""
    lines = context_blob.splitlines()
    keep = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if any(kw in s for kw in (
            "정렬", "RSI", "펀딩", "피보나치", "포지션", "잔고",
            "스큐", "OI", "거시", "추세", "ATR", "MACD", "레버리지", "배분"
        )):
            keep.append(s)
        if len(keep) >= 20:
            break
    return "\n".join(keep) if keep else context_blob[:500]


def _make_account_risk_block(context_blob: str) -> str:
    """context_blob에서 계좌·리스크 관련 라인만 추출해 Risk 에이전트에 명시 주입."""
    lines = context_blob.splitlines()
    acct_lines = []
    in_acct = False
    for line in lines:
        s = line.strip()
        if "[비트겟" in s or "[계좌" in s:
            in_acct = True
        if in_acct:
            if s:
                acct_lines.append(s)
            # 다음 섹션 구분선이 나오면 종료
            if s.startswith("━") and acct_lines:
                break
    if not acct_lines:
        return ""
    return "[계좌 & 리스크 제약 — 반드시 참고]\n" + "\n".join(acct_lines)


def run_risk_triad(
    context_blob: str,
    pair_label: str,
    bull_final: str,
    bear_final: str,
    max_rounds: Optional[int] = None,
    progress_cb: Optional[ProgressCallback] = None,
    agent_memories: Optional["AgentMemories"] = None,
    memory_query: str = "",
    judge_block: str = "",
) -> RiskTriadResult:
    """
    Aggressive/Conservative/Neutral 3자 토론을 실행한다.

    Parameters
    ----------
    context_blob : str
        공통 데이터 블록 (analyzer._build_context_blob 결과).
    pair_label : str
        "BTC/USDC" 등.
    bull_final, bear_final : str
        직전 Bull/Bear 토론 최종 발언. 비어 있어도 동작.
    max_rounds : int, optional
        None 이면 env RISK_MAX_ROUNDS 사용.
    progress_cb : callable, optional
        (phase, detail) — phase 는 "risk_aggressive"/"risk_conservative"/"risk_neutral".
    agent_memories : AgentMemories, optional
        역할별 과거 메모리 — aggressive/conservative/neutral 각자의 회상.
    memory_query : str
        BM25 쿼리용 상황 요약 문자열.
    judge_block : str
        투자 심판 결론 블록. 빈 문자열이면 생략.

    Returns
    -------
    RiskTriadResult
    """
    rounds = max_rounds if max_rounds is not None else RISK_MAX_ROUNDS

    if not RISK_ENABLED:
        return RiskTriadResult(enabled=False, rounds=0)
    if not CLAUDE_API_KEY:
        return RiskTriadResult(enabled=False, rounds=0, error="CLAUDE_API_KEY 미설정")

    # judge_block 에서 판정 결과 추출 (발언 순서 결정용)
    # 형식: "...판정: 상방 우위..." 또는 "판정: 하방 우위" 또는 "판정: 중립"
    judge_verdict = ""
    if judge_block:
        import re as _re
        m = _re.search(r"판정\s*[:：]\s*(상방 우위|하방 우위|중립)", judge_block)
        if m:
            judge_verdict = m.group(1)

    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    result = RiskTriadResult(enabled=True, rounds=rounds)

    # 메모리 쿼리 개선: 키워드 라인 추출
    if memory_query:
        _query = memory_query
    else:
        _kw_lines = []
        for _l in context_blob.splitlines():
            _s = _l.strip()
            if any(kw in _s for kw in ("RSI", "MACD", "펀딩", "추세", "정렬", "스큐", "OI", "포지션")):
                _kw_lines.append(_s)
            if len(_kw_lines) >= 8:
                break
        _query = " | ".join(_kw_lines) if _kw_lines else context_blob[:300]
    last = {"aggressive": "", "conservative": "", "neutral": ""}

    # 루프 전 한 번만 계산 (반복 낭비 방지)
    _context_summary   = _make_context_summary(context_blob)
    _account_risk_block = _make_account_risk_block(context_blob)

    # prompt caching prefix: pair_label + context_summary + account_risk + Bull/Bear/Judge.
    # 3턴 동안 동일하므로 첫 턴은 cache write (1.25x), 이후 2턴은 cache read (0.1x) → 큰 절감.
    cacheable_user = (
        f"{pair_label} 리스크 토론 — 시장 요약 + 사전 Bull/Bear 토론입니다.\n\n"
        f"[시장 핵심 요약]\n{_context_summary}\n\n"
        + (
            f"{_account_risk_block}\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            if _account_risk_block else ""
        )
        + f"[Bull 의 주장]\n{bull_final or '(직전 Bull 의견 없음)'}\n\n"
        + f"[Bear 의 주장]\n{bear_final or '(직전 Bear 의견 없음)'}\n"
        + (
            f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n{judge_block}\n"
            if judge_block else ""
        )
        + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )

    try:
        for r in range(rounds):
            speaking_order = _resolve_speaking_order(r, judge_verdict)
            for side in speaking_order:
                label, icon, system_prompt = SIDE_META[side]

                if progress_cb:
                    progress_cb(
                        f"risk_{side}",
                        f"{label} 라운드 {r + 1}/{rounds} 분석 중",
                    )

                opponent_block_text = risk_opponent_block(
                    aggressive_last=last["aggressive"],
                    conservative_last=last["conservative"],
                    neutral_last=last["neutral"],
                    speaking_side=side,
                )

                # 역할별 메모리 회상 (첫 라운드에만)
                past = ""
                if r == 0 and agent_memories is not None:
                    past = agent_memories.recall(side, _query, top_k=2)

                # 이번 라운드에서 본인 외 누군가 발언이 있었는지 → 반박 지시
                has_opponent = any(v for k, v in last.items() if k != side)
                rebuttal_instruction = (
                    "다른 두 분석관의 논리에서 약한 부분을 구체적으로 지적하고 "
                    "당신의 리스크 관점을 관철하세요."
                    if has_opponent
                    else "당신의 관점을 선제적으로 펼치세요."
                )

                # 가변부: opponent_block + past_memories + instruction
                variable_user = (
                    f"{opponent_block_text}\n"
                    + (
                        f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n{past}\n"
                        if past else ""
                    )
                    + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    + "위 정보를 바탕으로 당신의 리스크 관점을 펼치세요.\n"
                    + "계좌 정보(잔고·배분·레버리지)를 반드시 고려해 사이즈·손익비 판단을 구체화하세요.\n"
                    + rebuttal_instruction
                )

                t0 = time.time()
                reply = _call_llm(client, system_prompt, cacheable_user, variable_user)
                elapsed = time.time() - t0

                result.turns.append(RiskTurn(
                    side=side,
                    round_index=r,
                    content=reply,
                    model=RISK_MODEL,
                    elapsed_s=round(elapsed, 2),
                ))
                last[side] = reply

        result.final_aggressive = last["aggressive"]
        result.final_conservative = last["conservative"]
        result.final_neutral = last["neutral"]
        return result

    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        result.final_aggressive = last["aggressive"]
        result.final_conservative = last["conservative"]
        result.final_neutral = last["neutral"]
        return result


def format_risk_block(result: RiskTriadResult) -> str:
    """
    Risk Triad 토론 결과를 최종 프롬프트 주입용 텍스트 블록으로 변환.
    비활성/실패면 빈 문자열.
    """
    if not result.enabled:
        return ""
    if result.error and not result.turns:
        return f"[사전 리스크 토론]\n  수행 실패 — {result.error}"
    if not result.turns:
        return ""

    lines = ["[사전 리스크 토론 — Aggressive vs Conservative vs Neutral]"]
    lines.append(
        "  ⚠️ 세 관점 모두 같은 데이터를 보지만 리스크 성향이 서로 다릅니다. "
        "최종 애널리스트는 공격/보수 양 극단을 비교하고, 중도의 균형안을 참고해 "
        "'📝 대응' 섹션의 공격적·보수적 라인을 결정하세요."
    )

    for t in result.turns:
        label, icon, _ = SIDE_META[t.side]
        header = f"\n[라운드 {t.round_index + 1} · {icon} {label}]"
        lines.append(header)
        for raw in t.content.splitlines():
            lines.append(f"  {raw}" if raw else "")

    if result.error:
        lines.append(f"\n  ⚠️ 토론 일부 실패 — {result.error}")

    return "\n".join(lines)
