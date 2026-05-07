# =============================================
# Reflection Agent — 사후 결과로 메모리 업데이트 (역할별 분리)
# =============================================
# 원본: TradingAgents/tradingagents/agents/utils/agent_utils.py (Reflector 부분)
# 적용:
#   - "이번 판단 이후 실제 가격이 어떻게 움직였나" 를 간단한 변화율로 계측
#   - 역할별(analyst/bull/bear/judge/aggressive/conservative/neutral) 전용 시스템 프롬프트
#   - Claude 가 '무엇을 잘했고/놓쳤고/다음에는 어떻게' 를 1~2문단 리플렉션으로 작성
#   - FinancialSituationMemory.add_situation(...) 에 outcome 으로 기록
#
# 호출 방식:
#   - server.py 에 /reflect 엔드포인트
#   - schedule 스킬로 매 N시간 자동 실행
# =============================================
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import anthropic


def _get_client():
    """단일 anthropic 클라이언트 반환 (순환 임포트 회피용 lazy import)."""
    from agents import get_anthropic_client
    return get_anthropic_client()



from config import CLAUDE_API_KEY
from .memory import FinancialSituationMemory, get_memory


REFLECTION_MODEL = os.getenv("REFLECTION_MODEL", "claude-haiku-4-5-20251001")


# ── 역할별 시스템 프롬프트 ────────────────────────────

_BASE_RULES = """
출력 규칙:
- 마크다운(**,##,---), HTML 금지. 일반 텍스트 + 최소 이모지.
- 300~500자. 장황함 금지.
- 마지막 줄은 반드시 '다음 체크리스트:' 로 시작하는 1~2줄 요약."""

# 모든 reflection 시스템 프롬프트에 공통으로 들어가는 hindsight bias 방지 가드.
# LLM 이 사후 가격 변화를 보면 '당시 X 가 명백한 신호였는데 놓쳤다' 같은 결과론에
# 매우 잘 빠지는데, 이게 메모리에 저장되어 다음 분석 회상 시 잘못된 패턴을 강화함.
_HINDSIGHT_GUARD = """
결과론 금지 (절대 준수):
- 사후 가격을 알고 있다고 해서 '당시 X 가 명백한 신호였는데 놓쳤다' 라고 말하지 말 것.
- 가격 결과는 노이즈·거시 충격·유동성 등으로도 변동할 수 있습니다. 판단은 당시 데이터 기준으로만.
- 평가 기준: '당시 데이터로 그 판단이 합리적이었나' — 결과의 좋고나쁨은 부차적.
- 결과 좋음 ≠ 판단 옳음, 결과 나쁨 ≠ 판단 틀림. 두 차원을 분리하여 평가하세요."""

ROLE_REFLECTION_SYSTEMS: dict[str, str] = {
    "analyst": f"""당신은 BTC 선물 애널리스트의 'Reflection Coach' 입니다.
지난 종합 판단과 실제 시장 움직임을 대조해 교훈을 뽑아내는 역할입니다.

원칙:
1. 그때 가용했던 정보 기준으로 '놓친 단서'와 '과대평가한 근거'를 각각 짚으세요.
2. '무엇이 맞았는가' 와 '무엇이 틀렸는가' 를 분리해서 서술.
3. 방향성만 맞았다고 성공으로 포장하지 말 것. 진입 레벨/타이밍까지 평가.
{_HINDSIGHT_GUARD}
{_BASE_RULES}""",

    "bull": f"""당신은 BTC 선물 Bull Researcher 의 'Reflection Coach' 입니다.
과거 상방 논거가 실제 가격 움직임과 얼마나 정합했는지 평가하는 역할입니다.

원칙:
1. 상방 논거 중 가장 강력하게 작동한 근거와 실패한 근거를 각각 짚으세요.
2. Bear 의 반박 중 나중에 실제로 맞아떨어진 것이 있다면 인정하세요.
3. '다음 상방 논거에서 더 주목해야 할 신호' 를 구체화하세요.
{_HINDSIGHT_GUARD}
{_BASE_RULES}""",

    "bear": f"""당신은 BTC 선물 Bear Researcher 의 'Reflection Coach' 입니다.
과거 하방 논거가 실제 가격 움직임과 얼마나 정합했는지 평가하는 역할입니다.

원칙:
1. 하방 논거 중 가장 강력하게 작동한 근거와 실패한 근거를 각각 짚으세요.
2. Bull 의 주장 중 나중에 실제로 맞아떨어진 것이 있다면 인정하세요.
3. '다음 하방 논거에서 더 주목해야 할 신호' 를 구체화하세요.
{_HINDSIGHT_GUARD}
{_BASE_RULES}""",

    "judge": f"""당신은 Bull/Bear 토론의 'Investment Judge Reflection Coach' 입니다.
과거 심판 판정이 실제 가격 결과와 일치했는지 평가하는 역할입니다.

원칙:
1. 판정(상방/하방/중립)이 실제 방향과 일치했는가, 이유는 무엇인가.
2. 당시 Bull 과 Bear 중 어느 쪽 논거가 더 데이터와 정합했는지 사후 평가.
3. 앞으로 유사 상황에서 더 정확한 판정을 내리기 위한 패턴을 추출하세요.
{_HINDSIGHT_GUARD}
{_BASE_RULES}""",

    "aggressive": f"""당신은 Aggressive Risk Analyst 의 'Reflection Coach' 입니다.
과거 공격적 리스크 권고가 실제 결과와 얼마나 맞았는지 평가하는 역할입니다.

원칙:
1. 공격적 접근이 수익을 냈는지, 아니면 불필요한 리스크를 감수했는지 평가.
2. Conservative 의 우려가 현실이 된 경우 있으면 구체적으로 짚으세요.
3. '공격적 진입이 정당화되는 조건' 을 더 정밀하게 정의하세요.
{_HINDSIGHT_GUARD}
{_BASE_RULES}""",

    "conservative": f"""당신은 Conservative Risk Analyst 의 'Reflection Coach' 입니다.
과거 보수적 리스크 권고가 실제 결과와 얼마나 맞았는지 평가하는 역할입니다.

원칙:
1. 관망/축소 권고가 기회비용을 발생시켰는지, 아니면 손실을 막았는지 평가.
2. 지나치게 과도한 방어로 놓친 기회가 있으면 솔직하게 인정하세요.
3. '보수적 관망이 정당화되는 조건' 을 더 정밀하게 정의하세요.
{_HINDSIGHT_GUARD}
{_BASE_RULES}""",

    "neutral": f"""당신은 Neutral Risk Analyst 의 'Reflection Coach' 입니다.
과거 균형적 리스크 권고가 실제 결과와 얼마나 맞았는지 평가하는 역할입니다.

원칙:
1. 중도적 접근이 R:R 관점에서 실제로 최적이었는지 평가.
2. Aggressive/Conservative 중 어느 쪽 판단이 더 좋은 결과를 냈는지 확인.
3. '분할 진입/부분 청산 전략의 실제 효과' 를 구체적으로 평가하세요.
{_HINDSIGHT_GUARD}
{_BASE_RULES}""",
}

# 역할에 해당하는 시스템 프롬프트가 없으면 analyst 로 폴백
DEFAULT_REFLECTION_SYSTEM = ROLE_REFLECTION_SYSTEMS["analyst"]


# ── 유저 프롬프트 ───────────────────────────────────

REFLECTION_USER_TEMPLATE = """[과거 판단 시점] {past_ts}
[역할] {role}
[판단 당시 상황 요약]
{past_situation}

[판단 당시 발언/조언]
{past_advice}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[사후 가격 변화]
기준가: ${price_then:,.2f}
현재가: ${price_now:,.2f}
변화율: {pct_change:+.2f}% ({direction})
경과 시간: {elapsed_label}
{high_low_block}{tpsl_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{missed_block}위 정보를 바탕으로 리플렉션을 작성하세요.
{missed_instruction}마지막 줄은 반드시 '다음 체크리스트:' 로 시작하는 1~2줄 요약으로 마치세요.
"""


@dataclass
class ReflectionResult:
    """개별 기록 하나에 대한 리플렉션 결과."""
    timestamp: str          # 대상 MemoryRecord 의 timestamp
    role: str               # 역할 이름
    price_then: float
    price_now: float
    pct_change: float
    reflection_text: str
    updated: bool           # 메모리에 outcome 업데이트 성공 여부
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "timestamp":       self.timestamp,
            "role":            self.role,
            "price_then":      self.price_then,
            "price_now":       self.price_now,
            "pct_change":      round(self.pct_change, 4),
            "reflection_text": self.reflection_text,
            "updated":         self.updated,
            "error":           self.error,
        }


def _call_llm(client: anthropic.Anthropic, system: str, user: str) -> str:
    max_retries = 3
    wait = 8
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=REFLECTION_MODEL,
                max_tokens=800,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            if not hasattr(msg, "content") or not isinstance(msg.content, list):
                raise RuntimeError(
                    f"API 응답 형식 오류 — {type(msg).__name__}: {msg!r:.200}"
                )
            return next((b.text for b in msg.content if b.type == "text"), "").strip()
        except anthropic.APIStatusError as e:
            if e.status_code in (429, 529) and attempt < max_retries - 1:
                time.sleep(wait)
                wait *= 2
                continue
            raise


def _elapsed_label(seconds: float) -> str:
    if seconds < 3600:
        return f"{seconds / 60:.0f}분"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}시간"
    return f"{seconds / 86400:.1f}일"


def reflect_on_record(
    record_ts: str,
    situation: str,
    advice: str,
    price_then: float,
    price_now: float,
    elapsed_seconds: float,
    memory: Optional[FinancialSituationMemory] = None,
    memory_name: str = "analyst",
) -> ReflectionResult:
    """
    단일 과거 기록에 대해 리플렉션을 실행하고 메모리의 outcome 필드를 업데이트한다.
    하위 호환용 — 내부적으로 reflect_for_role 을 호출한다.
    """
    return reflect_for_role(
        role=memory_name,
        record_ts=record_ts,
        situation=situation,
        advice=advice,
        price_then=price_then,
        price_now=price_now,
        elapsed_seconds=elapsed_seconds,
        memory=memory,
    )


def reflect_for_role(
    role: str,
    record_ts: str,
    situation: str,
    advice: str,
    price_then: float,
    price_now: float,
    elapsed_seconds: float,
    memory: Optional[FinancialSituationMemory] = None,
    price_high: Optional[float] = None,
    price_low: Optional[float] = None,
) -> ReflectionResult:
    """
    역할별 전용 시스템 프롬프트를 사용해 리플렉션을 실행한다.

    Parameters
    ----------
    role : str
        "analyst" | "bull" | "bear" | "judge" | "aggressive" | "conservative" | "neutral"
    record_ts : str
        대상 MemoryRecord 의 timestamp.
    situation : str
        판단 당시 상황 요약 (situation_tags 또는 context_blob 일부).
    advice : str
        판단 당시 발언 / 조언 (debate 발언, 리스크 권고, 애널리스트 리포트 등).
    price_then : float
        판단 시점 가격.
    price_now : float
        현재 가격.
    elapsed_seconds : float
        경과 초.
    memory : FinancialSituationMemory, optional
        None 이면 get_memory(role) 로 자동 취득.
    """
    if memory is None:
        memory = get_memory(role)

    if not CLAUDE_API_KEY:
        return ReflectionResult(
            timestamp=record_ts,
            role=role,
            price_then=price_then,
            price_now=price_now,
            pct_change=0.0,
            reflection_text="",
            updated=False,
            error="CLAUDE_API_KEY 미설정",
        )

    # 가격 변화 계산
    pct = 0.0
    if price_then and price_then != 0:
        pct = (price_now - price_then) / price_then * 100.0
    direction = "상승" if pct > 0 else ("하락" if pct < 0 else "정체")

    # ── 최고가/최저가 블록 ────────────────────────────
    _ph = price_high if price_high else price_now
    _pl = price_low if price_low else price_now
    high_low_block = ""
    if price_high or price_low:
        high_low_block = f"구간 최고가: ${_ph:,.2f} | 구간 최저가: ${_pl:,.2f}\n"

    # ── TP/SL 도달 여부 평가 블록 ──────────────────────
    tpsl_block = ""
    missed_block = ""
    missed_instruction = ""

    # 메모리 record에서 trade_levels 꺼내기
    _rec = next((r for r in memory.list_pending_reflections(min_age_seconds=0, limit=9999)
                 if r.timestamp == record_ts), None)
    _meta = (_rec.meta or {}) if _rec else {}
    _tl = _meta.get("trade_levels") or {}
    _tp = _tl.get("target")
    _sl = _tl.get("stop")
    _is_missed = _meta.get("missed", False)
    _signal = _meta.get("signal", "")

    if _tp or _sl:
        tpsl_lines = ["\n[매매 파라미터 사후 평가]"]
        if _tp:
            _tp_reached = (
                (_signal == "매수" and _ph >= _tp) or
                (_signal == "매도" and _pl <= _tp)
            )
            tpsl_lines.append(
                f"목표가(TP): ${_tp:,.2f} → {'✅ 도달' if _tp_reached else '❌ 미도달'}"
                f" (구간 {'최고' if _signal=='매수' else '최저'}가 ${_ph if _signal=='매수' else _pl:,.2f})"
            )
        if _sl:
            _sl_hit = (
                (_signal == "매수" and _pl <= _sl) or
                (_signal == "매도" and _ph >= _sl)
            )
            tpsl_lines.append(
                f"손절가(SL): ${_sl:,.2f} → {'🔴 손절 발생' if _sl_hit else '✅ 미터치'}"
                f" (구간 {'최저' if _signal=='매수' else '최고'}가 ${_pl if _signal=='매수' else _ph:,.2f})"
            )
        # 되돌림 깊이 평가
        if _signal == "매수" and _sl:
            _pullback_pct = (price_then - _pl) / price_then * 100 if _pl < price_then else 0
            if _pullback_pct > 0:
                tpsl_lines.append(
                    f"되돌림 깊이: -{_pullback_pct:.2f}% "
                    f"(진입가 대비 ${price_then - _pl:,.2f} 최저)"
                )
        tpsl_block = "\n".join(tpsl_lines)

    if _is_missed:
        missed_block = (
            "⚠️ 이 기록은 진입 거부된 케이스입니다.\n"
            "진입하지 못한 이유와 이후 가격 흐름을 대조해 '진입 조건이 너무 엄격했는지' 또는 "
            "'거부가 올바른 판단이었는지' 평가하세요.\n특히 되돌림 깊이와 진입 구간의 적절성에 집중하세요.\n\n"
        )
        missed_instruction = (
            "되돌림 패턴 관점에서: 진입 구간 설정이 적절했는가, "
            "얕은 되돌림 후 재상승 패턴이 나타났는가를 반드시 포함하세요.\n"
        )

    system_prompt = ROLE_REFLECTION_SYSTEMS.get(role, DEFAULT_REFLECTION_SYSTEM)
    prompt = REFLECTION_USER_TEMPLATE.format(
        past_ts=record_ts,
        role=role,
        past_situation=situation.strip() or "(기록 없음)",
        past_advice=advice.strip() or "(기록 없음)",
        price_then=price_then,
        price_now=price_now,
        pct_change=pct,
        direction=direction,
        elapsed_label=_elapsed_label(elapsed_seconds),
        high_low_block=high_low_block,
        tpsl_block=tpsl_block,
        missed_block=missed_block,
        missed_instruction=missed_instruction,
    )

    client = _get_client()
    try:
        reflection_text = _call_llm(client, system_prompt, prompt)
    except Exception as exc:
        return ReflectionResult(
            timestamp=record_ts,
            role=role,
            price_then=price_then,
            price_now=price_now,
            pct_change=pct,
            reflection_text="",
            updated=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    # 메모리에 outcome 기록
    outcome_block = (
        f"[사후 {_elapsed_label(elapsed_seconds)}] "
        f"${price_then:,.2f} → ${price_now:,.2f} ({pct:+.2f}%, {direction})\n"
        f"{reflection_text}"
    )
    updated = memory.update_outcome(record_ts, outcome_block)

    return ReflectionResult(
        timestamp=record_ts,
        role=role,
        price_then=price_then,
        price_now=price_now,
        pct_change=pct,
        reflection_text=reflection_text,
        updated=updated,
    )
