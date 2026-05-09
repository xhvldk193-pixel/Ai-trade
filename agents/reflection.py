from __future__ import annotations
import os
import time
import requests
from dataclasses import dataclass
from typing import Optional
import anthropic
from config import CLAUDE_API_KEY

try:
    from agents.memory import get_memory
except ImportError:
    from memory import get_memory

REFLECTION_MODEL = os.getenv("REFLECTION_MODEL", "claude-haiku-4-5")

def _get_client():
    from agents import get_anthropic_client
    return get_anthropic_client()

def get_historical_price_4h(timestamp_unix: float):
    try:
        target_time = int((timestamp_unix + 14400) * 1000)
        url = "https://api.binance.com/api/v3/klines"
        res = requests.get(url, params={"symbol":"BTCUSDT", "interval":"1m", "startTime":target_time, "limit":1}, timeout=5).json()
        return float(res[0][4])
    except: return None

# ── 원본 프롬프트 및 규칙 (100% 보존) ────────────────────────────
_BASE_RULES = """
출력 규칙:
- 마크다운(**,##,---), HTML 금지. 일반 텍스트 + 최소 이모지.
- 300~500자. 장황함 금지.
- 마지막 줄은 반드시 '다음 체크리스트:' 로 시작하는 1~2줄 요약."""

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

DEFAULT_REFLECTION_SYSTEM = ROLE_REFLECTION_SYSTEMS["analyst"]

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
    timestamp: str
    role: str
    price_then: float
    price_now: float
    pct_change: float
    reflection_text: str
    updated: bool
    error: Optional[str] = None

def _call_llm(client: anthropic.Anthropic, system: str, user: str) -> str:
    msg = client.messages.create(
        model=REFLECTION_MODEL,
        max_tokens=800,
        system=[{"type": "text", "text": system}],
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text.strip()

def _elapsed_label(seconds: float) -> str:
    if seconds < 3600: return f"{seconds / 60:.0f}분"
    if seconds < 86400: return f"{seconds / 3600:.1f}시간"
    return f"{seconds / 86400:.1f}일"

def reflect_for_role(role: str, record_ts: str, situation: str, advice: str, price_then: float, price_now: float, elapsed_seconds: float, memory=None):
    if memory is None: memory = get_memory().get(role)
    if not CLAUDE_API_KEY: return None

    pct = ((price_now - price_then) / price_then * 100.0) if price_then else 0
    direction = "상승" if pct > 0 else "하락"
    
    # 원본 TP/SL 평가 로직
    rec = next((r for r in memory.records if r.timestamp == record_ts), None)
    meta = rec.meta if rec else {}
    tl = meta.get("trade_levels", {})
    tpsl_block = ""
    if tl.get("target") or tl.get("stop"):
        tpsl_block = f"\n목표가(TP): ${tl.get('target',0):,.2f} / 손절가(SL): ${tl.get('stop',0):,.2f}"

    system_prompt = ROLE_REFLECTION_SYSTEMS.get(role, DEFAULT_REFLECTION_SYSTEM)
    prompt = REFLECTION_USER_TEMPLATE.format(
        past_ts=record_ts, role=role, past_situation=situation[:500], past_advice=advice[:500],
        price_then=price_then, price_now=price_now, pct_change=pct, direction=direction,
        elapsed_label=_elapsed_label(elapsed_seconds), high_low_block="", tpsl_block=tpsl_block,
        missed_block="", missed_instruction=""
    )

    client = _get_client()
    try:
        reflection_text = _call_llm(client, system_prompt, prompt)
        outcome_block = f"[사후 {_elapsed_label(elapsed_seconds)}] ${price_then:,.2f} -> ${price_now:,.2f} ({pct:+.2f}%)\n{reflection_text}"
        memory.update_outcome(record_ts, outcome_block)
    except: pass

def reflect_all():
    mem_manager = get_memory()
    roles = ["analyst", "bull", "bear", "judge", "aggressive", "conservative", "neutral"]
    for role in roles:
        memory = mem_manager.get(role)
        now = time.time()
        pending = [r for r in memory.records if not r.outcome and (now - r.timestamp_unix > 14400)]
        for rec in pending:
            price_then = rec.meta.get("price_at_analysis")
            price_after = get_historical_price_4h(rec.timestamp_unix)
            if price_then and price_after:
                reflect_for_role(role, rec.timestamp, rec.situation, rec.advice, price_then, price_after, 14400, memory)
