# =============================================
# Reflection Agent — 사후 결과로 메모리 업데이트 (학습 & 정제 통합 버전)
# =============================================
from __future__ import annotations

import os
import time
import anthropic
from dataclasses import dataclass, asdict
from typing import Optional

# 설정 및 환경 변수
def _get_client():
    """Anthropic 클라이언트를 가져옵니다."""
    from agents import get_anthropic_client
    return get_anthropic_client()

from config import CLAUDE_API_KEY
from .memory import FinancialSituationMemory, get_memory

# 리플렉션에 사용할 모델 설정
REFLECTION_MODEL = os.getenv("REFLECTION_MODEL", "claude-3-5-sonnet-20240620")

# ── 역할별 시스템 프롬프트 (학습 강화 지침 포함) ────────────────────────────

_BASE_RULES = """
출력 규칙:
1. 결과 판정: 수익 / 손절 / 미진입 성공 / 미진입 실패(기회비용) 중 하나를 반드시 선택.
2. 실패 원인: 기술적 오류, 지표 과신, 거시 변수 중 핵심 이유 기술.
3. 반대 시나리오: 만약 반대 포지션을 잡았다면 어땠을지 1줄 언급.
4. 마크다운 사용 금지, 일반 텍스트로 작성.
5. 마지막 줄은 반드시 '다음 체크리스트:' 로 시작하는 요약."""

_HINDSIGHT_GUARD = """
결과론 금지: 사후 가격을 안다고 해서 '당시 알 수 있었던 신호'라고 치부하지 마세요. 
당시 데이터 기준으로 판단이 합리적이었는지만 평가하세요."""

ROLE_REFLECTION_SYSTEMS: dict[str, str] = {
    "analyst": f"""당신은 BTC 애널리스트 코치입니다. {_HINDSIGHT_GUARD} {_BASE_RULES}""",
    "bull": f"""당신은 Bull Researcher 코치입니다. 상방 근거를 재검토하세요. {_BASE_RULES}""",
    "bear": f"""당신은 Bear Researcher 코치입니다. 하방 근거를 재검토하세요. {_BASE_RULES}""",
    "judge": f"""당신은 심판 판정 코치입니다. 판정이 적절했는지 평가하세요. {_BASE_RULES}""",
    "aggressive": f"""공격적 리스크 코치입니다. 불필요한 리스크였는지 확인하세요. {_BASE_RULES}""",
    "conservative": f"""보수적 리스크 코치입니다. 놓친 기회를 인정하세요. {_BASE_RULES}""",
    "neutral": f"""중립 리스크 코치입니다. 전략의 실효성을 평가하세요. {_BASE_RULES}""",
}

DEFAULT_REFLECTION_SYSTEM = ROLE_REFLECTION_SYSTEMS["analyst"]

# ── 유저 프롬프트 템플릿 ───────────────────────────────────

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
{high_low_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{missed_block}위 정보를 바탕으로 리플렉션을 작성하세요.
{missed_instruction}마지막 줄은 반드시 '다음 체크리스트:' 로 시작하세요.
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
    """Claude API를 호출하여 답변을 받습니다."""
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
    """개별 기록에 대해 리플렉션을 실행합니다."""
    if memory is None: memory = get_memory(role)
    if not CLAUDE_API_KEY:
        return ReflectionResult(record_ts, role, price_then, price_now, 0.0, "", False, "API_KEY_MISSING")

    pct = ((price_now - price_then) / price_then * 100.0) if price_then else 0.0
    direction = "상승" if pct > 0 else "하락"

    high_low_block = f"구간 최고가: ${price_high:,.2f} | 구간 최저가: ${price_low:,.2f}\n" if price_high else ""
    
    # 메타 정보 확인
    _rec = next((r for r in memory.records if r.timestamp == record_ts), None)
    _meta = _rec.meta if _rec else {}
    _conf = _meta.get("confidence", 100)
    
    missed_block = ""
    missed_instruction = ""
    if _conf < 65: # 확신도가 낮아 진입 안 한 케이스
        missed_block = "⚠️ 이 기록은 진입 거부된 케이스입니다. 기회비용을 분석하세요.\n"
        missed_instruction = "관망 판단이 자산을 지켰는지, 아니면 큰 수익을 놓쳤는지 반드시 평가하세요.\n"

    system_prompt = ROLE_REFLECTION_SYSTEMS.get(role, DEFAULT_REFLECTION_SYSTEM)
    prompt = REFLECTION_USER_TEMPLATE.format(
        past_ts=record_ts, role=role, past_situation=situation, past_advice=advice,
        price_then=price_then, price_now=price_now, pct_change=pct, direction=direction,
        elapsed_label=_elapsed_label(elapsed_seconds), high_low_block=high_low_block,
        missed_block=missed_block, missed_instruction=missed_instruction
    )

    try:
        reflection_text = _call_llm(_get_client(), system_prompt, prompt)
        outcome_block = f"[사후 {_elapsed_label(elapsed_seconds)}] {reflection_text}"
        updated = memory.update_outcome(record_ts, outcome_block)
        return ReflectionResult(record_ts, role, price_then, price_now, pct, reflection_text, updated)
    except Exception as exc:
        return ReflectionResult(record_ts, role, price_then, price_now, pct, "", False, str(exc))

def reflect_all():
    """모든 역할의 메모리를 정리하고 리플렉션을 수행하는 통합 입구입니다."""
    roles = ["analyst", "bull", "bear", "judge", "aggressive", "conservative", "neutral"]
    
    for role in roles:
        memory = get_memory(role)
        if not memory: continue
        
        # 1. 결과가 없는(리플렉션 대기 중인) 기록 찾기
        pending = [r for r in memory.records if not r.outcome or len(r.outcome.strip()) == 0]
        
        for rec in pending:
            # 실제 운영 시에는 여기서 현재 가격을 가져와야 함 (현재는 구조적 실행만 담당)
            # reflect_for_role(...) 호출 로직이 이 자리에 위치함
            pass

        # 2. 리플렉션이 끝난 후 3일 지난 쓰레기 데이터 청소
        deleted = memory.cleanup_old_no_outcome_records(days_threshold=3)
        if deleted > 0:
            print(f"[{role}] {deleted}개의 무의미한 기록 정제 완료.")

if __name__ == "__main__":
    reflect_all()
