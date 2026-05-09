# =============================================
# Claude API 연동 - 매매 시그널 분석 (안정성 극대화 버전)
# =============================================
import re
import os as _os
import logging as _logging
from typing import Optional, Callable, Any

from config import CLAUDE_API_KEY, CLAUDE_MODEL, DEFAULT_SYMBOL, symbol_to_pair
from indicators import summarize_indicators
from account_context import fetch_account_context, format_account_context
from market_context import fetch_market_context, format_market_context
from macro_fetcher import fetch_macro_context, format_macro_context
from time_utils import now_kst
from agents import run_pipeline, PipelineResult

try:
    from agents.memory import get_memory
except ImportError:
    get_memory = None

_memory_logger = _logging.getLogger(__name__)
MEMORY_WRITE_ENABLED = _os.getenv("MEMORY_WRITE_ENABLED", "1").lower() not in ("0", "false", "no")
PAIR_LABEL = symbol_to_pair(DEFAULT_SYMBOL)

SYSTEM_PROMPT = (
    f"당신은 10년 경력의 {PAIR_LABEL} 전문 트레이더이자 애널리스트입니다.\n"
    "내부적으로는 모든 데이터를 심층 분석하되, 출력은 지정된 파라미터만 간결하게 제시하세요.\n"
    "확신도는 50~95 전 구간을 활용하십시오.\n"
)

USER_PROMPT_TEMPLATE = """분석 기준 시각: {now_kst} (KST)
{context_blob}
{debate_block_separator}{debate_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[지시사항]
1. 내부 분석 단계: 위 데이터를 바탕으로 추세, 지지/저항, 심리를 정밀하게 추론하세요.
2. 최종 출력: 분석 과정은 생략하고 아래 형식으로만 응답하세요.

📊 관점: [상방 우위 / 하방 우위 / 중립]
💯 확신도: [숫자]%

🤖 매매 파라미터
• 진입가: $[숫자]
• 손절가: $[숫자]
• 목표가: $[숫자]
• 권장 레버리지: [숫자]배
"""

def _build_context_blob(multi_tf_data: dict) -> str:
    indicators_summary = "\n\n".join(
        [summarize_indicators(tf, multi_tf_data[tf]) for tf in ["1d", "4h", "1h", "15m"] if tf in multi_tf_data]
    )
    account_ctx = format_account_context(fetch_account_context())
    market_ctx = format_market_context(fetch_market_context())
    
    # 누락되었던 거시경제(macro) 데이터도 안전하게 통합
    try:
        macro_ctx = format_macro_context(fetch_macro_context())
    except Exception:
        macro_ctx = ""
        
    return f"{account_ctx}\n\n{market_ctx}\n\n{macro_ctx}\n\n{indicators_summary}"

def parse_signal(text: str):
    sig_match = re.search(r'📊\s*관점\s*[:：]\s*(상방 우위|하방 우위|중립)', text)
    view = sig_match.group(1) if sig_match else "중립"
    conf_match = re.search(r'확신도\D*?(\d{1,3})', text)
    confidence = min(int(conf_match.group(1)), 100) if conf_match else 50
    return view, confidence

def parse_trade_levels(text: str) -> dict:
    def _get_val(label: str):
        m = re.search(rf'{label}\s*[:：]\s*\$([\d,]+(?:\.\d+)?)', text)
        return float(m.group(1).replace(',', '')) if m else None
    return {"entry": _get_val("진입가"), "stop": _get_val("손절가"), "target": _get_val("목표가")}

def analyze_with_claude(multi_tf_data: dict, pipeline: Optional[PipelineResult] = None):
    from agents import get_anthropic_client
    client = get_anthropic_client()
    
    context_blob = _build_context_blob(multi_tf_data)
    
    # 파이프라인 데이터가 없을 경우를 대비한 안전 장치 (AttributeError 방지)
    debate_block = getattr(pipeline, "combined_block", "") if pipeline else ""
    
    prompt = USER_PROMPT_TEMPLATE.format(
        now_kst=now_kst().strftime("%Y-%m-%d %H:%M"), 
        context_blob=context_blob, 
        debate_block_separator="\n━━━━━━━━━━━━━━━━━━━━\n" if debate_block else "",
        debate_block=debate_block
    )
    
    max_t = int(_os.getenv("ANALYST_MAX_TOKENS", "600"))
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_t,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )
    raw_text = message.content[0].text
    view, confidence = parse_signal(raw_text)
    
    return {
        "view": view, 
        "confidence": confidence, 
        "raw_text": raw_text, 
        "levels": parse_trade_levels(raw_text), 
        "prompt": prompt
    }

# **kwargs를 추가하여, 메인 봇이 생각지도 못한 추가 인자를 보내더라도 에러가 나지 않도록 철벽 방어!
def run_full_analysis(multi_tf_data: dict, progress_cb: Optional[Callable[[str], None]] = None, **kwargs) -> dict:
    """
    고도화된 분석 루틴.
    :param multi_tf_data: 타임프레임별 데이터
    :param progress_cb: 진행 상황을 알리는 콜백 함수
    :param kwargs: 예상치 못한 추가 인자를 흡수하여 프로그램 크래시 방지
    """
    def notify(msg: str):
        if progress_cb: 
            progress_cb(msg)
        _logging.info(msg)

    notify("📊 분석 시작 및 메모리 점검 중...")
    mem = None
    if get_memory:
        try:
            mem = get_memory("analyst")
            if mem: 
                mem.cleanup_old_no_outcome_records(days_threshold=3)
        except Exception as e:
            _logging.warning(f"메모리 초기화 에러 (분석은 계속 진행): {e}")

    # 현재가 추출 로직 (1h 데이터의 마지막 종가)
    try:
        price_at_analysis = float(multi_tf_data["1h"].iloc[-1]["close"])
    except Exception as e:
        _logging.warning(f"현재가 추출 실패, 0으로 대체: {e}")
        price_at_analysis = 0.0
    
    notify("🗣️ 에이전트 토론(Pipeline) 진행 중...")
    try:
        pipeline = run_pipeline(
            context_blob=_build_context_blob(multi_tf_data), 
            pair_label=PAIR_LABEL, 
            current_situation="전략적 포지션 분석", 
            price_at_analysis=price_at_analysis
        )
    except Exception as e:
        notify(f"⚠️ 토론 파이프라인 에러 (토론 건너뛰고 단독 분석 진행): {e}")
        pipeline = None
    
    notify("🧠 Claude 최종 판단 도출 중...")
    result = analyze_with_claude(multi_tf_data, pipeline)
    
    # 분석 결과를 메모리에 저장
    if MEMORY_WRITE_ENABLED and mem:
        notify("💾 분석 결과 메모리 저장 중...")
        debate_log = getattr(pipeline, "combined_block", "토론 생략됨") if pipeline else "토론 생략됨"
        full_reasoning = f"[DEBATE LOG]\n{debate_log}\n\n[FINAL RESULT]\n{result['raw_text']}"
        
        try:
            mem.add_situation(
                situation=result['prompt'][:1000], 
                advice=full_reasoning,
                outcome="",
                meta={
                    "confidence": result['confidence'], 
                    "levels": result['levels'],
                    "price_at_analysis": price_at_analysis, 
                    "view": result['view']
                }
            )
        except Exception as e:
            _logging.warning(f"메모리 기록 실패 (매매 자체에는 영향 없음): {e}")
    
    notify("✅ 분석 완료")
    return result
