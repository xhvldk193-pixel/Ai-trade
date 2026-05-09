# =============================================
# Claude API 연동 - 매매 시그널 분석 (학습 강화형)
# =============================================
import re
import time
import os as _os
import logging as _logging
from typing import Optional
from config import CLAUDE_API_KEY, CLAUDE_MODEL, DEFAULT_SYMBOL, symbol_to_pair
from indicators import summarize_indicators
from account_context import fetch_account_context, format_account_context
from market_context import fetch_market_context, format_market_context
from macro_fetcher import fetch_macro_context, format_macro_context
from time_utils import now_kst
from agents import run_pipeline, PipelineResult

try:
    from agents.memory import get_memory
except Exception:
    get_memory = None

_memory_logger = _logging.getLogger(__name__)
MEMORY_WRITE_ENABLED = _os.getenv("MEMORY_WRITE_ENABLED", "1").lower() not in ("0", "false", "no")
PAIR_LABEL = symbol_to_pair(DEFAULT_SYMBOL)

SYSTEM_PROMPT = (
    f"당신은 10년 경력의 {PAIR_LABEL} 애널리스트입니다.\n"
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

def _build_context_blob(multi_tf_data: dict):
    indicators_summary = "\n\n".join([summarize_indicators(tf, multi_tf_data[tf]) for tf in ["1d", "4h", "1h", "15m"] if tf in multi_tf_data])
    account_ctx = format_account_context(fetch_account_context())
    market_ctx = format_market_context(fetch_market_context())
    return f"{account_ctx}\n\n{market_ctx}\n\n{indicators_summary}"

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

def analyze_with_claude(multi_tf_data: dict, pipeline: PipelineResult = None):
    from agents import get_anthropic_client
    client = get_anthropic_client()
    
    context_blob = _build_context_blob(multi_tf_data)
    debate_block = pipeline.combined_block if pipeline else ""
    prompt = USER_PROMPT_TEMPLATE.format(
        now_kst=now_kst().strftime("%Y-%m-%d %H:%M"), 
        context_blob=context_blob, 
        debate_block_separator="\n━━━━━━━━━━━━━━━━━━━━\n" if debate_block else "",
        debate_block=debate_block
    )
    
    # 600토큰으로 설정하여 내부 추론 안정성 확보
    max_t = int(_os.getenv("ANALYST_MAX_TOKENS", "600"))
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_t,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )
    raw_text = message.content[0].text
    view, confidence = parse_signal(raw_text)
    
    return {"view": view, "confidence": confidence, "raw_text": raw_text, "levels": parse_trade_levels(raw_text), "prompt": prompt}

def run_full_analysis(multi_tf_data: dict):
    # 분석 전 메모리 자동 정제 실행
    if get_memory:
        mem = get_memory("analyst")
        if mem: mem.cleanup_old_no_outcome_records(days_threshold=3)

    price_at_analysis = float(multi_tf_data["1h"].iloc[-1]["close"])
    pipeline = run_pipeline(
        context_blob=_build_context_blob(multi_tf_data), 
        pair_label=PAIR_LABEL, 
        current_situation="...", 
        price_at_analysis=price_at_analysis
    )
    
    result = analyze_with_claude(multi_tf_data, pipeline)
    
    if MEMORY_WRITE_ENABLED and mem:
        # [학습 강화] 저장 시에는 에이전트 토론 로그를 포함하여 저장
        full_reasoning = f"[DEBATE LOG]\n{pipeline.combined_block}\n\n[FINAL RESULT]\n{result['raw_text']}"
        mem.add_situation(
            situation=result['prompt'][:1000], 
            advice=full_reasoning,
            outcome="", # reflection.py가 채울 예정
            meta={
                "confidence": result['confidence'], "levels": result['levels'],
                "price_at_analysis": price_at_analysis, "view": result['view']
            }
        )
    return result
