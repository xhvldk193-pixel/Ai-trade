import re
import os
import logging
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
    from memory import get_memory

_logger = logging.getLogger(__name__)
PAIR_LABEL = symbol_to_pair(DEFAULT_SYMBOL)

def _build_context_blob(multi_tf_data: dict) -> str:
    indicators_summary = "\n\n".join(
        [summarize_indicators(tf, multi_tf_data[tf]) for tf in ["1d", "4h", "1h", "15m"] if tf in multi_tf_data]
    )
    account_ctx = format_account_context(fetch_account_context())
    market_ctx = format_market_context(fetch_market_context())
    try:
        macro_ctx = format_macro_context(fetch_macro_context())
    except: macro_ctx = ""
    return f"{account_ctx}\n\n{market_ctx}\n\n{macro_ctx}\n\n{indicators_summary}"

def analyze_with_claude(multi_tf_data: dict, pipeline: Optional[PipelineResult] = None):
    from agents import get_anthropic_client
    client = get_anthropic_client()
    
    context_blob = _build_context_blob(multi_tf_data)
    mem_manager = get_memory() 
    past_memories = mem_manager.get_memories_text("analyst", context_blob[:1000], top_k=3)
    
    debate_block = getattr(pipeline, "combined_block", "") if pipeline else ""
    
    system_prompt = f"""당신은 {PAIR_LABEL} 전문 애널리스트입니다. 
토론 내용을 바탕으로 결론을 내리되, 출력은 반드시 아래 형식만 지키세요. 
형식: [관점 / 확신도 / 진입가 / 손절가 / 목표가 / 레버리지]"""

    user_prompt = f"시각: {now_kst()}\n\n{context_blob}\n\n{past_memories}\n\n{debate_block}"
    
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=300, 
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )
    raw_text = message.content[0].text
    
    # [학습 보호 로직] 출력은 짧지만, 메모리에는 토론 내용을 포함해 저장
    storage_advice = f"최종 결론: {raw_text}\n\n[당시 분석 논거]\n{debate_block[:800]}"

    return {
        "raw_text": raw_text, 
        "storage_advice": storage_advice,
        "view": "중립", 
        "confidence": 50, 
        "levels": {}, 
        "prompt": user_prompt
    }

def run_full_analysis(multi_tf_data: dict, progress_cb: Optional[Callable] = None, **kwargs):
    if progress_cb: progress_cb("📊 분석 프로세스 시작...")
    
    try: price_now = float(multi_tf_data["1h"].iloc[-1]["close"])
    except: price_now = 0.0
    
    pipeline = run_pipeline(_build_context_blob(multi_tf_data), PAIR_LABEL, "시장 분석", price_now)
    result = analyze_with_claude(multi_tf_data, pipeline)
    
    mem_manager = get_memory()
    # advice 항목에 storage_advice를 넣어 나중에 회상 시 논리가 살아있게 함
    mem_manager.get("analyst").add_situation(
        situation=result.get("prompt", "")[:500],
        advice=result.get("storage_advice", result["raw_text"]),
        meta={"price_at_analysis": price_now} 
    )
    
    if progress_cb: progress_cb("✅ 분석 및 고밀도 데이터 저장 완료")
    return result
