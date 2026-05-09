import re
import os
import logging
from typing import Optional, Callable, Any

# 원본 config 및 모듈들 (절대 경로 유지)
from config import CLAUDE_API_KEY, CLAUDE_MODEL, DEFAULT_SYMBOL, symbol_to_pair
from indicators import summarize_indicators
from account_context import fetch_account_context, format_account_context
from market_context import fetch_market_context, format_market_context
from macro_fetcher import fetch_macro_context, format_macro_context
from time_utils import now_kst
from agents import run_pipeline, PipelineResult

# [수정] 원본 memory.py의 get_memory를 가져오는 정확한 경로
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
    
    prompt = f"시각: {now_kst()}\n\n{context_blob}\n\n{past_memories}\n\n{debate_block}"
    
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1000,
        system=f"당신은 {PAIR_LABEL} 전문 애널리스트입니다.",
        messages=[{"role": "user", "content": prompt}]
    )
    raw_text = message.content[0].text
    return {"raw_text": raw_text, "view": "중립", "confidence": 50, "levels": {}, "prompt": prompt}

def run_full_analysis(multi_tf_data: dict, progress_cb: Optional[Callable] = None, **kwargs):
    if progress_cb: progress_cb("분석 시작...")
    
    try: price_now = float(multi_tf_data["1h"].iloc[-1]["close"])
    except: price_now = 0.0
    
    pipeline = run_pipeline(_build_context_blob(multi_tf_data), PAIR_LABEL, "시장 분석", price_now)
    result = analyze_with_claude(multi_tf_data, pipeline)
    
    # 사후 복기를 위해 현재 가격 저장
    mem_manager = get_memory()
    mem_manager.get("analyst").add_situation(
        situation=result.get("prompt", "")[:500],
        advice=result["raw_text"],
        meta={"price_at_analysis": price_now}
    )
    
    if progress_cb: progress_cb("분석 완료")
    return result
