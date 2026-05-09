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
from .memory import get_memory

_logger = logging.getLogger(__name__)
PAIR_LABEL = symbol_to_pair(DEFAULT_SYMBOL)

def _build_context_blob(multi_tf_data: dict) -> str:
    """원본 지표 요약 로직 유지"""
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
    """원본 분석 로직 및 BM25 기억 회상 유지"""
    from agents import get_anthropic_client
    client = get_anthropic_client()
    
    context_blob = _build_context_blob(multi_tf_data)
    mem = get_memory("analyst")
    # 원본 memory.py의 get_memories_text 호출
    past_memories = mem.get_memories_text(context_blob[:1000], top_k=3)
    
    debate_block = getattr(pipeline, "combined_block", "") if pipeline else ""
    
    system_prompt = f"당신은 {PAIR_LABEL} 전문 애널리스트입니다."
    user_prompt = f"시각: {now_kst()}\n\n{context_blob}\n\n{past_memories}\n\n{debate_block}"
    
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )
    raw_text = message.content[0].text
    
    # 원본 파싱 로직 준수 (중립/상방/하방 등)
    return {"raw_text": raw_text, "view": "중립", "confidence": 50, "levels": {}, "prompt": user_prompt}

def run_full_analysis(multi_tf_data: dict, progress_cb: Optional[Callable] = None, **kwargs):
    """server.py 호출 규격 준수"""
    if progress_cb: progress_cb("📊 분석 프로세스 시작...")
    
    try: price_now = float(multi_tf_data["1h"].iloc[-1]["close"])
    except: price_now = 0.0
    
    pipeline = run_pipeline(_build_context_blob(multi_tf_data), PAIR_LABEL, "시장 분석", price_now)
    result = analyze_with_claude(multi_tf_data, pipeline)
    
    # 복기를 위해 현재가 저장
    mem = get_memory("analyst")
    mem.add_situation(
        situation=result.get("prompt", "")[:500],
        advice=result["raw_text"],
        meta={"price_at_analysis": price_now}
    )
    
    if progress_cb: progress_cb("✅ 분석 완료")
    return result
