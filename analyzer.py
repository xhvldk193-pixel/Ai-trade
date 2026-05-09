import re
import os
import logging
from typing import Optional, Callable

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
    mem = get_memory("analyst")
    # 원본 memory.py의 BM25 기능 유지
    past_memories = mem.get_memories_text(context_blob[:1000], top_k=3)
    
    debate_block = getattr(pipeline, "combined_block", "") if pipeline else ""
    
    # 원본 프롬프트 구조 유지
    prompt = f"시각: {now_kst()}\n\n{context_blob}\n\n{past_memories}\n\n{debate_block}"
    
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1000,
        system=f"당신은 {PAIR_LABEL} 전문 애널리스트입니다.",
        messages=[{"role": "user", "content": prompt}]
    )
    raw_text = message.content[0].text
    
    # server.py가 기대하는 반환값 형식 유지
    return {"raw_text": raw_text, "view": "중립", "confidence": 50, "levels": {}, "prompt": prompt}

def run_full_analysis(multi_tf_data: dict, progress_cb: Optional[Callable] = None, **kwargs):
    """server.py 인자 충돌 방지를 위해 **kwargs 유지"""
    if progress_cb: progress_cb("분석 시작...")
    
    price_now = float(multi_tf_data["1h"].iloc[-1]["close"])
    pipeline = run_pipeline(_build_context_blob(multi_tf_data), PAIR_LABEL, "시장 분석", price_now)
    result = analyze_with_claude(multi_tf_data, pipeline)
    
    # [수정] 복기용 가격 데이터 추가 저장
    mem = get_memory("analyst")
    mem.add_situation(
        situation=result.get("prompt", "")[:500],
        advice=result["raw_text"],
        meta={"price_at_analysis": price_now}
    )
    
    if progress_cb: progress_cb("분석 완료")
    return result
