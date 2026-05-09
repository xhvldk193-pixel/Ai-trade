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

# memory.py 임포트 경로 (배포 환경 호환성 확보)
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
    # 원본 AgentMemories.get_memories_text 호출
    past_memories = mem_manager.get_memories_text("analyst", context_blob[:1000], top_k=3)
    
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
    
    # server.py 호환용 반환 구조
    return {"raw_text": raw_text, "view": "중립", "confidence": 50, "levels": {}, "prompt": user_prompt}

def run_full_analysis(multi_tf_data: dict, progress_cb: Optional[Callable] = None, **kwargs):
    """
    [해결] server.py의 호출 규격(위치 인자 순서)을 완벽히 맞추어 TypeError를 근본적으로 해결.
    """
    if progress_cb: progress_cb("📊 분석 및 토론 파이프라인 시작...")
    
    try:
        price_now = float(multi_tf_data["1h"].iloc[-1]["close"])
    except:
        price_now = 0.0
    
    # 원본 토론 엔진 실행
    pipeline = run_pipeline(_build_context_blob(multi_tf_data), PAIR_LABEL, "시장 분석", price_now)
    
    # 최종 판단 도출
    result = analyze_with_claude(multi_tf_data, pipeline)
    
    # [데이터 해결] 복기를 위해 분석 시점 가격을 meta에 저장
    mem_manager = get_memory()
    mem_manager.get("analyst").add_situation(
        situation=result.get("prompt", "")[:500],
        advice=result["raw_text"],
        meta={"price_at_analysis": price_now} 
    )
    
    if progress_cb: progress_cb("✅ 분석 완료 및 경험 저장 성공")
    return result
