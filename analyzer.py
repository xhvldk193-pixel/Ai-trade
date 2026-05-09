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

# 임포트 경로 호환성
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
    # 과거 기억 회상 (이전 리플렉션 결과가 포함된 storage_advice를 읽어옴)
    past_memories = mem_manager.get_memories_text("analyst", context_blob[:1000], top_k=3)
    
    debate_block = getattr(pipeline, "combined_block", "") if pipeline else ""
    
    system_prompt = f"""당신은 {PAIR_LABEL} 전문 애널리스트입니다. 
토론 내용을 바탕으로 결론을 내리되, 출력은 반드시 아래 형식만 지키세요. 
형식: [관점 / 확신도 / 진입가 / 손절가 / 목표가 / 레버리지]"""

    user_prompt = f"시각: {now_kst()}\n\n{context_blob}\n\n{past_memories}\n\n{debate_block}"
    
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=300, # 사용자님 요청대로 토큰 절약
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )
    raw_text = message.content[0].text
    
    # [중요] 사용자에게는 안 보이지만, 메모리(JSONL)에는 토론 내용을 꽉 채워 저장
    # 나중에 BM25가 이 내용을 읽어와서 past_memories에 넣어줍니다.
    storage_advice = f"결론: {raw_text}\n\n[상세 논거]\n{debate_block[:800]}"

    return {
        "raw_text": raw_text, 
        "storage_advice": storage_advice,
        "view": "중립", 
        "confidence": 50, 
        "levels": {}, 
        "prompt": user_prompt
    }

def run_full_analysis(multi_tf_data: dict, *args, **kwargs):
    """
    [TypeError 해결] server.py가 progress_cb를 위치 인자로 주든, 키워드 인자로 주든 
    중복 에러 없이 받아내도록 구조를 변경함.
    """
    # progress_cb 추출 로직
    progress_cb = kwargs.get("progress_cb") or (args[0] if args else None)
    
    if progress_cb and callable(progress_cb):
        progress_cb("📊 분석 및 토론 파이프라인 시작...")
    
    try:
        price_now = float(multi_tf_data["1h"].iloc[-1]["close"])
    except:
        price_now = 0.0
    
    # 1. Bull/Bear/Judge 토론 (내부 사고 과정)
    pipeline = run_pipeline(_build_context_blob(multi_tf_data), PAIR_LABEL, "시장 분석", price_now)
    
    # 2. 최종 요약 (사용자용 출력)
    result = analyze_with_claude(multi_tf_data, pipeline)
    
    # 3. 메모리 저장 (학습용 고밀도 데이터 저장)
    mem_manager = get_memory()
    mem_manager.get("analyst").add_situation(
        situation=result.get("prompt", "")[:500],
        advice=result.get("storage_advice", result["raw_text"]), # raw_text 대신 고밀도 데이터 저장
        meta={"price_at_analysis": price_now} 
    )
    
    if progress_cb and callable(progress_cb):
        progress_cb("✅ 분석 완료 및 고밀도 데이터 저장 성공")
        
    return result
