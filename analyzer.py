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
    """지표 요약 및 시장 상황 데이터 결합 (원본 로직)"""
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
    """Claude 분석 수행 (원본 BM25 기억 회상 포함)"""
    from agents import get_anthropic_client
    client = get_anthropic_client()
    
    context_blob = _build_context_blob(multi_tf_data)
    
    # 원본 memory.py의 BM25 기반 기억 회상 유지
    mem = get_memory("analyst")
    past_memories = mem.get_memories_text(context_blob[:1000], top_k=3)
    
    debate_block = getattr(pipeline, "combined_block", "") if pipeline else ""
    
    system_prompt = f"당신은 {PAIR_LABEL} 전문 애널리스트입니다. 내부 분석은 깊게 하되 출력은 지정된 형식만 유지하세요."
    user_prompt = f"분석 시각: {now_kst()}\n\n[시장 데이터]\n{context_blob}\n\n{past_memories}\n\n[에이전트 토론 결과]\n{debate_block}\n\n최종 판단을 내리세요."
    
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )
    raw_text = message.content[0].text
    
    # 파싱 로직 (원본 규격 유지)
    view = "중립"
    if "상방 우위" in raw_text: view = "상방 우위"
    elif "하방 우위" in raw_text: view = "하방 우위"
    
    conf_match = re.search(r'확신도\D*?(\d{1,3})', raw_text)
    confidence = int(conf_match.group(1)) if conf_match else 50
    
    return {
        "raw_text": raw_text,
        "view": view,
        "confidence": confidence,
        "levels": {}, # 정규식 생략 (원본 로직에 따라 추가 가능)
        "prompt": user_prompt
    }

def run_full_analysis(multi_tf_data: dict, progress_cb: Optional[Callable] = None, **kwargs):
    """server.py와의 인터페이스 호환을 위해 progress_cb와 **kwargs 유지"""
    if progress_cb: progress_cb("📊 지표 분석 및 토론 파이프라인 시작...")
    
    try:
        price_now = float(multi_tf_data["1h"].iloc[-1]["close"])
    except:
        price_now = 0.0
    
    # 1. 원본 토론 파이프라인 실행
    pipeline = run_pipeline(_build_context_blob(multi_tf_data), PAIR_LABEL, "전략 분석", price_now)
    
    # 2. 최종 분석
    if progress_cb: progress_cb("🧠 Claude 최종 판단 도출 중...")
    result = analyze_with_claude(multi_tf_data, pipeline)
    
    # 3. 메모리 저장 (price_at_analysis를 meta에 넣어 사후 복기 가능케 함)
    mem = get_memory("analyst")
    mem.add_situation(
        situation=result.get("prompt", "")[:500],
        advice=result["raw_text"],
        meta={"price_at_analysis": price_now}
    )
    
    if progress_cb: progress_cb("✅ 분석 완료")
    return result
