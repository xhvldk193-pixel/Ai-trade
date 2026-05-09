import re
import os
import logging
import time
from typing import Optional, Callable, Any

# [1] 필수 모듈 임포트 (NameError 방지)
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
    
    # [2] 출력은 사용자님 요청대로 초간결하게 (토큰 절약)
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
    
    # [3] 저장(학습)은 고밀도로 (내부 사고 과정 보존)
    storage_advice = f"결론: {raw_text}\n\n[당시 상세 논거]\n{debate_block[:800]}"

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
    [4] 모든 인터페이스 에러(Multiple values, detail 인자 누락) 강제 해결
    """
    progress_cb = kwargs.get("progress_cb") or (args[0] if args else None)
    
    def safe_progress(msg: str, detail: str = ""):
        if progress_cb and callable(progress_cb):
            try:
                progress_cb(msg, detail) # detail을 요구하는 서버 대응
            except TypeError:
                try:
                    progress_cb(msg) # 메시지만 요구하는 서버 대응
                except:
                    pass

    safe_progress("📊 분석 엔진 가동", "데이터 로딩 및 메모리 스캔")
    
    try:
        price_now = float(multi_tf_data["1h"].iloc[-1]["close"])
    except:
        price_now = 0.0
    
    # 지능을 유지하는 토론 엔진
    pipeline = run_pipeline(_build_context_blob(multi_tf_data), PAIR_LABEL, "시장 분석", price_now)
    
    safe_progress("🧠 전략 수립 중", "Claude 3.5 Sonnet 연산 중")
    result = analyze_with_claude(multi_tf_data, pipeline)
    
    # 메모리에 '진짜 논리'를 저장 (망가진 학습 복구)
    mem_manager = get_memory()
    mem_manager.get("analyst").add_situation(
        situation=result.get("prompt", "")[:500],
        advice=result.get("storage_advice", result["raw_text"]),
        meta={"price_at_analysis": price_now} 
    )
    
    safe_progress("✅ 분석 완료", "데이터베이스 저장 및 전송 완료")
    return result
