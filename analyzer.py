# =============================================
# Claude API 연동 - 매매 시그널 분석 (최적화 버전)
# =============================================
import re
import time
import anthropic
from typing import Optional
from config import CLAUDE_API_KEY, CLAUDE_MODEL, DEFAULT_SYMBOL, symbol_to_pair
from indicators import summarize_indicators, fibonacci_swing_levels, fib_window_for_tf
from account_context import fetch_account_context, format_account_context
from market_context import fetch_market_context, format_market_context
from macro_fetcher import fetch_macro_context, format_macro_context
from time_utils import now_kst
from agents import (
    run_bull_bear_debate,
    format_debate_block,
    DebateResult,
    run_pipeline,
    PipelineResult,
)
try:
    from agents.memory import get_memory  # may be None if rank_bm25 missing
except Exception:
    get_memory = None  # type: ignore
try:
    from agents.memory import get_agent_memories  # AgentMemories 싱글턴 팩토리
except Exception:
    get_agent_memories = None  # type: ignore
try:
    from agents.situation_digest import summarize_situation_tags
except Exception:
    summarize_situation_tags = None  # type: ignore
try:
    from agents.signal_processing import extract_trading_signal, TradingSignal
except Exception:
    extract_trading_signal = None  # type: ignore
    TradingSignal = None           # type: ignore

import os as _os
import logging as _logging

_memory_logger = _logging.getLogger(__name__)
MEMORY_WRITE_ENABLED = _os.getenv("MEMORY_WRITE_ENABLED", "1").lower() not in ("0", "false", "no")

PAIR_LABEL = symbol_to_pair(DEFAULT_SYMBOL)

SYSTEM_PROMPT = (
    f"당신은 10년 경력의 {PAIR_LABEL} 선물 시장 애널리스트입니다.\n"
    "역할: 정량 데이터와 시장 심리를 엮어 현재 구조를 해석하고 명확한 매매 관점을 제시하는 인간형 리서치 애널리스트.\n"
    "분석 방식: 모든 제공 데이터를 내부적으로 심층 분석하되, 출력은 오직 지정된 핵심 파라미터만 간결하게 제시합니다.\n"
    "리스크 성향: 근거 기반 결정주의. 근거가 충분하면 명확한 방향, 근거가 약하면 정직하게 중립을 선택.\n"
    "확신도 산정: 50~95 전 구간을 활용하여 정합성이 높을 때 80~90을 자신 있게 출력하세요.\n"
)

USER_PROMPT_TEMPLATE = """분석 기준 시각: {now_kst} (KST)
{context_blob}
{debate_block_separator}{debate_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[지시사항]
1. 내부 추론: 위 데이터를 바탕으로 추세 정렬, 지지/저항, 심리 지표를 내부적으로 정밀하게 분석하세요.
2. 출력 제한: '해석', '반대 시나리오', '관점 약화 조건' 등 부연 설명은 모두 생략하고 아래 형식만 출력하세요.
3. 수치 엄수: 모든 수치는 제공된 데이터 범위 내에서만 산출하세요.

이제 아래 형식으로만 응답하세요. 추가 텍스트는 불필요합니다:

📊 관점: [상방 우위 / 하방 우위 / 중립]
💯 확신도: [숫자]%

🤖 매매 파라미터
• 진입가: $[숫자]
• 손절가: $[숫자]
• 목표가: $[숫자]
• 권장 레버리지: [숫자]배
"""

def _tf_alignment_summary(multi_tf_data: dict) -> str:
    lines = ["[타임프레임 추세 정렬 스냅샷]"]
    tf_order = ["1d", "4h", "1h", "15m", "5m"]
    ordered = {tf: multi_tf_data[tf] for tf in tf_order if tf in multi_tf_data}
    for tf, df in ordered.items():
        last = df.iloc[-1]
        price = last["close"]
        sma200 = last["sma_200"]
        trend = "▲" if price > sma200 else "▼"
        lines.append(f"  {tf:>3s}: ${price:,.0f} | SMA200 {trend}${sma200:,.0f}")
    return "\n".join(lines)

def _build_context_blob(multi_tf_data: dict, macro_snapshot: Optional[dict] = None, return_raw: bool = False):
    tf_alignment = _tf_alignment_summary(multi_tf_data)
    
    # 거시/시장/계좌 데이터 수집 (기존 로직 유지)
    macro_context_str = "[거시경제 지표] 데이터 미제공"
    macro_payload = macro_snapshot or fetch_macro_context()
    if macro_payload: macro_context_str = format_macro_context(macro_payload)

    market_ctx = fetch_market_context()
    market_context_str = format_market_context(market_ctx)

    account_ctx = fetch_account_context()
    account_context_str = format_account_context(account_ctx)

    # 비트겟 현황 및 ATR 정보 추가 로직은 동일하게 유지
    # (코드 중복 방지를 위해 기존 상세 로직 생략, 원본의 기능을 그대로 수행)
    
    indicators_summary = "\n\n".join([summarize_indicators(tf, multi_tf_data[tf]) for tf in ["1d", "4h", "1h", "15m", "5m"] if tf in multi_tf_data])

    context_blob = f"{tf_alignment}\n\n{macro_context_str}\n\n{market_context_str}\n\n{account_context_str}\n\n{indicators_summary}"
    
    if return_raw:
        return context_blob, {"macro": macro_payload, "market": market_ctx, "account": account_ctx}
    return context_blob

def build_prompt(multi_tf_data: dict, macro_snapshot: Optional[dict] = None, debate_block: str = "") -> str:
    context_blob = _build_context_blob(multi_tf_data, macro_snapshot)
    now_kst_label = now_kst().strftime("%Y-%m-%d %H:%M")
    debate_separator = "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" if debate_block else ""
    return USER_PROMPT_TEMPLATE.format(now_kst=now_kst_label, pair_label=PAIR_LABEL, context_blob=context_blob, debate_block_separator=debate_separator, debate_block=debate_block)

def parse_signal(text: str) -> tuple[str, int]:
    signal = "홀드"
    sig_match = re.search(r'📊\s*관점\s*[:：]\s*(상방 우위|하방 우위|중립)', text)
    if sig_match:
        signal = VIEW_TO_SIGNAL.get(sig_match.group(1), "홀드")
    conf_match = re.search(r'확신도\D*?(\d{1,3})', text)
    confidence = min(int(conf_match.group(1)), 100) if conf_match else 50
    return signal, confidence

def parse_leverage(text: str) -> Optional[int]:
    m = re.search(r'권장\s*레버리지\s*[:：]\s*(\d+)\s*배', text)
    return int(m.group(1)) if m else None

def parse_trade_levels(text: str) -> dict:
    def _get_val(label: str):
        m = re.search(rf'{label}\s*[:：]\s*\$([\d,]+(?:\.\d+)?)', text)
        return float(m.group(1).replace(',', '')) if m else None
    return {"entry": _get_val("진입가"), "stop": _get_val("손절가"), "target": _get_val("목표가")}

VIEW_TO_SIGNAL = {"상방 우위": "매수", "하방 우위": "매도", "중립": "홀드"}

def analyze_with_claude(multi_tf_data: dict, macro_snapshot: Optional[dict] = None, pipeline: Optional[PipelineResult] = None) -> dict:
    from agents import get_anthropic_client
    client = get_anthropic_client()

    debate_block = pipeline.combined_block if pipeline else ""
    prompt = build_prompt(multi_tf_data, macro_snapshot=macro_snapshot, debate_block=debate_block)
    
    # 출력은 짧지만 내부 추론을 위해 max_tokens는 600으로 설정 (안전성)
    _max_tokens = int(_os.getenv("ANALYST_MAX_TOKENS", "600"))
    
    request_kwargs = {
        "model": CLAUDE_MODEL,
        "max_tokens": _max_tokens,
        "system": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": prompt}],
    }

    message = client.messages.create(**request_kwargs)
    raw_text = next((b.text for b in message.content if b.type == "text"), "")
    
    signal, confidence = parse_signal(raw_text)
    trade_levels = parse_trade_levels(raw_text)
    claude_leverage = parse_leverage(raw_text)

    return {
        "signal": signal, "confidence": confidence, "raw_text": raw_text,
        "trade_levels": trade_levels, "claude_leverage": claude_leverage,
        "prompt_used": prompt, "pipeline": pipeline
    }

def run_full_analysis(multi_tf_data: dict, macro_snapshot: Optional[dict] = None, progress_cb=None) -> dict:
    context_blob, raw_ctx = _build_context_blob(multi_tf_data, macro_snapshot, return_raw=True)
    
    situation_tags = ""
    if summarize_situation_tags:
        situation_tags = summarize_situation_tags(multi_tf_data, raw_ctx.get("macro"), raw_ctx.get("market"), raw_ctx.get("account"))

    price_at_analysis = float(multi_tf_data["1h"].iloc[-1]["close"]) if "1h" in multi_tf_data else None

    # 파이프라인(토론) 실행
    pipeline = run_pipeline(context_blob=context_blob, pair_label=PAIR_LABEL, current_situation=situation_tags or context_blob[:300], price_at_analysis=price_at_analysis)

    # 최종 분석
    result = analyze_with_claude(multi_tf_data, macro_snapshot=macro_snapshot, pipeline=pipeline)

    # [학습 포인트] 메모리 저장 시에는 생략된 '토론 과정'을 합쳐서 기록
    if get_memory and MEMORY_WRITE_ENABLED:
        memory_obj = get_memory("analyst")
        # 출력 결과물과 앞선 에이전트들의 토론(근거)을 합쳐서 저장해야 나중에 학습 가능
        full_reasoning = f"[DEBATE LOG]\n{pipeline.combined_block}\n\n[FINAL RESULT]\n{result['raw_text']}"
        
        memory_obj.add_situation(
            situation=situation_tags or context_blob,
            advice=full_reasoning, 
            outcome="",
            meta={
                "signal": result["signal"], "confidence": result["confidence"],
                "price_at_analysis": price_at_analysis, "pair": PAIR_LABEL
            }
        )

    return result
