from __future__ import annotations
import os
import time
import requests
from dataclasses import dataclass
from typing import Optional
import anthropic
from config import CLAUDE_API_KEY
from .memory import FinancialSituationMemory, get_memory

# 원본 모델 설정 복구
REFLECTION_MODEL = os.getenv("REFLECTION_MODEL", "claude-haiku-4-5")

def _get_client():
    from agents import get_anthropic_client
    return get_anthropic_client()

def get_historical_price_4h(timestamp_unix: float):
    """정확한 4시간 뒤 가격 조회 함수 추가"""
    try:
        target_time = int((timestamp_unix + (4 * 3600)) * 1000)
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": "BTCUSDT", "interval": "1m", "startTime": target_time, "limit": 1}
        res = requests.get(url, params=params, timeout=5).json()
        return float(res[0][4]) if res else None
    except: return None

# 원본의 모든 규칙 및 프롬프트 (건드리지 않음)
_BASE_RULES = """
출력 규칙:
- 마크다운 금지. 일반 텍스트 사용.
- 당시 토론(Short vs Long)의 판단 미스 및 기회비용 집중 분석.
"""

def reflect_all():
    roles = ["analyst", "bull", "bear", "judge", "aggressive", "conservative", "neutral"]
    client = _get_client()
    
    for role in roles:
        memory = get_memory(role)
        now = time.time()
        # 4시간 경과 데이터 추출
        pending = [r for r in memory.records if not r.outcome and (now - r.timestamp_unix > 14400)]
        
        for rec in pending:
            price_then = rec.meta.get("price_at_analysis")
            price_after = get_historical_price_4h(rec.timestamp_unix)
            
            if not price_then or not price_after: continue
            
            diff = ((price_after - price_then) / price_then) * 100
            prompt = f"과거 분석: {rec.advice}\n결과: ${price_then} -> ${price_after} ({diff:+.2f}%)\n토론의 오판 이유를 분석하세요."
            
            try:
                msg = client.messages.create(
                    model=REFLECTION_MODEL,
                    max_tokens=800,
                    system=f"당신은 {role} 트레이딩 코치입니다. {_BASE_RULES}",
                    messages=[{"role": "user", "content": prompt}]
                )
                memory.update_outcome(rec.timestamp, msg.content[0].text)
            except: continue
        
        memory.cleanup_old_no_outcome_records(days_threshold=3)

if __name__ == "__main__":
    reflect_all()
