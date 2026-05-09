import os
import time
import requests
from typing import Optional
from .memory import get_memory

# 원본 모델 설정 유지
REFLECTION_MODEL = os.getenv("REFLECTION_MODEL", "claude-haiku-4-5")

def _get_client():
    from agents import get_anthropic_client
    return get_anthropic_client()

def get_historical_price_4h(timestamp_unix: float):
    """4시간 뒤 가격을 바이낸스에서 조회"""
    try:
        target_time = int((timestamp_unix + (4 * 3600)) * 1000)
        url = "https://api.binance.com/api/v3/klines"
        res = requests.get(url, params={"symbol":"BTCUSDT", "interval":"1m", "startTime":target_time, "limit":1}).json()
        return float(res[0][4])
    except: return None

def reflect_all():
    client = _get_client()
    roles = ["analyst", "bull", "bear", "judge", "aggressive", "conservative", "neutral"]
    
    for role in roles:
        memory = get_memory(role)
        now = time.time()
        # 4시간 경과 & 복기 없는 데이터만
        pending = [r for r in memory.records if not r.outcome and (now - r.timestamp_unix > 14400)]
        
        for rec in pending:
            price_then = rec.meta.get("price_at_analysis")
            price_after = get_historical_price_4h(rec.timestamp_unix)
            
            if not price_then or not price_after: continue
            
            diff = ((price_after - price_then) / price_then) * 100
            # 원본의 '숏/롱 오판 분석' 의도를 담은 프롬프트
            prompt = f"분석: {rec.advice}\n결과: ${price_then} -> ${price_after} ({diff:+.2f}%)\n토론의 오판 이유를 분석하세요."
            
            try:
                msg = client.messages.create(
                    model=REFLECTION_MODEL,
                    max_tokens=800,
                    system=f"당신은 {role} 코치입니다. 토론 내용과 실제 결과의 차이를 분석하세요.",
                    messages=[{"role": "user", "content": prompt}]
                )
                memory.update_outcome(rec.timestamp, msg.content[0].text)
            except: continue
        
        memory.cleanup_old_no_outcome_records(3)
