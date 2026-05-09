import os
import time
import requests
from .memory import get_memory

REFLECTION_MODEL = os.getenv("REFLECTION_MODEL", "claude-haiku-4-5")

def get_historical_price_4h(timestamp_unix: float):
    """바이낸스 API로 정확히 4시간 뒤 가격 조회"""
    try:
        target_time = int((timestamp_unix + 14400) * 1000)
        url = "https://api.binance.com/api/v3/klines"
        res = requests.get(url, params={"symbol":"BTCUSDT", "interval":"1m", "startTime":target_time, "limit":1}, timeout=5).json()
        return float(res[0][4])
    except: return None

def reflect_all():
    from agents import get_anthropic_client
    client = get_anthropic_client()
    mem_manager = get_memory()
    roles = ["analyst", "bull", "bear", "judge", "aggressive", "conservative", "neutral"]
    
    for role in roles:
        memory = mem_manager.get(role)
        now = time.time()
        pending = [r for r in memory.records if not r.outcome and (now - r.timestamp_unix > 14400)]
        
        for rec in pending:
            price_then = rec.meta.get("price_at_analysis")
            price_after = get_historical_price_4h(rec.timestamp_unix)
            
            if not price_then or not price_after: continue
            
            diff = ((price_after - price_then) / price_then) * 100
            prompt = f"4시간 전 분석: {rec.advice}\n결과: ${price_then:,.2f} -> ${price_after:,.2f} ({diff:+.2f}%)\n판단 미스나 기회비용을 분석하세요."
            
            try:
                msg = client.messages.create(
                    model=REFLECTION_MODEL,
                    max_tokens=800,
                    system=f"당신은 {role} 코치입니다. 마크다운 없이 작성하세요.",
                    messages=[{"role": "user", "content": prompt}]
                )
                memory.update_outcome(rec.timestamp, msg.content[0].text)
            except: continue
        
        memory.cleanup_old_records(3)
