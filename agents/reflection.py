import os
import time
import requests
from dataclasses import dataclass
from typing import Optional
import anthropic
from config import CLAUDE_API_KEY
from .memory import FinancialSituationMemory, get_memory

REFLECTION_MODEL = os.getenv("REFLECTION_MODEL", "claude-3-5-sonnet-20240620")

def _get_client():
    from agents import get_anthropic_client
    return get_anthropic_client()

def get_historical_price_4h(timestamp_unix: float):
    """정확히 분석 4시간 뒤의 가격을 바이낸스에서 가져옴"""
    try:
        target_time = int((timestamp_unix + (4 * 3600)) * 1000)
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": "BTCUSDT", "interval": "1m", "startTime": target_time, "limit": 1}
        res = requests.get(url, params=params, timeout=5).json()
        return float(res[0][4]) if res else None
    except: return None

# 원본의 시스템 프롬프트 및 규칙들 (숏/롱 오판 분석 포함)
_BASE_RULES = """
- 당시 에이전트 간의 토론(Short vs Long) 결과와 실제 가격 움직임이 다를 경우 '치명적 오판' 이유를 분석할 것.
- 기회비용(Missed Opportunity) 관점에서 무엇을 놓쳤는지 반드시 언급할 것.
- 마크다운 금지, 일반 텍스트로 작성.
"""

def reflect_all():
    """모든 역할에 대해 4시간 경과 후 복기 실행"""
    roles = ["analyst", "bull", "bear", "judge", "aggressive", "conservative", "neutral"]
    client = _get_client()
    
    for role in roles:
        memory = get_memory(role)
        now = time.time()
        
        # 4시간이 지났고 복기가 아직 안 된 기록들
        pending = [r for r in memory.records if not r.outcome and (now - r.timestamp_unix > 14400)]
        
        for rec in pending:
            price_then = rec.meta.get("price_at_analysis")
            price_after = get_historical_price_4h(rec.timestamp_unix)
            
            if not price_then or not price_after: continue
            
            diff = ((price_after - price_then) / price_then) * 100
            direction = "상승" if diff > 0 else "하락"
            
            prompt = f"""
            [4시간 전 분석 내용]
            {rec.advice}
            
            [실제 시장 결과]
            시작 가격: ${price_then:,.2f}
            4시간 후 가격: ${price_after:,.2f}
            변화율: {diff:+.2f}% ({direction})
            
            위 결과를 토대로 당시의 토론이 적절했는지, 숏/롱 판단 미스가 있었다면 왜 발생했는지 복기하세요.
            """
            
            try:
                msg = client.messages.create(
                    model=REFLECTION_MODEL,
                    max_tokens=800,
                    system=f"당신은 {role} 관점의 트레이딩 코치입니다. {_BASE_RULES}",
                    messages=[{"role": "user", "content": prompt}]
                )
                memory.update_outcome(rec.timestamp, msg.content[0].text)
            except: continue
        
        # 3일 지난 미복기 데이터 정제
        memory.cleanup_old_no_outcome_records(days_threshold=3)

if __name__ == "__main__":
    reflect_all()
