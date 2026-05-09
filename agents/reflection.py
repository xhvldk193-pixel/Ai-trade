# =============================================
# Reflection Agent — 특정 시간 경과 후 가격 기반 복기 (완성판)
# =============================================
from __future__ import annotations

import os
import time
import anthropic
import requests
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timedelta

# 설정 및 환경 변수
def _get_client():
    from agents import get_anthropic_client
    return get_anthropic_client()

from config import CLAUDE_API_KEY
from .memory import FinancialSituationMemory, get_memory

REFLECTION_MODEL = os.getenv("REFLECTION_MODEL", "claude-3-5-sonnet-20240620")

# ── 과거 특정 시점의 가격 가져오기 함수 ────────────────────────────
def get_historical_price(timestamp_unix: float, hours_after: int = 4, symbol: str = "BTCUSDT") -> float:
    """
    분석 시점(timestamp_unix)으로부터 특정 시간(hours_after) 후의 가격을 바이낸스에서 가져옵니다.
    """
    try:
        # 분석 시점 + n시간 (밀리초 단위로 변환)
        target_time = int((timestamp_unix + (hours_after * 3600)) * 1000)
        
        # 바이낸스 과거 데이터 API (Klines) 호출
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": "1m",
            "startTime": target_time,
            "limit": 1
        }
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        
        if data and len(data) > 0:
            # 해당 봉의 종가(Close price) 반환
            return float(data[0][4])
        return 0.0
    except Exception as e:
        print(f"과거 가격 조회 오류: {e}")
        return 0.0

# ── 역할별 시스템 프롬프트 (전과 동일) ────────────────────────────
_BASE_RULES = """
출력 규칙:
1. 결과 판정: 수익 / 손절 / 미진입 성공 / 미진입 실패(기회비용) 중 하나를 반드시 선택.
2. 분석 기준: '판단 시점으로부터 약 4시간 뒤'의 결과를 바탕으로 당시 판단의 합리성을 평가.
3. 마크다운 사용 금지, 일반 텍스트로 작성.
4. 마지막 줄은 반드시 '다음 체크리스트:' 로 시작하는 요약."""

_HINDSIGHT_GUARD = "결과론 금지: 사후 가격을 안다고 해서 당시 판단을 무조건 틀렸다고 하지 마세요."

ROLE_REFLECTION_SYSTEMS = {
    "analyst": f"당신은 BTC 애널리스트 코치입니다. {_HINDSIGHT_GUARD} {_BASE_RULES}",
    # ... 다른 역할들도 이 프롬프트를 공유하거나 개별 설정 가능
}

# ── 복기 수행 함수 ───────────────────────────────────
def reflect_for_role(role, record_ts, situation, advice, price_then, price_after, hours, memory):
    client = _get_client()
    pct = ((price_after - price_then) / price_then * 100.0) if price_then else 0.0
    direction = "상승" if pct > 0 else "하락"

    # 유저 프롬프트 구성
    prompt = f"""
[과거 판단 시점] {record_ts}
[역할] {role}
[당시 상황] {situation}
[당시 조언] {advice}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[판단 {hours}시간 후 결과]
기준 가격: ${price_then:,.2f}
{hours}시간 후 가격: ${price_after:,.2f}
변화율: {pct:+.2f}% ({direction})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
이 결과를 바탕으로 리플렉션을 작성하세요.
"""

    system_prompt = ROLE_REFLECTION_SYSTEMS.get(role, ROLE_REFLECTION_SYSTEMS["analyst"])
    
    try:
        msg = client.messages.create(
            model=REFLECTION_MODEL,
            max_tokens=800,
            system=[{"type": "text", "text": system_prompt}],
            messages=[{"role": "user", "content": prompt}]
        )
        reflection_text = msg.content[0].text.strip()
        
        outcome_block = f"[{hours}시간 후 복기] {reflection_text}"
        memory.update_outcome(record_ts, outcome_block)
        return True
    except Exception as e:
        print(f"복기 도중 오류: {e}")
        return False

def reflect_all():
    """모든 역할에 대해 '4시간 뒤' 가격을 기준으로 복기를 수행합니다."""
    roles = ["analyst", "bull", "bear", "judge", "aggressive", "conservative", "neutral"]
    HOURS_AFTER = 4 # 4시간 후를 기준으로 설정
    
    for role in roles:
        memory = get_memory(role)
        if not memory: continue
        
        # 결과가 없는 기록 중, 발생한 지 4시간이 넘은 기록들만 대상 (과거 데이터를 가져와야 하므로)
        now_unix = time.time()
        pending = [
            r for r in memory.records 
            if (not r.outcome) and (now_unix - r.timestamp_unix > HOURS_AFTER * 3600)
        ]
        
        print(f"[{role}] {len(pending)}개의 {HOURS_AFTER}시간 경과 기록 복기 시작...")
        
        for rec in pending:
            price_then = rec.meta.get("price_at_analysis", 0)
            # 바이낸스에서 정확히 '그때부터 4시간 뒤'의 가격을 가져옴
            price_after = get_historical_price(rec.timestamp_unix, hours_after=HOURS_AFTER)
            
            if price_then > 0 and price_after > 0:
                reflect_for_role(role, rec.timestamp, rec.situation, rec.advice, 
                                 price_then, price_after, HOURS_AFTER, memory)
                print(f"  - {rec.timestamp} (4시간 후 가격: {price_after}) 복기 완료.")

        # 3일 지난 쓰레기 데이터 정리
        memory.cleanup_old_no_outcome_records(days_threshold=3)

if __name__ == "__main__":
    reflect_all()
