# =============================================
# Binance USDⓈ-M Futures Public REST API - OHLCV 데이터 가져오기
# =============================================
import requests
import pandas as pd
from config import BINANCE_FUTURES_URL, CANDLE_LIMIT
from http_client import _session as _http  # 프록시 환경변수 무시 세션


def fetch_ohlcv(symbol: str, interval: str, limit: int = CANDLE_LIMIT) -> pd.DataFrame:
    """
    바이낸스 USDⓈ-M Futures 공개 API로 OHLCV 캔들 데이터를 가져옵니다.

    Parameters
    ----------
    symbol   : 'BTCUSDT' 형식
    interval : '15m' | '1h' | '4h' | '1d'
    limit    : 최대 1000, 기본 200

    Returns
    -------
    DataFrame(index=datetime, columns=[open, high, low, close, volume])
    """
    url = f"{BINANCE_FUTURES_URL}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    resp = _http.get(url, params=params, timeout=10)
    resp.raise_for_status()
    raw = resp.json()

    columns = [
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(raw, columns=columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df.set_index("timestamp", inplace=True)
    return df[["open", "high", "low", "close", "volume"]]


def fetch_current_price(symbol: str) -> float:
    """비트겟 선물 현재 체결가 반환."""
    # 비트겟 API 우선 시도
    try:
        url = "https://api.bitget.com/api/v2/mix/market/ticker"
        resp = _http.get(url, params={"symbol": symbol, "productType": "USDT-FUTURES"}, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        # data 가 list 또는 dict 둘 다 가능 — 명시적으로 분기.
        # 기존 한 줄 식은 연산자 우선순위 때문에 dict 응답 시 항상 None 이었음.
        d = data.get("data")
        price = None
        if isinstance(d, list) and d:
            price = (d[0] or {}).get("lastPr")
        elif isinstance(d, dict):
            price = d.get("lastPr")
        if price:
            return float(price)
    except Exception:
        pass

    # 폴백: 바이낸스
    url = f"{BINANCE_FUTURES_URL}/fapi/v1/ticker/price"
    resp = _http.get(url, params={"symbol": symbol}, timeout=5)
    resp.raise_for_status()
    return float(resp.json()["price"])

def fetch_high_low_since(symbol: str, since_ts: str) -> dict:
    """
    since_ts(ISO8601) 이후부터 현재까지의 최고가/최저가/현재가 반환.
    비트겟 API 우선, 실패 시 바이낸스 폴백.
    
    Returns: {"high": float, "low": float, "current": float}
    """
    import datetime as _dt

    try:
        since = _dt.datetime.fromisoformat(since_ts.replace("Z", "+00:00"))
    except Exception:
        since = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=24)

    # 기간 계산 → 적절한 interval 선택
    elapsed_h = (_dt.datetime.now(_dt.timezone.utc) - since).total_seconds() / 3600

    if elapsed_h <= 6:
        interval = "15m"
        limit = int(elapsed_h * 4) + 5
    elif elapsed_h <= 24:
        interval = "1h"
        limit = int(elapsed_h) + 5
    elif elapsed_h <= 168:
        interval = "4h"
        limit = int(elapsed_h / 4) + 5
    else:
        interval = "1d"
        limit = int(elapsed_h / 24) + 5

    limit = min(max(limit, 5), 200)

    try:
        df = fetch_ohlcv(symbol, interval, limit)
        # df.index 가 tz-aware 인지 naive 인지 확인 후 since 를 같은 형태로 맞춤
        # (pandas 버전이나 fetch_ohlcv 변경 시 tz 가 붙을 수 있음)
        idx_is_tz_aware = (
            isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None
        )
        if idx_is_tz_aware:
            since_aligned = since.astimezone(_dt.timezone.utc)
        else:
            since_aligned = since.astimezone(_dt.timezone.utc).replace(tzinfo=None)

        df_since = df[df.index >= since_aligned]
        if df_since.empty:
            df_since = df

        high = float(df_since["high"].max())
        low  = float(df_since["low"].min())
        current = float(df_since["close"].iloc[-1])
        return {"high": high, "low": low, "current": current}
    except Exception:
        # 폴백: 현재가만 반환
        try:
            current = fetch_current_price(symbol)
            return {"high": current, "low": current, "current": current}
        except Exception:
            return {"high": 0, "low": 0, "current": 0}

