# =============================================
# Crypto Trading Signal Analyzer - Config
# =============================================
import hmac
import os
from pathlib import Path
from dotenv import load_dotenv

_BASE_DIR = Path(__file__).resolve().parent
load_dotenv(_BASE_DIR / ".env")
load_dotenv()


def _safe_env(key: str, default: str = "") -> str:
    val = os.getenv(key, default) or default
    return val.replace("\r", "").replace("\n", "").strip()


_OWNER_PASSWORD_DEFAULT = "changeme"


def owner_password_configured() -> bool:
    pw = OWNER_PASSWORD
    return bool(pw) and pw != _OWNER_PASSWORD_DEFAULT


def verify_owner_password(supplied: object) -> bool:
    if not owner_password_configured():
        return False
    if not isinstance(supplied, str):
        return False
    return hmac.compare_digest(supplied.encode("utf-8"), OWNER_PASSWORD.encode("utf-8"))


def sanitize_env_value(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.replace("\r", "").replace("\n", "").replace("\x00", "").strip()


# ── Claude ───────────────────────────────────
CLAUDE_API_KEY = _safe_env("CLAUDE_API_KEY")
CLAUDE_MODEL   = _safe_env("CLAUDE_MODEL", "claude-sonnet-4-6")

# ── Binance (시세 데이터용) ───────────────────
BINANCE_BASE_URL    = "https://api.binance.com"
BINANCE_FUTURES_URL = "https://fapi.binance.com"
BINANCE_API_KEY     = _safe_env("BINANCE_API_KEY")
BINANCE_SECRET_KEY  = _safe_env("BINANCE_SECRET_KEY")
DEFAULT_SYMBOL      = _safe_env("DEFAULT_SYMBOL", "BTCUSDT").upper()

# ── Bitget 자동매매 ───────────────────────────
BITGET_API_KEY    = _safe_env("BITGET_API_KEY")
BITGET_SECRET_KEY = _safe_env("BITGET_SECRET_KEY")
BITGET_PASSPHRASE = _safe_env("BITGET_PASSPHRASE")

# TP/SL 전략: Claude AI의 피보나치 기반 목표가/손절가를 그대로 거래소에 전달
AUTO_TRADE_ENABLED  = _safe_env("AUTO_TRADE_ENABLED", "false").lower() == "true"
AUTO_TRADE_USDT     = float(_safe_env("AUTO_TRADE_USDT",    "20"))    # 1회 거래 증거금 (USDT)
AUTO_TRADE_LEVERAGE = int(_safe_env("AUTO_TRADE_LEVERAGE",  "3"))     # 레버리지 배수
AUTO_TRADE_MIN_CONF = int(_safe_env("AUTO_TRADE_MIN_CONF",  "65"))    # 최소 AI 확신도 (%)
AUTO_TRADE_USE_TP   = _safe_env("AUTO_TRADE_USE_TP", "true").lower()  == "true"
AUTO_TRADE_USE_SL   = _safe_env("AUTO_TRADE_USE_SL", "true").lower()  == "true"
# ATR 관련 파라미터 제거 — AI 피보나치 TP/SL 직접 사용


def symbol_to_pair(symbol: str) -> str:
    symbol = (symbol or "").upper()
    quote_candidates = ("USDC", "USDT", "FDUSD", "BUSD", "TUSD", "USD", "BTC", "ETH", "BNB")
    for quote in quote_candidates:
        if symbol.endswith(quote) and len(symbol) > len(quote):
            return f"{symbol[:-len(quote)]}/{quote}"
    return symbol


DEFAULT_LEVERAGE     = AUTO_TRADE_LEVERAGE
OWNER_PASSWORD       = _safe_env("OWNER_PASSWORD", _OWNER_PASSWORD_DEFAULT)

TIMEFRAMES     = ["1h", "4h", "1d"]
CANDLE_LIMIT   = 200
AUTO_REFRESH_INTERVAL = 1800

# 색상 팔레트
BG_COLOR     = "#0d0d1a"
PANEL_COLOR  = "#13132a"
ACCENT_COLOR = "#1e1e4a"
TEXT_COLOR   = "#dce1f0"
GREEN_COLOR  = "#00e676"
RED_COLOR    = "#ff1744"
YELLOW_COLOR = "#ffd740"
BLUE_COLOR   = "#40c4ff"
PURPLE_COLOR = "#ce93d8"
