#!/usr/bin/env python3
from __future__ import annotations

# =============================================
# BTC Signal Analyzer — FastAPI backend
# =============================================
import asyncio
import copy
import concurrent.futures
import contextlib
import functools
import json
import math
import os
import random
import time
import uuid

import pandas as pd
import websockets
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

import os as _os, requests as _req
def _tg(msg):
    t=_os.environ.get("TELEGRAM_BOT_TOKEN","");c=_os.environ.get("TELEGRAM_CHAT_ID","")
    if t and c:
        # parse_mode 제거 — 메시지에 < 들어가면 HTML 파싱 실패로 전송 자체가 실패함
        try:_req.post(f"https://api.telegram.org/bot{t}/sendMessage",json={"chat_id":c,"text":msg},timeout=5)
        except Exception:
            pass

import config as runtime_config
from config import (
    CANDLE_LIMIT, DEFAULT_SYMBOL, TIMEFRAMES, symbol_to_pair,
    BITGET_API_KEY, BITGET_SECRET_KEY, BITGET_PASSPHRASE,
    AUTO_TRADE_ENABLED, AUTO_TRADE_USDT, AUTO_TRADE_LEVERAGE,
    AUTO_TRADE_MIN_CONF, AUTO_TRADE_USE_TP, AUTO_TRADE_USE_SL,
)

# ── 텔레그램 알림 ────────────────────────────
import telegram_alert

# ── Bitget 자동매매 ────────────────────────────
try:
    from bitget_trader import BitgetAutoTrader
    _BITGET_AVAILABLE = True
except ImportError:
    BitgetAutoTrader = None  # type: ignore
    _BITGET_AVAILABLE = False
from account_context import (
    close_user_data_stream,
    fetch_account_context,
    keepalive_user_data_stream,
    open_user_data_stream,
)
from data_fetcher import fetch_ohlcv, fetch_current_price
from indicators import add_all_indicators, fibonacci_swing_levels, fib_window_for_tf
from analyzer import analyze_with_claude, run_full_analysis
from macro_fetcher import fetch_macro_context
from time_utils import format_kst, now_kst

# ── Reflection / Memory (optional: rank_bm25 미설치 시 None) ──
try:
    from agents.memory import get_memory as _get_memory, get_agent_memories as _get_agent_memories
    _MEMORY_OK = True
except Exception as _mem_exc:
    _get_memory = None
    _get_agent_memories = None
    _MEMORY_OK = False
    print(f"[WARN] agents.memory 로드 실패 — {_mem_exc}", flush=True)

try:
    from agents.reflection import reflect_on_record as _reflect_on_record, reflect_for_role as _reflect_for_role
    _REFLECT_OK = True
except Exception as _ref_exc:
    _reflect_on_record = None
    _reflect_for_role = None
    _REFLECT_OK = False
    print(f"[WARN] agents.reflection 로드 실패 — {_ref_exc}", flush=True)

_REFLECTION_ENABLED = _MEMORY_OK and _REFLECT_OK
print(f"[init] REFLECTION_ENABLED={_REFLECTION_ENABLED} (memory={_MEMORY_OK}, reflect={_REFLECT_OK})", flush=True)

# 에이전트 메모리 역할 목록 (analyst 는 별도 memory 이름 사용)
_AGENT_ROLES_FOR_REFLECT = ("bull", "bear", "judge", "aggressive", "conservative", "neutral")

# ── 색상 팔레트 ────────────────────────────────
C = {
    "bg":      "#FAF8F5",
    "surface": "#FFFFFF",
    "border":  "#E8DDD4",
    "txt":     "#1C1917",
    "dim":     "#78716C",
    "muted":   "#A8978A",
    "orange":  "#EA580C",
    "amber":   "#D97706",
    "bull":    "#16A34A",
    "bear":    "#DC2626",
    "sky":     "#0284C7",
    "violet":  "#7C3AED",
    "teal":    "#0D9488",
    "plot_bg": "#FDFBF8",
    "grid":    "#EDE8E0",
}
FIB_COLORS = {
    "0.000": "#DC2626", "0.236": "#EA580C", "0.382": "#D97706",
    "0.500": "#78716C", "0.618": "#16A34A", "0.786": "#0284C7",
    "1.000": "#16A34A",
}

app = FastAPI()

# ── 2단계 인증 ────────────────────────────
import secrets
_pending_codes = {}  # {code: expire_time}
# 세션 토큰 저장: {token: expire_ts}  — 평문이지만 만료 검증으로 7일 무한 도용 방지
_auth_sessions: dict[str, float] = {}
_AUTH_SESSION_TTL = 86400 * 7  # 7일

def _gen_code() -> str:
    return str(random.randint(100000, 999999))

def _send_auth_code(code: str):
    import os as _os2, requests as _req2
    t=_os2.environ.get("TELEGRAM_BOT_TOKEN","");c=_os2.environ.get("TELEGRAM_CHAT_ID","")
    if t and c:
        try:_req2.post(f"https://api.telegram.org/bot{t}/sendMessage",json={"chat_id":c,"text":f"🔐 로그인 인증코드: {code}\n5분 내 입력하세요."},timeout=5)
        except Exception:
            pass

def _is_session_valid(token: str) -> bool:
    """세션 토큰이 존재하고 만료되지 않았는지 검증."""
    if not token:
        return False
    expire_ts = _auth_sessions.get(token)
    if expire_ts is None:
        return False
    if expire_ts < time.time():
        # 만료된 세션 즉시 폐기
        _auth_sessions.pop(token, None)
        return False
    return True

@app.middleware("http")
async def auth_middleware(request, call_next):
    path = request.url.path
    # 정적 파일, 인증 엔드포인트는 통과
    if path in ("/auth/login", "/auth/verify", "/auth/check", "/api/symbol", "/api/memory/reset"):
        return await call_next(request)

    # 세션 토큰 확인 (만료 검증 포함)
    token = request.cookies.get("session_token")
    if _is_session_valid(token):
        return await call_next(request)

    # 미인증 → 로그인 페이지
    if path == "/" or not path.startswith("/api"):
        from fastapi.responses import HTMLResponse
        return HTMLResponse(_login_html())

    from fastapi.responses import JSONResponse
    return JSONResponse({"error": "unauthorized"}, status_code=401)

@app.post("/auth/login")
async def auth_login(request: Request):
    body = await request.json()
    pw = body.get("password", "")
    site_pw = os.environ.get("SITE_PASSWORD", "")
    if not site_pw or pw != site_pw:
        return {"ok": False, "error": "비밀번호 오류"}
    code = _gen_code()
    expire = time.time() + 300  # 5분
    # 신 포맷: brute force 방어를 위한 attempts 카운터 포함
    _pending_codes[code] = {"expire": expire, "attempts": 0}
    _send_auth_code(code)
    return {"ok": True}

@app.post("/auth/verify")
async def auth_verify(request: Request):
    body = await request.json()
    code = body.get("code", "")
    now = time.time()
    # 만료 코드 정리
    expired = [k for k, v in _pending_codes.items() if (v["expire"] if isinstance(v, dict) else v) < now]
    for k in expired: del _pending_codes[k]

    if code in _pending_codes:
        info = _pending_codes[code]
        # 구 포맷(epoch float) 호환 — 신 포맷은 dict
        if not isinstance(info, dict):
            info = {"expire": info, "attempts": 0}
            _pending_codes[code] = info
        info["attempts"] = info.get("attempts", 0) + 1
        if info["attempts"] > 3:
            # brute force 차단 — 3회 실패 시 코드 폐기
            del _pending_codes[code]
            return {"ok": False, "error": "시도 횟수 초과 — 다시 로그인하세요."}
        del _pending_codes[code]  # 정상 검증도 즉시 폐기 (one-time)
        token = secrets.token_hex(32)
        _auth_sessions[token] = now + _AUTH_SESSION_TTL
        _save_auth_sessions()
        from fastapi.responses import JSONResponse
        resp = JSONResponse({"ok": True})
        # 쿠키 보안 옵션 — samesite=lax 로 CSRF 방어, secure 는 HTTPS 환경에서만
        resp.set_cookie(
            "session_token", token,
            httponly=True,
            samesite="lax",
            secure=os.environ.get("HTTPS_ONLY", "false").lower() == "true",
            max_age=_AUTH_SESSION_TTL,
        )
        return resp
    return {"ok": False, "error": "코드 오류 또는 만료"}

@app.get("/auth/check")
async def auth_check(request: Request):
    token = request.cookies.get("session_token")
    return {"ok": _is_session_valid(token)}

def _login_html():
    return """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>로그인</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d0d1a; color: #dce1f0; font-family: sans-serif; display: flex; align-items: center; justify-content: center; min-height: 100vh; }
.box { background: #12122a; border: 1px solid #3a3a6a; border-radius: 16px; padding: 32px; width: min(380px, 94vw); }
h2 { font-size: 18px; margin-bottom: 24px; color: #a0a8f8; text-align: center; }
input { width: 100%; padding: 12px; background: #1c1c3e; border: 1px solid #3a3a6a; border-radius: 8px; color: #e0e6ff; font-size: 16px; margin-bottom: 12px; }
button { width: 100%; padding: 12px; background: #3030a0; color: #e0e6ff; border: none; border-radius: 8px; font-size: 15px; font-weight: 700; cursor: pointer; }
.msg { font-size: 13px; text-align: center; margin-top: 10px; min-height: 18px; }
.step2 { display: none; }
</style>
</head>
<body>
<div class="box">
  <h2>🔐 AI 자동매매</h2>
  <div id="step1">
    <input type="password" id="pw" placeholder="비밀번호" autocomplete="current-password">
    <button onclick="doLogin()">로그인</button>
    <div class="msg" id="msg1"></div>
  </div>
  <div id="step2" class="step2">
    <p style="margin-bottom:12px;font-size:13px;color:#9098c0;text-align:center">텔레그램으로 전송된 6자리 코드를 입력하세요</p>
    <input type="text" id="code" placeholder="123456" maxlength="6" inputmode="numeric">
    <button onclick="doVerify()">확인</button>
    <div class="msg" id="msg2"></div>
  </div>
</div>
<script>
async function doLogin() {
  const pw = document.getElementById('pw').value;
  const r = await fetch('/auth/login', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({password: pw})});
  const d = await r.json();
  if (d.ok) {
    document.getElementById('step1').style.display='none';
    document.getElementById('step2').style.display='block';
  } else {
    document.getElementById('msg1').textContent = '비밀번호 오류';
    document.getElementById('msg1').style.color = '#ff5252';
  }
}
async function doVerify() {
  const code = document.getElementById('code').value;
  const r = await fetch('/auth/verify', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({code: code})});
  const d = await r.json();
  if (d.ok) {
    location.reload();
  } else {
    document.getElementById('msg2').textContent = '코드 오류 또는 만료';
    document.getElementById('msg2').style.color = '#ff5252';
  }
}
document.addEventListener('keydown', e => {
  if (e.key === 'Enter') {
    if (document.getElementById('step2').style.display === 'block') doVerify();
    else doLogin();
  }
});
</script>
</body>
</html>"""

# 분석(Claude API) + 데이터 fetch가 동시에 실행될 수 있도록 워커 수 충분히 확보
# 기본 4개는 분석 1건만으로 전부 포화 → CPU 코어 × 4 또는 최소 16
_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=max(16, (os.cpu_count() or 4) * 4)
)


def _require_owner(password: object) -> None:
    """OWNER_PASSWORD 를 타이밍-공격 내성으로 검증. 실패 시 HTTP 예외.

    - OWNER_PASSWORD 가 설정되지 않았거나 기본값이면 모든 요청을 403 으로 거절
      → 로컬-only 기능을 외부에서 우연히 호출하는 사고 방지.
    """
    if not runtime_config.owner_password_configured():
        raise HTTPException(
            status_code=403,
            detail="OWNER_PASSWORD 환경변수가 설정되지 않아 이 기능은 비활성화돼 있습니다.",
        )
    if not runtime_config.verify_owner_password(password):
        raise HTTPException(status_code=403, detail="비밀번호가 틀렸습니다.")
CHART_WINDOW = 120
BASE_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
ACCOUNT_STREAM_WS_BASE = "wss://fstream.binance.com/ws"
ACCOUNT_STREAM_KEEPALIVE_SECS = 50 * 60
ACCOUNT_REFRESH_DEBOUNCE_SECS = 0.35
ACCOUNT_REFRESH_MIN_INTERVAL_SECS = 1.5
MARKET_STREAM_WS_BASE = "wss://fstream.binance.com/stream"   # 레거시 (IP 차단 시 미사용)
BYBIT_MARKET_WS_URL   = "wss://stream.bybit.com/v5/public/linear"
# Bybit ↔ TIMEFRAMES 변환 테이블
_TF_TO_BYBIT  = {"5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "D"}
_BYBIT_TF_MAP = {v: k for k, v in _TF_TO_BYBIT.items()}
MACRO_REFRESH_SECS = 60 * 60
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_AUTH_SESSIONS_PATH = os.path.join(BASE_DIR, "data", "auth_sessions.json")

def _load_auth_sessions():
    try:
        if os.path.exists(_AUTH_SESSIONS_PATH):
            with open(_AUTH_SESSIONS_PATH) as f:
                data = json.load(f)
            now = time.time()
            # 신 포맷: {"sessions": {token: expire_ts, ...}}
            sessions_obj = data.get("sessions", {})
            if isinstance(sessions_obj, dict):
                for tok, exp in sessions_obj.items():
                    try:
                        exp_f = float(exp)
                        if exp_f > now:
                            _auth_sessions[tok] = exp_f
                    except (TypeError, ValueError):
                        continue
            elif isinstance(sessions_obj, list):
                # 구 포맷 호환: [token, ...] — 모두 7일 기준 만료 부여
                for tok in sessions_obj:
                    if isinstance(tok, str):
                        _auth_sessions[tok] = now + _AUTH_SESSION_TTL
    except Exception:
        pass

def _save_auth_sessions():
    try:
        os.makedirs(os.path.dirname(_AUTH_SESSIONS_PATH), exist_ok=True)
        # 만료된 토큰은 저장 안 함 (디스크 무한 누적 방지)
        now = time.time()
        valid = {tok: exp for tok, exp in _auth_sessions.items() if exp > now}
        with open(_AUTH_SESSIONS_PATH, "w") as f:
            json.dump({"sessions": valid}, f)
    except Exception:
        pass

LATEST_ANALYSIS_PATH   = os.path.join(BASE_DIR, "data", "latest_analysis.json")
ANALYSIS_HISTORY_PATH  = os.path.join(BASE_DIR, "data", "analysis_history.jsonl")
ANALYSIS_HISTORY_MAX   = 500   # JSONL 최대 보관 건수


# ══════════════════════════════════════════════
# ECharts용 차트 데이터 직렬화
# ══════════════════════════════════════════════
def _series_values(values) -> list:
    return [_safe(v) for v in values]


def _series_labels(index) -> list[str]:
    labels = []
    for ts in index:
        try:
            labels.append(format_kst(ts, "%Y-%m-%d %H:%M"))
        except Exception:
            labels.append(str(ts)[:16])
    return labels


def _now_label() -> str:
    return now_kst().strftime("%H:%M:%S")


def _now_iso() -> str:
    return now_kst().isoformat(timespec="seconds")


def _load_latest_analysis() -> dict | None:
    try:
        with open(LATEST_ANALYSIS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except FileNotFoundError:
        return None
    except Exception as exc:
        print(f"[analysis-cache] load failed: {exc}")
        return None


def _persist_latest_analysis(payload: dict):
    try:
        os.makedirs(os.path.dirname(LATEST_ANALYSIS_PATH), exist_ok=True)
        tmp_path = f"{LATEST_ANALYSIS_PATH}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_path, LATEST_ANALYSIS_PATH)
    except Exception as exc:
        print(f"[analysis-cache] persist failed: {exc}")


# 히스토리 라인 수 캐시 (프로세스 생애 동안 유지) — 매 append 마다 파일 전체를
# 읽어 길이를 세지 않도록 lazy 초기화 후 카운터만 증가시킨다.
_analysis_history_line_count: int | None = None


def _count_history_lines() -> int:
    try:
        if not os.path.exists(ANALYSIS_HISTORY_PATH):
            return 0
        count = 0
        with open(ANALYSIS_HISTORY_PATH, "rb") as f:
            for _ in f:
                count += 1
        return count
    except Exception:
        return 0


def _truncate_history_file() -> None:
    """JSONL 히스토리 파일을 최대 건수로 잘라낸다 (tail 만 유지)."""
    from collections import deque
    try:
        with open(ANALYSIS_HISTORY_PATH, "r", encoding="utf-8") as f:
            tail = deque(f, maxlen=ANALYSIS_HISTORY_MAX)
        tmp_path = f"{ANALYSIS_HISTORY_PATH}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.writelines(tail)
        os.replace(tmp_path, ANALYSIS_HISTORY_PATH)
    except Exception as exc:
        print(f"[analysis-history] truncate failed: {exc}")


def _persist_analysis_history(payload: dict):
    """분석 결과 핵심 필드를 JSONL 히스토리 파일에 누적 저장."""
    global _analysis_history_line_count
    try:
        os.makedirs(os.path.dirname(ANALYSIS_HISTORY_PATH), exist_ok=True)
        sections = payload.get("report_sections") or {}
        entry = {
            "timestamp":   _now_iso(),
            "signal":      payload.get("signal"),
            "confidence":  payload.get("confidence"),
            "price":       payload.get("price"),
            "pair_label":  payload.get("pair_label"),
            "regime":      sections.get("regime"),
            "summary":     sections.get("summary"),
            "trade_levels": payload.get("trade_levels"),
        }
        with open(ANALYSIS_HISTORY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # lazy 카운터 초기화
        if _analysis_history_line_count is None:
            _analysis_history_line_count = _count_history_lines()
        else:
            _analysis_history_line_count += 1

        # 최대 건수를 일정 버퍼 이상 초과했을 때만 실제로 잘라냄 → 빈번한 rewrite 방지
        if _analysis_history_line_count > ANALYSIS_HISTORY_MAX + 50:
            _truncate_history_file()
            _analysis_history_line_count = ANALYSIS_HISTORY_MAX
    except Exception as exc:
        print(f"[analysis-history] persist failed: {exc}")


def build_chart_payload(df, fib: dict | None) -> dict:
    plot_df = df.tail(CHART_WINDOW).copy()
    labels = _series_labels(plot_df.index)

    candles = []
    volume_colors = []
    macd_hist_colors = []
    for row in plot_df.itertuples():
        open_v = _safe(row.open)
        close_v = _safe(row.close)
        low_v = _safe(row.low)
        high_v = _safe(row.high)
        hist_v = _safe(row.macd_hist)

        candles.append([open_v, close_v, low_v, high_v])

        is_bull = (
            open_v is not None and close_v is not None and close_v >= open_v
        )
        volume_colors.append(
            "rgba(22,163,74,.45)" if is_bull else "rgba(220,38,38,.45)"
        )
        macd_hist_colors.append(
            "rgba(22,163,74,.65)"
            if hist_v is not None and hist_v >= 0
            else "rgba(220,38,38,.65)"
        )

    active_levels = fib["levels"] if fib and fib.get("is_active") else {}
    fib_lines = [
        {
            "ratio": ratio,
            "price": round(float(price), 2),
            "color": FIB_COLORS.get(ratio, C["dim"]),
        }
        for ratio, price in active_levels.items()
    ]

    return {
        "labels": labels,
        "candles": candles,
        "bb_upper": _series_values(plot_df["bb_upper"]),
        "bb_lower": _series_values(plot_df["bb_lower"]),
        "sma_50": _series_values(plot_df["sma_50"]),
        "sma_200": _series_values(plot_df["sma_200"]),
        "ema_9": _series_values(plot_df["ema_9"]),
        "rsi": _series_values(plot_df["rsi"]),
        "macd": _series_values(plot_df["macd"]),
        "macd_signal": _series_values(plot_df["macd_signal"]),
        "macd_hist": _series_values(plot_df["macd_hist"]),
        "macd_hist_colors": macd_hist_colors,
        "volume": _series_values(plot_df["volume"]),
        "volume_ma": _series_values(plot_df["volume_ma"]),
        "volume_colors": volume_colors,
        "fib": fib,
        "fib_lines": fib_lines,
    }


# ══════════════════════════════════════════════
# 헬퍼
# ══════════════════════════════════════════════
def _safe(v):
    """NaN/None → None (JSON-safe float)."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _fetch_all(symbol: str) -> dict:
    result = {}
    for tf in TIMEFRAMES:
        df = fetch_ohlcv(symbol, tf)
        df = add_all_indicators(df, tf=tf)
        result[tf] = df
    return result


def _upsert_ohlcv(df, timestamp, row: dict) -> pd.DataFrame:
    if df is None or df.empty:
        base_df = pd.DataFrame(columns=BASE_OHLCV_COLUMNS)
    else:
        base_df = df[BASE_OHLCV_COLUMNS].copy()

    base_df = base_df.astype(float, copy=False)
    base_df.loc[timestamp, BASE_OHLCV_COLUMNS] = [
        row["open"], row["high"], row["low"], row["close"], row["volume"]
    ]
    base_df.sort_index(inplace=True)
    base_df = base_df[~base_df.index.duplicated(keep="last")]
    return base_df.tail(CANDLE_LIMIT)


def _serialize_swing_fib(tf: str, df) -> dict | None:
    result = fibonacci_swing_levels(df, window=fib_window_for_tf(tf))
    if result is None:
        return None

    current_price = _safe(df.iloc[-1]["close"]) if len(df.index) else None
    swing_low = round(float(result["swing_low"]), 2)
    swing_high = round(float(result["swing_high"]), 2)
    is_active = (
        current_price is not None and swing_low <= current_price <= swing_high
    )
    invalid_reason = None
    if current_price is not None:
        if current_price < swing_low:
            invalid_reason = "below_swing_low"
        elif current_price > swing_high:
            invalid_reason = "above_swing_high"

    return {
        "source_tf": tf,
        "direction": result["direction"],
        "direction_label": "상승 스윙" if result["direction"] == "up" else "하락 스윙",
        "swing_low": swing_low,
        "swing_low_ago": int(result["swing_low_ago"]),
        "swing_high": swing_high,
        "swing_high_ago": int(result["swing_high_ago"]),
        "leg_start": round(float(result["leg_start"]), 2),
        "leg_start_ago": int(result["leg_start_ago"]),
        "leg_start_type": result["leg_start_type"],
        "leg_end": round(float(result["leg_end"]), 2),
        "leg_end_ago": int(result["leg_end_ago"]),
        "leg_end_type": result["leg_end_type"],
        "current_price": current_price,
        "is_active": is_active,
        "invalid_reason": invalid_reason,
        "anchors": {
            ratio: round(float(level), 2)
            for ratio, level in result.get("anchors", {}).items()
        },
        "levels": {
            ratio: round(float(level), 2)
            for ratio, level in result["levels"].items()
        },
        "display_levels": {
            ratio: round(float(level), 2)
            for ratio, level in result.get("display_levels", result["levels"]).items()
        },
    }


def _build_swing_fibs(tf_data: dict, target_tfs: list[str]) -> dict:
    fibs = {}
    for tf in target_tfs:
        if tf in tf_data:
            fibs[tf] = _serialize_swing_fib(tf, tf_data[tf])
    return fibs


def _market_overview(tf_data: dict, price: float, last_update: str | None) -> dict:
    h1 = tf_data.get("1h")
    if h1 is None:
        h1 = tf_data.get(TIMEFRAMES[0]) if TIMEFRAMES else None

    if h1 is not None and not h1.empty:
        last = h1.iloc[-1]
        close_v = _safe(last["close"]) or 1.0
        indicators = {
            "rsi":       _safe(last["rsi"]),
            "macd_hist": _safe(last["macd_hist"]),
            "bb_pct":    _safe(last["bb_pct"]),
            "vol_ratio": round(_safe(last["volume"]) / max(_safe(last["volume_ma"]) or 1, 1) * 100, 1),
            "atr":       _safe(last["atr"]),
            "close":     _safe(last["close"]),
            "atr_pct":   round((_safe(last["atr"]) or 0) / close_v * 100, 2),
        }
    else:
        # tf_data 아직 없음 — WebSocket 스트림이 채울 때까지 빈 indicators
        indicators = {
            "rsi": None, "macd_hist": None, "bb_pct": None,
            "vol_ratio": None, "atr": None, "close": None, "atr_pct": None,
        }

    overview = {
        "symbol":      DEFAULT_SYMBOL,
        "pair_label":  symbol_to_pair(DEFAULT_SYMBOL),
        "price":       price,
        "last_update": last_update or _now_label(),
        "indicators":  indicators,
    }
    return overview


def _build_account_payload() -> dict:
    ctx = fetch_account_context()
    return {
        "updated_at": _now_label(),
        **ctx,
    }


def build_market_payload(
    tf_data: dict,
    price: float,
    last_update: str | None = None,
    chart_tfs: list[str] | tuple[str, ...] | set[str] | None = None,
    include_overview: bool = True,
) -> dict:
    target_tfs = TIMEFRAMES if chart_tfs is None else [tf for tf in TIMEFRAMES if tf in chart_tfs]
    fib_targets = TIMEFRAMES if include_overview else target_tfs
    fibs = _build_swing_fibs(tf_data, fib_targets)

    if include_overview:
        payload = _market_overview(tf_data, price, last_update)
        payload["fibs"] = fibs
    else:
        payload = {
            "symbol":      DEFAULT_SYMBOL,
            "pair_label":  symbol_to_pair(DEFAULT_SYMBOL),
            "price":       price,
            "last_update": last_update or _now_label(),
        }

    charts = {}
    for tf in target_tfs:
        if tf in tf_data:
            charts[tf] = build_chart_payload(tf_data[tf], fibs.get(tf))
    payload["charts"] = charts
    return payload


def _build_payload(tf_data: dict, price: float, analysis: dict) -> dict:
    payload = build_market_payload(tf_data, price, include_overview=True)
    return {
        **payload,
        "analysis_time": _now_label(),
        "signal":       analysis["signal"],
        "confidence":   analysis["confidence"],
        "raw_text":     analysis["raw_text"],
        "trade_levels": analysis["trade_levels"],
        "report_sections": analysis.get("report_sections", {}),
        "report_format_ok": analysis.get("report_format_ok", False),
        "report_missing_sections": analysis.get("report_missing_sections", []),
        # 구조화 트레이딩 시그널 (signal_processing.TradingSignal.to_dict())
        "trading_signal": analysis.get("trading_signal"),
        "claude_leverage": analysis.get("claude_leverage"),
        # Bull/Bear 사전 토론 결과
        "debate": analysis.get("debate"),
        # 투자 심판 결론
        "judge": analysis.get("judge"),
        # Risk Triad (Aggressive / Conservative / Neutral) 토론 결과
        "risk": analysis.get("risk"),
        # 과거 유사 상황 (BM25 회상 결과)
        "memories": analysis.get("memories", []),
        # account 는 account-stream 에서 실시간으로 별도 관리 — 여기서 포함하지 않음
    }


class MarketStreamManager:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self._tf_data: dict = {}
        self._price: float | None = None
        self._last_update: str | None = None
        self._ready = asyncio.Event()
        self._lock = asyncio.Lock()
        self._listeners: set[asyncio.Queue] = set()
        self._runner_task: asyncio.Task | None = None
        self._market_flush_task: asyncio.Task | None = None
        self._price_flush_task: asyncio.Task | None = None
        self._dirty_tfs: set[str] = set()
        self._needs_full_refresh = False
        self._stopped = False
        self._price_tick_task: asyncio.Task | None = None
        self._trade_count: int = 0          # aggTrade 수신 카운터 (디버그용)
        self._kline_count: int = 0          # kline 수신 카운터 (디버그용)
        self._ws_connect_count: int = 0     # WebSocket 재연결 횟수
        self._pending_kline: dict[str, dict] = {}    # 확정 kline 처리 큐
        self._kline_running: dict[str, bool] = {}    # 타임프레임별 계산 진행 플래그
        self._forming_kline: dict[str, dict] = {}    # 미확정(진행 중) 캔들 최신 OHLCV
        self._forming_refresh_task: asyncio.Task | None = None

    async def start(self):
        if self._runner_task and not self._runner_task.done():
            return
        self._stopped = False
        self._runner_task = asyncio.create_task(self._run_forever(), name="binance-market-stream")
        self._price_tick_task = asyncio.create_task(self._periodic_price_tick(), name="price-tick-1s")
        self._forming_refresh_task = asyncio.create_task(self._forming_indicator_refresh(), name="forming-refresh-2s")

    async def stop(self):
        self._stopped = True
        tasks = [self._runner_task, self._market_flush_task, self._price_flush_task,
                 self._price_tick_task, self._forming_refresh_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
        for task in tasks:
            if task:
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=16)
        async with self._lock:
            self._listeners.add(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue):
        async with self._lock:
            self._listeners.discard(q)

    async def wait_until_ready(self):
        await self._ready.wait()

    def is_ready(self) -> bool:
        return self._ready.is_set()

    async def get_full_snapshot(self) -> dict:
        await self.wait_until_ready()
        async with self._lock:
            return build_market_payload(
                self._tf_data,
                self._price,
                self._last_update,
                chart_tfs=TIMEFRAMES,
                include_overview=True,
            )

    async def get_analysis_inputs(self) -> tuple[dict, float | None]:
        """분석용 데이터 반환. 미확정 캔들 OHLCV를 반영해 지표를 재계산한다."""
        await self.wait_until_ready()
        async with self._lock:
            raw_snapshot = {tf: df.copy(deep=True) for tf, df in self._tf_data.items()}
            forming = dict(self._forming_kline)  # 미확정 캔들 최신본
            price = self._price

        def _recompute_all(snapshot: dict, forming_klines: dict) -> dict:
            result = {}
            for tf, df in snapshot.items():
                try:
                    # 미확정 캔들이 있으면 OHLCV를 먼저 upsert
                    k = forming_klines.get(tf)
                    if k:
                        ts = pd.to_datetime(int(k["t"]), unit="ms")
                        row = {
                            "open": float(k["o"]), "high": float(k["h"]),
                            "low":  float(k["l"]), "close": float(k["c"]),
                            "volume": float(k["v"]),
                        }
                        df = _upsert_ohlcv(df, ts, row)
                    result[tf] = add_all_indicators(df, tf=tf)
                except Exception:
                    result[tf] = snapshot[tf]  # 실패 시 원본 유지
            return result

        tf_data = await asyncio.to_thread(_recompute_all, raw_snapshot, forming)
        return tf_data, price

    async def _run_forever(self):
        # 지수 백오프 — 고정 3초 였던 기존 구현은 Binance 다운 시 IP 차단 위험.
        # 실패 시 backoff *= 2, 성공 시 리셋. 최대 60초 cap.
        bootstrap_backoff = 5
        ws_backoff = 3
        while not self._stopped:
            if not self._ready.is_set():
                try:
                    await self._bootstrap()
                    bootstrap_backoff = 5  # 성공 시 리셋
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    print(f"[market-stream] bootstrap failed: {exc} — {bootstrap_backoff}초 후 재시도")
                    await asyncio.sleep(min(bootstrap_backoff, 60))
                    bootstrap_backoff = min(bootstrap_backoff * 2, 60)
                    continue

            try:
                await self._consume_stream()
                ws_backoff = 3  # 성공 종료 시 리셋
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[market-stream] websocket disconnected: {exc} — {ws_backoff}초 후 재연결")
                await asyncio.sleep(min(ws_backoff, 60))
                ws_backoff = min(ws_backoff * 2, 60)

    async def _bootstrap(self):
        """REST API로 초기 데이터 로드. 실패해도 WebSocket 기동은 계속한다."""
        # ── OHLCV 히스토리 fetch (실패 시 빈 dict로 폴백) ──
        try:
            tf_data = await asyncio.wait_for(
                asyncio.to_thread(_fetch_all, self.symbol),
                timeout=30,
            )
            print(f"[market-stream] bootstrap OHLCV 완료: {list(tf_data.keys())}")
        except asyncio.TimeoutError:
            print("[market-stream] bootstrap OHLCV timeout — 빈 데이터로 계속")
            tf_data = {}
        except Exception as exc:
            print(f"[market-stream] bootstrap OHLCV 실패: {exc} — 빈 데이터로 계속")
            tf_data = {}

        # ── 현재가 fetch (실패 허용) ──
        try:
            price = await asyncio.wait_for(
                asyncio.to_thread(fetch_current_price, self.symbol),
                timeout=8,
            )
            print(f"[market-stream] bootstrap 현재가: {price}")
        except Exception as exc:
            print(f"[market-stream] bootstrap 현재가 실패: {exc}")
            price = None

        async with self._lock:
            self._tf_data = tf_data
            self._price = price
            self._last_update = _now_label()

        # OHLCV 없어도 ready 설정 — WebSocket이 실시간으로 채움
        self._ready.set()
        print(f"[market-stream] ready! tf_data={list(tf_data.keys())} price={price}")

    def _bybit_subscribe_args(self) -> list[str]:
        """Bybit v5 WebSocket 구독 topic 목록 생성."""
        sym = self.symbol.upper()
        topics = [f"publicTrade.{sym}"]
        for tf in TIMEFRAMES:
            interval = _TF_TO_BYBIT.get(tf)
            if interval:
                topics.append(f"kline.{interval}.{sym}")
        return topics

    @staticmethod
    def _normalize_bybit_kline(kline: dict, interval: str) -> dict:
        """
        Bybit kline 데이터를 기존 Binance kline 포맷으로 변환.
        기존 _enqueue_kline / _process_kline / _forming_kline 로직을 그대로 재사용하기 위함.
        """
        tf = _BYBIT_TF_MAP.get(interval, "")
        return {
            "i": tf,
            "t": kline.get("start"),       # 캔들 시작 ms (Binance kline["t"])
            "o": kline.get("open"),
            "h": kline.get("high"),
            "l": kline.get("low"),
            "c": kline.get("close"),
            "v": kline.get("volume"),
            "x": kline.get("confirm", False),  # True = 확정 캔들 (Binance kline["x"])
        }

    async def _consume_stream(self):
        """
        Bybit Linear Perpetual WebSocket 스트림 수신.
        Binance fstream.binance.com 은 특정 AWS IP 대역에서 스트림 데이터를 무음 차단하므로
        Bybit stream.bybit.com 으로 교체. OHLCV 히스토리·현재가 초기화는 Binance REST 유지.
        """
        self._ws_connect_count += 1
        args = self._bybit_subscribe_args()
        print(f"[market-stream] Bybit WebSocket 연결 시도 #{self._ws_connect_count}: {BYBIT_MARKET_WS_URL}")
        print(f"[market-stream] 구독 topics: {args}")

        async with websockets.connect(
            BYBIT_MARKET_WS_URL,
            ping_interval=None,   # Bybit은 JSON ping {"op":"ping"} 사용 — 아래 _bybit_ping 태스크에서 처리
            ping_timeout=None,
            close_timeout=5,
            max_size=2**20,
        ) as ws:
            # 구독 메시지 전송
            await ws.send(json.dumps({"op": "subscribe", "args": args}))
            print(f"[market-stream] Bybit WebSocket 연결 성공 #{self._ws_connect_count}")

            # Bybit 연결 유지: 20초마다 JSON ping 필요 (미전송 시 30초 후 서버가 끊음)
            async def _bybit_ping():
                while True:
                    await asyncio.sleep(20)
                    try:
                        await ws.send(json.dumps({"op": "ping"}))
                    except Exception:
                        break

            ping_task = asyncio.create_task(_bybit_ping(), name="bybit-ping")
            try:
                async for raw_msg in ws:
                    payload = json.loads(raw_msg)
                    topic: str = payload.get("topic", "")
                    if not topic:
                        # pong / subscribe confirm — 무시
                        continue

                    if topic.startswith("publicTrade."):
                        # 실시간 체결 이벤트 → 가격 갱신
                        for trade in payload.get("data", []):
                            self._trade_count += 1
                            price = _safe(trade.get("p"))
                            if price is None:
                                continue
                            async with self._lock:
                                self._price = price
                                self._last_update = _now_label()
                            self._schedule_price_flush()

                    elif topic.startswith("kline."):
                        # topic 형식: "kline.{interval}.{symbol}"
                        parts = topic.split(".")
                        interval = parts[1] if len(parts) >= 2 else ""
                        for kline in payload.get("data", []):
                            self._kline_count += 1
                            normalized = self._normalize_bybit_kline(kline, interval)
                            await self._enqueue_kline(normalized)
            finally:
                ping_task.cancel()

    async def _handle_trade(self, data: dict):
        price = _safe(data.get("p"))
        if price is None:
            return

        async with self._lock:
            self._price = price
            self._last_update = _now_label()

        self._schedule_price_flush()

    async def _enqueue_kline(self, kline: dict):
        """
        확정 캔들(x=True)만 지표 재계산 큐에 넣는다.

        진행 중 캔들(x=False)은 무시 — 이유:
          1) RSI/MACD/BB 등 지표는 과거 확정 캔들 기준으로 계산되어
             현재 캔들이 진행 중일 때 재계산해도 의미있는 변화가 없다.
          2) 현재 캔들의 실시간 가격(close/high/low)은
             프론트엔드 patchLastCandlePrice가 aggTrade 가격으로 직접 처리한다.
          3) 초당 4회 지표 계산 → 이벤트 루프 과부하 → 실시간 가격 지연의 근본 원인.

        미확정 캔들은 OHLCV만 tf_data에 업데이트 (지표 계산 없이 빠르게).
        → 분석 실행 시 최신 OHLCV가 포함된 상태로 지표를 재계산해서 Claude에 전달.
        확정 빈도: 15m=15분, 1h=1시간, 4h=4시간, 1d=1일 → 하루 총 수십 회.
        """
        tf = kline.get("i")
        if tf not in TIMEFRAMES:
            return

        if not kline.get("x", False):
            # 미확정 캔들: _forming_kline에만 보관 — tf_data는 건드리지 않음
            # tf_data를 수정하면 지표 컬럼이 제거돼 KeyError 발생
            # 분석 시 get_analysis_inputs에서 _forming_kline을 합산해 처리
            self._forming_kline[tf] = kline
            return

        # 확정 캔들: 계산 중에 또 확정이 오면 최신 것으로 덮어씀
        self._pending_kline[tf] = kline
        if not self._kline_running.get(tf, False):
            self._kline_running[tf] = True
            asyncio.create_task(self._process_kline(tf), name=f"kline-{tf}")

    async def _process_kline(self, tf: str):
        """확정 캔들에 대해 지표를 재계산하고 브로드캐스트. pending 소진까지 반복."""
        try:
            while True:
                kline = self._pending_kline.pop(tf, None)
                if kline is None:
                    break

                timestamp = pd.to_datetime(int(kline["t"]), unit="ms")
                row = {
                    "open":   float(kline["o"]),
                    "high":   float(kline["h"]),
                    "low":    float(kline["l"]),
                    "close":  float(kline["c"]),
                    "volume": float(kline["v"]),
                }

                async with self._lock:
                    current = self._tf_data.get(tf)

                def _compute(cur=current, ts=timestamp, r=row, _tf=tf):
                    up = _upsert_ohlcv(cur, ts, r)
                    return add_all_indicators(up, tf=_tf)

                updated = await asyncio.to_thread(_compute)

                async with self._lock:
                    self._tf_data[tf] = updated
                    self._last_update = _now_label()
                    self._dirty_tfs.add(tf)
                    if tf == "1h":
                        self._needs_full_refresh = True

                self._schedule_market_flush()
        finally:
            self._kline_running[tf] = False
            if self._pending_kline.get(tf):
                self._kline_running[tf] = True
                asyncio.create_task(self._process_kline(tf), name=f"kline-{tf}")

    def _schedule_price_flush(self):
        if self._price_flush_task and not self._price_flush_task.done():
            return
        self._price_flush_task = asyncio.create_task(self._flush_price_update())

    def _schedule_market_flush(self):
        if self._market_flush_task and not self._market_flush_task.done():
            return
        self._market_flush_task = asyncio.create_task(self._flush_market_update())

    async def _periodic_price_tick(self):
        """1초 주기 가격 + 진행 중 캔들 OHLCV 브로드캐스트.

        price: 실시간 aggTrade 가격
        forming: 타임프레임별 현재 캔들 OHLCV (Binance kline 기준)
                 → 프론트가 마지막 캔들을 실시간으로 갱신하는 데 사용
        지표(RSI·MACD 등)는 캔들 확정 시에만 재계산 — 여기서는 포함하지 않음
        """
        while not self._stopped:
            await asyncio.sleep(1.0)
            if self._stopped:
                break
            async with self._lock:
                price = self._price
                last_update = self._last_update
                forming_raw = dict(self._forming_kline)
            if price is None:
                continue

            # 진행 중 캔들 OHLCV 직렬화 (계산 없음 — 단순 숫자 변환)
            forming = {}
            for tf, k in forming_raw.items():
                try:
                    forming[tf] = {
                        "open":   float(k["o"]),
                        "high":   float(k["h"]),
                        "low":    float(k["l"]),
                        "close":  float(k["c"]),
                        "volume": float(k["v"]),
                    }
                except Exception:
                    pass

            await self._broadcast({"type": "price", "data": {
                "price": price,
                "last_update": last_update,
                "forming": forming,   # tf → {open, high, low, close, volume}
            }})

    async def _forming_indicator_refresh(self):
        """2초마다 진행 중 캔들 포함 지표를 재계산해 브로드캐스트.

        확정 캔들 처리(_process_kline)와 독립적으로 동작.
        tf_data는 수정하지 않고 스냅샷 + forming_kline으로 임시 계산 후 브로드캐스트.
        이벤트 루프는 비블로킹 — 계산은 asyncio.to_thread 스레드풀에서 실행.
        """
        while not self._stopped:
            await asyncio.sleep(2.0)
            if self._stopped or not self._ready.is_set():
                continue

            async with self._lock:
                if not self._forming_kline:
                    continue  # forming kline 아직 없음 (서버 막 시작)
                forming_raw = dict(self._forming_kline)
                tf_snap = {tf: df.copy(deep=True) for tf, df in self._tf_data.items()}
                price = self._price
                last_update = self._last_update

            def _compute(snap, forming):
                result = {}
                for tf, df in snap.items():
                    k = forming.get(tf)
                    if k:
                        ts = pd.to_datetime(int(k["t"]), unit="ms")
                        row = {
                            "open":   float(k["o"]), "high": float(k["h"]),
                            "low":    float(k["l"]), "close": float(k["c"]),
                            "volume": float(k["v"]),
                        }
                        df = _upsert_ohlcv(df, ts, row)
                    try:
                        result[tf] = add_all_indicators(df, tf=tf)
                    except Exception:
                        result[tf] = df
                return result

            try:
                updated = await asyncio.to_thread(_compute, tf_snap, forming_raw)
                payload = await asyncio.to_thread(
                    build_market_payload,
                    updated, price, last_update,
                    chart_tfs=TIMEFRAMES,
                    include_overview=True,
                )
                await self._broadcast({"type": "market", "data": payload})
            except Exception as exc:
                print(f"[forming-refresh] 오류: {exc}")

    async def _flush_price_update(self):
        """aggTrade 수신 후 100ms 디바운스 — 최신 가격을 빠르게 브로드캐스트."""
        await asyncio.sleep(0.10)
        async with self._lock:
            price = self._price
            last_update = self._last_update
        if price is None:
            return
        await self._broadcast({"type": "price", "data": {
            "price": price,
            "last_update": last_update,
        }})

    async def _flush_market_update(self):
        await asyncio.sleep(0.35)
        # 락 안에서는 데이터만 복사 — CPU 연산은 락 밖에서 스레드로 실행
        async with self._lock:
            dirty_tfs = set(self._dirty_tfs)
            full_refresh = self._needs_full_refresh
            self._dirty_tfs.clear()
            self._needs_full_refresh = False

            if not dirty_tfs and not full_refresh:
                return

            tf_data_snapshot = dict(self._tf_data)
            price_snapshot = self._price
            last_update_snapshot = self._last_update

        # build_market_payload 는 CPU 집약적 — 스레드풀에서 실행하여 이벤트 루프 비블로킹
        payload = await asyncio.to_thread(
            build_market_payload,
            tf_data_snapshot,
            price_snapshot,
            last_update_snapshot,
            chart_tfs=TIMEFRAMES if full_refresh else dirty_tfs,
            include_overview=full_refresh,
        )

        await self._broadcast({"type": "market", "data": payload})

    async def _broadcast(self, message: dict):
        async with self._lock:
            listeners = list(self._listeners)

        for listener in listeners:
            if listener.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    listener.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                listener.put_nowait(message)


_market_stream = MarketStreamManager(DEFAULT_SYMBOL)


ACCOUNT_PERIODIC_REFRESH_SECS = 60  # 주기적 REST 폴링 간격 (초)


class AccountStreamManager:
    def __init__(self):
        self._payload: dict | None = None
        self._ready = asyncio.Event()
        self._lock = asyncio.Lock()
        self._listeners: set[asyncio.Queue] = set()
        self._runner_task: asyncio.Task | None = None
        self._refresh_task: asyncio.Task | None = None
        self._periodic_task: asyncio.Task | None = None
        self._pending_refresh = False
        self._last_refresh_started_at = 0.0
        self._stopped = False

    async def start(self):
        if self._runner_task and not self._runner_task.done():
            return
        self._stopped = False
        self._runner_task = asyncio.create_task(self._run_forever(), name="binance-account-stream")
        self._periodic_task = asyncio.create_task(self._periodic_refresh(), name="binance-account-periodic")

    async def stop(self):
        self._stopped = True
        tasks = [self._runner_task, self._refresh_task, self._periodic_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
        for task in tasks:
            if task:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        with contextlib.suppress(Exception):
            await asyncio.to_thread(close_user_data_stream)

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=8)
        async with self._lock:
            self._listeners.add(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue):
        async with self._lock:
            self._listeners.discard(q)

    async def wait_until_ready(self):
        await self._ready.wait()

    def is_ready(self) -> bool:
        return self._ready.is_set()

    async def get_snapshot(self) -> dict:
        await self.wait_until_ready()
        async with self._lock:
            return copy.deepcopy(self._payload or {})

    async def _run_forever(self):
        # 지수 백오프 — Binance 다운 시 IP 차단 방지.
        bootstrap_backoff = 5
        ws_backoff = 3
        while not self._stopped:
            if not self._ready.is_set():
                try:
                    await self._refresh_payload(delay=0, broadcast=False)
                    bootstrap_backoff = 5
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    print(f"[account-stream] bootstrap failed: {exc} — {bootstrap_backoff}초 후 재시도")
                    await asyncio.sleep(min(bootstrap_backoff, 60))
                    bootstrap_backoff = min(bootstrap_backoff * 2, 60)
                    continue

            if not runtime_config.BINANCE_API_KEY:
                await asyncio.sleep(30)
                continue

            try:
                await self._consume_stream()
                ws_backoff = 3
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                msg = str(exc)
                if "인증 실패 (401)" in msg:
                    print(f"[account-stream] auth failed, using REST snapshot fallback: {exc}")
                    with contextlib.suppress(Exception):
                        await self._refresh_payload(delay=0, broadcast=True)
                    await asyncio.sleep(10)
                else:
                    print(f"[account-stream] websocket disconnected: {exc} — {ws_backoff}초 후 재연결")
                    # 재연결 전 REST 스냅샷 갱신 — 끊긴 동안의 변경사항 반영
                    with contextlib.suppress(Exception):
                        await self._refresh_payload(delay=0, broadcast=True)
                    await asyncio.sleep(min(ws_backoff, 60))
                    ws_backoff = min(ws_backoff * 2, 60)

    async def _periodic_refresh(self):
        """WebSocket 이벤트 여부와 무관하게 주기적으로 REST 스냅샷을 갱신."""
        while not self._stopped:
            await asyncio.sleep(ACCOUNT_PERIODIC_REFRESH_SECS)
            if self._stopped:
                break
            if not runtime_config.BINANCE_API_KEY:
                continue
            with contextlib.suppress(Exception):
                await self._refresh_payload(delay=0, broadcast=True)

    async def _consume_stream(self):
        listen_key = await asyncio.to_thread(open_user_data_stream)
        ws_url = f"{ACCOUNT_STREAM_WS_BASE}/{listen_key}"

        try:
            async with websockets.connect(
                ws_url,
                ping_interval=None,
                ping_timeout=None,
                close_timeout=5,
                max_size=2**20,
            ) as ws:
                while not self._stopped:
                    try:
                        raw_msg = await asyncio.wait_for(
                            ws.recv(),
                            timeout=ACCOUNT_STREAM_KEEPALIVE_SECS,
                        )
                    except asyncio.TimeoutError:
                        await asyncio.to_thread(keepalive_user_data_stream)
                        continue

                    payload = json.loads(raw_msg)
                    event_type = payload.get("e")
                    if event_type in {
                        "ACCOUNT_UPDATE",
                        "ACCOUNT_CONFIG_UPDATE",
                        "MARGIN_CALL",
                        "ORDER_TRADE_UPDATE",
                        "TRADE_LITE",
                    }:
                        self._schedule_refresh()
                    elif event_type == "listenKeyExpired":
                        raise RuntimeError("listenKey expired")
        finally:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(close_user_data_stream)

    def _schedule_refresh(self):
        if self._refresh_task and not self._refresh_task.done():
            self._pending_refresh = True
            return
        self._refresh_task = asyncio.create_task(self._refresh_payload())

    async def _refresh_payload(
        self,
        delay: float = ACCOUNT_REFRESH_DEBOUNCE_SECS,
        broadcast: bool = True,
    ):
        loop = asyncio.get_running_loop()
        min_delay = max(
            0.0,
            ACCOUNT_REFRESH_MIN_INTERVAL_SECS - (loop.time() - self._last_refresh_started_at),
        )
        wait_for = max(delay, min_delay)
        if wait_for > 0:
            await asyncio.sleep(wait_for)

        self._last_refresh_started_at = loop.time()
        try:
            payload = await asyncio.to_thread(_build_account_payload)
            async with self._lock:
                self._payload = payload
                self._ready.set()
            if broadcast:
                await self._broadcast({"type": "account", "data": payload})
        finally:
            pending = self._pending_refresh
            self._pending_refresh = False
            if pending and not self._stopped:
                self._refresh_task = asyncio.create_task(
                    self._refresh_payload(delay=ACCOUNT_REFRESH_DEBOUNCE_SECS)
                )
            else:
                self._refresh_task = None

    async def _broadcast(self, message: dict):
        async with self._lock:
            listeners = list(self._listeners)

        for listener in listeners:
            if listener.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    listener.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                listener.put_nowait(message)


_account_stream = AccountStreamManager()


class MacroSnapshotManager:
    def __init__(self):
        self._payload: dict | None = None
        self._ready = asyncio.Event()
        self._lock = asyncio.Lock()
        self._runner_task: asyncio.Task | None = None
        self._stopped = False

    async def start(self):
        if self._runner_task and not self._runner_task.done():
            return
        self._stopped = False
        self._runner_task = asyncio.create_task(self._run_forever(), name="macro-snapshot-refresh")

    async def stop(self):
        self._stopped = True
        task = self._runner_task
        if task and not task.done():
            task.cancel()
        if task:
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def wait_until_ready(self):
        await self._ready.wait()

    def is_ready(self) -> bool:
        return self._ready.is_set()

    async def get_snapshot(self) -> dict:
        await self.wait_until_ready()
        async with self._lock:
            return copy.deepcopy(self._payload or {})

    async def _run_forever(self):
        while not self._stopped:
            sleep_for = MACRO_REFRESH_SECS
            try:
                payload = await asyncio.to_thread(fetch_macro_context)
                async with self._lock:
                    self._payload = payload
                    self._ready.set()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[macro-refresh] fetch failed: {exc}")
                sleep_for = 120
            await asyncio.sleep(sleep_for)


_macro_snapshot = MacroSnapshotManager()


# 수동 분석 버튼 쿨다운: 10분
MANUAL_COOLDOWN_SECS = 600
MANUAL_COOLDOWN_STATE_PATH = os.path.join(BASE_DIR, "data", "manual_cooldown.json")


def _load_manual_cooldown_time() -> float:
    """서버 재시작 후 수동 클릭 시각을 복원. 없으면 0.0 반환."""
    try:
        with open(MANUAL_COOLDOWN_STATE_PATH, "r", encoding="utf-8") as f:
            return float(json.load(f).get("last_manual_started_at", 0.0))
    except Exception:
        return 0.0


def _save_manual_cooldown_time(ts: float) -> None:
    try:
        os.makedirs(os.path.dirname(MANUAL_COOLDOWN_STATE_PATH), exist_ok=True)
        with open(MANUAL_COOLDOWN_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump({"last_manual_started_at": ts}, f)
    except Exception as exc:
        import logging as _log
        _log.getLogger(__name__).warning("manual cooldown 저장 실패 — %s", exc)


class AnalysisManager:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._job: dict | None = None
        self._task: asyncio.Task | None = None
        self._latest_result: dict | None = _load_latest_analysis()
        # deepcopy 방지용 JSON 캐시 — 시작 시 디스크 로드 결과도 즉시 직렬화
        _lr = self._latest_result
        self._latest_result_json: str | None = (
            json.dumps(_lr, ensure_ascii=False) if _lr is not None else None
        )
        self._last_manual_started_at: float = _load_manual_cooldown_time()

    async def stop(self):
        task = None
        async with self._lock:
            task = self._task
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def start_job(self, bypass_cooldown: bool = False) -> tuple[dict, bool]:
        async with self._lock:
            if self._job and self._job["status"] in {"pending", "running"}:
                return self._serialize_job(self._job), False

            # 쿨다운 체크 (스케줄러는 bypass)
            if not bypass_cooldown:
                remaining = max(0.0, MANUAL_COOLDOWN_SECS - (time.time() - self._last_manual_started_at))
                if remaining > 0:
                    raise HTTPException(
                        status_code=429,
                        detail={
                            "reason": "cooldown",
                            "cooldown_remaining_secs": int(remaining),
                        },
                    )
                self._last_manual_started_at = time.time()
                _save_manual_cooldown_time(self._last_manual_started_at)

            started_at = _now_iso()
            job = {
                "id": uuid.uuid4().hex,
                "status": "running",
                "step": 0,
                # phase/phase_detail: step 2 내부의 세부 진행 상태
                # (예: "bull", "bear", "final"). UI 진행 표시용.
                "phase": None,
                "phase_detail": None,
                "error": None,
                "result": None,
                "started_at": started_at,
                "updated_at": started_at,
                "completed_at": None,
            }
            self._job = job
            self._task = asyncio.create_task(
                self._run_job(job["id"]),
                name=f"analysis-job-{job['id'][:8]}",
            )
            return self._serialize_job(job), True

    async def get_status(self, include_latest: bool = False) -> dict:
        async with self._lock:
            response = {
                "job": self._serialize_job(self._job),
                "cooldown_remaining_secs": int(max(0.0, MANUAL_COOLDOWN_SECS - (time.time() - self._last_manual_started_at))),
            }
            if (
                self._job
                and self._job["status"] == "completed"
                and self._job.get("result") is not None
            ):
                # deepcopy 대신 JSON 역직렬화 — 더 빠름
                result_json = self._job.get("_result_json")
                if result_json is None:
                    result_json = json.dumps(self._job["result"], ensure_ascii=False)
                    self._job["_result_json"] = result_json
                response["result"] = json.loads(result_json)
            elif include_latest and self._latest_result_json is not None:
                response["latest_result"] = json.loads(self._latest_result_json)
            return response

    def _serialize_job(self, job: dict | None) -> dict | None:
        if not job:
            return None
        return {
            "id": job["id"],
            "status": job["status"],
            "step": job["step"],
            "phase": job.get("phase"),
            "phase_detail": job.get("phase_detail"),
            "error": job["error"],
            "started_at": job["started_at"],
            "updated_at": job["updated_at"],
            "completed_at": job["completed_at"],
        }

    async def _set_step(self, job_id: str, step: int):
        async with self._lock:
            if not self._job or self._job["id"] != job_id:
                return
            self._job["status"] = "running"
            self._job["step"] = step
            self._job["updated_at"] = _now_iso()

    async def _set_phase(self, job_id: str, phase: str, detail: str):
        """step 2 내부의 세부 진행 상태(phase)를 갱신한다.
        토론 진행(Bull/Bear) 중 UI 프로그레스 표시를 위한 용도."""
        async with self._lock:
            if not self._job or self._job["id"] != job_id:
                return
            self._job["phase"] = phase
            self._job["phase_detail"] = detail
            self._job["updated_at"] = _now_iso()

    async def _complete(self, job_id: str, payload: dict):
        completed_at = _now_iso()
        async with self._lock:
            if not self._job or self._job["id"] != job_id:
                return
            self._job["status"] = "completed"
            self._job["step"] = 3
            self._job["error"] = None
            self._job["result"] = copy.deepcopy(payload)
            self._job["updated_at"] = completed_at
            self._job["completed_at"] = completed_at
            self._latest_result = copy.deepcopy(payload)
            self._latest_result_json = json.dumps(self._latest_result, ensure_ascii=False)

    async def _fail(self, job_id: str, message: str, status: str = "error"):
        failed_at = _now_iso()
        async with self._lock:
            if not self._job or self._job["id"] != job_id:
                return
            self._job["status"] = status
            self._job["error"] = message
            self._job["updated_at"] = failed_at
            self._job["completed_at"] = failed_at

    async def _run_job(self, job_id: str):
        loop = asyncio.get_event_loop()
        try:
            await self._set_step(job_id, 1)
            if _market_stream.is_ready():
                tf_data, price = await _market_stream.get_analysis_inputs()
            else:
                tf_data = await loop.run_in_executor(_executor, _fetch_all, _current_symbol)
                try:
                    price = await loop.run_in_executor(_executor, fetch_current_price, _current_symbol)
                except Exception:
                    price = None

            if _macro_snapshot.is_ready():
                macro_snapshot = await _macro_snapshot.get_snapshot()
            else:
                macro_snapshot = await asyncio.to_thread(fetch_macro_context)

            await self._set_step(job_id, 2)

            # 토론 진행률 콜백 — 워커 스레드에서 호출되므로 이벤트 루프로 넘긴다.
            def _progress_cb(phase: str, detail: str):
                try:
                    asyncio.run_coroutine_threadsafe(
                        self._set_phase(job_id, phase, detail), loop
                    )
                except Exception:
                    # 진행률 실패가 분석 흐름을 끊어서는 안 된다.
                    pass

            analysis = await loop.run_in_executor(
                _executor,
                functools.partial(
                    run_full_analysis,
                    tf_data,
                    macro_snapshot,
                    progress_cb=_progress_cb,
                ),
            )

            await self._set_step(job_id, 3)
            payload = await loop.run_in_executor(
                _executor, _build_payload, tf_data, price, analysis
            )
            await self._complete(job_id, payload)
            await asyncio.to_thread(_persist_latest_analysis, payload)
            await asyncio.to_thread(_persist_analysis_history, payload)

            # ── 포지션 청산 감지 → 리플렉션 실행 ──────
            global _prev_position_side
            if _auto_trader:
                try:
                    _cur_positions = await asyncio.to_thread(_auto_trader.get_positions)
                    _cur_side = _cur_positions[0].get("holdSide") if _cur_positions else None
                    if _prev_position_side and not _cur_side:
                        _lg.getLogger("auto-trader").info("[리플렉션] 포지션 청산 감지")
                        # 청산 결과 스냅샷 업데이트
                        try:
                            snap_path = os.path.join(BASE_DIR, "data", "trade_snapshots.jsonl")
                            acct = await asyncio.to_thread(_auto_trader.get_account)
                            equity = float(acct.get("equity", 0) or 0)
                            if os.path.exists(snap_path):
                                with open(snap_path) as f:
                                    lines = f.readlines()
                            else:
                                lines = []
                            if lines:
                                last = json.loads(lines[-1])
                                if not last.get("closed"):
                                    last["closed"] = True
                                    last["close_time"] = _now_iso()
                                    last["close_equity"] = equity
                                    entry_price = float(last.get("price", 0) or 0)
                                    close_price = float(acct.get("markPrice", 0) or 0)
                                    last["result_pct"] = round((equity - float(last.get("equity_at_entry", equity))) / max(float(last.get("equity_at_entry", equity)), 1) * 100, 2)
                                    lines[-1] = json.dumps(last, ensure_ascii=False) + "\n"
                                    with open(snap_path, "w") as f:
                                        f.writelines(lines)
                        except Exception:
                            pass
                        _tg("🔄 포지션 청산 감지 → 리플렉션 실행 중...")
                        asyncio.create_task(_run_reflection())
                    _prev_position_side = _cur_side
                except Exception:
                    pass

            # ── 포지션 있으면 분석 스킵 ──────
            if _auto_trader:
                try:
                    _cur_pos = await asyncio.to_thread(_auto_trader.get_positions)
                    if _cur_pos:
                        return
                except Exception:
                    pass

            # ── 미체결 주문 취소 (이전 지정가 주문 정리) ──────
            if _auto_trader:
                try:
                    await asyncio.to_thread(_auto_trader.client.cancel_all_tpsl, _auto_trader.symbol)
                except Exception:
                    pass

            # ── Bitget 자동매매 실행 ──────────────────────────
            await _run_auto_trade(analysis, price, tf_data)
            await _save_account_snapshot()

            # ── 텔레그램 분석 알림 (확신도 65% 이상)
            await asyncio.to_thread(
                telegram_alert.alert_analysis, analysis, price or 0,
                AUTO_TRADE_MIN_CONF
            )

            # ── 포지션 있을 때 관리 알림 + 거래소 TP/SL 동적 업데이트
            if _auto_trader:
                try:
                    positions = await asyncio.to_thread(_auto_trader.get_positions)
                    if positions:
                        trade_levels = analysis.get("trade_levels") or {}
                        new_sl = trade_levels.get("stop")
                        new_tp = trade_levels.get("target")
                        for p in positions:
                            side = p.get("holdSide", "")
                            # Bitget 가 string 으로 반환하는 경우 있음 — float 강제 변환
                            entry = float(p.get("averageOpenPrice", 0) or 0)
                            upnl  = float(p.get("unrealizedPL", 0) or 0)
                            emoji = "🟢" if upnl >= 0 else "🔴"
                            msg = f"📊 포지션 관리 알림\n{side.upper()} 진입가 ${entry:,.2f}\n미실현: {emoji}${upnl:+,.2f}\n"
                            if new_sl:
                                msg += f"AI 권고 손절: ${new_sl:,.2f}\n"
                            if new_tp:
                                msg += f"AI 권고 익절: ${new_tp:,.2f}"
                            _tg(msg)

                        # 거래소 TP/SL 실제 업데이트 — 본전 이동 + AI 권고 반영.
                        # 환경변수 AUTO_TRADE_DYNAMIC_TPSL=true 여야 동작 (기본 false).
                        if os.environ.get("AUTO_TRADE_DYNAMIC_TPSL", "false").lower() == "true":
                            try:
                                update_result = await asyncio.to_thread(
                                    _auto_trader.update_position_tpsl,
                                    new_tp, new_sl,
                                    float(os.environ.get("AUTO_TRADE_BREAKEVEN_PCT", "1.0")),
                                )
                                if update_result.get("updated"):
                                    _tg(
                                        f"🔧 TP/SL 거래소 반영\n"
                                        f"{update_result.get('side','').upper()} 진입가 ${update_result.get('entry',0):,.2f}\n"
                                        f"새 TP: ${update_result.get('applied_tp') or 0:,.2f} | "
                                        f"새 SL: ${update_result.get('applied_sl') or 0:,.2f}"
                                    )
                            except Exception as _exc:
                                print(f"[update_tpsl] 실패: {_exc}", flush=True)
                except Exception:
                    pass

        except asyncio.CancelledError:
            await self._fail(job_id, "분석 작업이 취소되었습니다.", status="cancelled")
            raise
        except Exception as exc:
            import traceback as _tb
            _tb.print_exc()
            _tg(f"⚠️ 분석 오류\n{type(exc).__name__}: {str(exc)[:200]}")
            await self._fail(job_id, str(exc))


_analysis_manager = AnalysisManager()


# ══════════════════════════════════════════════
# 서버 사이드 자동 분석 스케줄러
# ══════════════════════════════════════════════
import datetime as _dt

SCHEDULE_STATE_PATH = os.path.join(BASE_DIR, "data", "schedule_state.json")

class ScheduleManager:
    """서버에서 30분마다 분석을 실행하는 백그라운드 스케줄러."""

    INTERVAL_MIN = 240  # 4시간

    def __init__(self):
        self._enabled: bool = False
        self._next_run_at: str | None = None
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._load_state()

    def _load_state(self):
        """서버 재시작 후에도 설정을 복원한다."""
        try:
            if os.path.exists(SCHEDULE_STATE_PATH):
                with open(SCHEDULE_STATE_PATH, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                self._enabled = bool(saved.get("enabled", False))
        except Exception:
            pass

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(SCHEDULE_STATE_PATH), exist_ok=True)
            with open(SCHEDULE_STATE_PATH, "w", encoding="utf-8") as f:
                json.dump({"enabled": self._enabled, "interval_min": self.INTERVAL_MIN}, f)
        except Exception:
            pass

    def status(self) -> dict:
        return {
            "enabled":      self._enabled,
            "interval_min": self.INTERVAL_MIN,
            "next_run_at":  self._next_run_at,
        }

    async def set_schedule(self, enabled: bool, interval_min: int = INTERVAL_MIN):
        async with self._lock:
            self._enabled = enabled
            self._save_state()
            await self._restart_task()

    async def start(self):
        """앱 시작 시 호출. 저장된 설정이 enabled면 태스크 재개."""
        if self._enabled:
            await self._restart_task()

    async def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _restart_task(self):
        await self.stop()
        if self._enabled:
            self._task = asyncio.create_task(
                self._loop(), name="schedule-auto-analyze"
            )

    async def _loop(self):
        while True:
            # 다음 정각(0, 4, 8, 12, 16, 20시 UTC) 까지 대기
            now     = _dt.datetime.now(_dt.timezone.utc)
            hour    = now.hour
            # INTERVAL_MIN=240 기준 → 0,4,8,12,16,20시 정각
            next_h  = ((hour // self.INTERVAL_MIN * 60 + 1) * self.INTERVAL_MIN * 60 // 60) % 24
            # 실제로는 간격(시간)으로 다음 정각 계산
            interval_h = self.INTERVAL_MIN // 60
            next_h  = ((hour // interval_h) + 1) * interval_h % 24
            next_dt = now.replace(minute=0, second=0, microsecond=0)
            if next_h <= hour:
                next_dt += _dt.timedelta(days=1)
            next_dt = next_dt.replace(hour=next_h)
            wait_sec = (next_dt - now).total_seconds()
            self._next_run_at = next_dt.isoformat()
            print(f"[scheduler] 다음 실행: {next_dt.strftime('%Y-%m-%d %H:%M UTC')} ({int(wait_sec//60)}분 후)")
            try:
                await asyncio.sleep(wait_sec)
            except asyncio.CancelledError:
                self._next_run_at = None
                raise
            # 분석 실행 (이미 진행 중이면 skip, 스케줄러는 쿨다운 우회)
            _, started = await _analysis_manager.start_job(bypass_cooldown=True)
            if not started:
                print("[scheduler] analysis already running – skipped")


_schedule_manager = ScheduleManager()


# Bitget AutoTrader singleton
def _make_auto_trader():
    if not _BITGET_AVAILABLE:
        return None
    if not (BITGET_API_KEY and BITGET_SECRET_KEY and BITGET_PASSPHRASE):
        return None
    return BitgetAutoTrader(
        api_key=BITGET_API_KEY,
        secret_key=BITGET_SECRET_KEY,
        passphrase=BITGET_PASSPHRASE,
        symbol=DEFAULT_SYMBOL,
        usdt_per_trade=AUTO_TRADE_USDT,
        leverage=AUTO_TRADE_LEVERAGE,
        min_confidence=AUTO_TRADE_MIN_CONF,
        use_tp=AUTO_TRADE_USE_TP,
        use_sl=AUTO_TRADE_USE_SL,
    )

_auto_trader = _make_auto_trader()
_auto_trade_enabled = AUTO_TRADE_ENABLED
_auto_trade_lock = asyncio.Lock()
_last_trade_result = {}
_prev_position_side = None  # 이전 포지션 방향 추적
_trade_log = []
_TRADE_LOG_MAX = 100
_AT_SETTINGS_PATH = os.path.join(BASE_DIR, "data", "auto_trade_settings.json")
_SYMBOL_PATH = os.path.join(BASE_DIR, "data", "symbol.json")
_current_symbol = DEFAULT_SYMBOL

def _load_symbol():
    global _current_symbol
    try:
        if os.path.exists(_SYMBOL_PATH):
            with open(_SYMBOL_PATH) as f:
                _current_symbol = json.load(f).get("symbol", DEFAULT_SYMBOL)
    except Exception:
        pass

def _save_symbol(symbol: str):
    global _current_symbol
    _current_symbol = symbol
    try:
        os.makedirs(os.path.dirname(_SYMBOL_PATH), exist_ok=True)
        with open(_SYMBOL_PATH, "w") as f:
            json.dump({"symbol": symbol}, f)
    except Exception:
        pass

_load_symbol()
_cached_account = {}

def _load_at_settings():
    global _auto_trade_enabled
    try:
        if os.path.exists(_AT_SETTINGS_PATH):
            with open(_AT_SETTINGS_PATH) as f:
                s = json.load(f)
            _auto_trade_enabled = s.get("enabled", AUTO_TRADE_ENABLED)
            if _auto_trader:
                _auto_trader.usdt_per_trade = s.get("usdt_per_trade", AUTO_TRADE_USDT)
                _auto_trader.leverage = s.get("leverage", AUTO_TRADE_LEVERAGE)
                _auto_trader.min_confidence = s.get("min_confidence", AUTO_TRADE_MIN_CONF)
                _auto_trader.use_tp = s.get("use_tp", AUTO_TRADE_USE_TP)
                _auto_trader.use_sl = s.get("use_sl", AUTO_TRADE_USE_SL)
    except Exception as e:
        print("[auto-trade] 설정 로드 실패:", e)

def _save_at_settings():
    try:
        os.makedirs(os.path.dirname(_AT_SETTINGS_PATH), exist_ok=True)
        cfg = _auto_trader
        s = {"enabled": _auto_trade_enabled, "usdt_per_trade": cfg.usdt_per_trade if cfg else AUTO_TRADE_USDT, "leverage": cfg.leverage if cfg else AUTO_TRADE_LEVERAGE, "min_confidence": cfg.min_confidence if cfg else AUTO_TRADE_MIN_CONF, "use_tp": cfg.use_tp if cfg else AUTO_TRADE_USE_TP, "use_sl": cfg.use_sl if cfg else AUTO_TRADE_USE_SL}
        with open(_AT_SETTINGS_PATH, "w") as f:
            json.dump(s, f)
    except Exception as e:
        print("[auto-trade] 설정 저장 실패:", e)

_load_at_settings()



# ══════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════
async def _run_reflection():
    """포지션 청산 시 리플렉션 실행.

    예전엔 http://localhost:8000/api/reflect 자기 자신 호출이었으나:
    - PaaS 배포 환경(Railway/Render/Fly)에선 포트 8000 가정이 깨짐
    - auth_middleware 가 /api/reflect 를 막아 401
    → 함수 직접 호출이 모든 환경에서 안전.
    """
    _rlog = _lg.getLogger("reflection")
    try:
        result = await _run_reflect_background()
        # 버그3 수정: ok=False 반환(가격 수집 실패 등)을 조용히 무시하지 않고 로깅
        if isinstance(result, dict) and not result.get("ok"):
            _rlog.warning("리플렉션 비정상 종료: %s", result.get("error", result))
        else:
            processed = result.get("processed", 0) if isinstance(result, dict) else "?"
            _rlog.info("리플렉션 완료: processed=%s", processed)
    except Exception as e:
        _rlog.warning("리플렉션 실패: %s", e)

async def _reflection_loop():
    """6시간마다 자동으로 /api/reflect 를 내부 호출해 리플렉션을 수행한다."""
    import logging as _rlog
    _rlog = _rlog.getLogger("reflection-loop")
    INTERVAL = 6 * 60 * 60  # 6시간
    await asyncio.sleep(60)  # 서버 시작 후 1분 대기 (스트림 안정화)
    while True:
        if _REFLECTION_ENABLED:
            try:
                _rlog.info("[reflection-loop] 자동 리플렉션 시작")
                _loop_result = await _run_reflect_background()
                if isinstance(_loop_result, dict) and not _loop_result.get("ok"):
                    _rlog.warning("[reflection-loop] 비정상 종료: %s", _loop_result.get("error", _loop_result))
                else:
                    _processed = _loop_result.get("processed", 0) if isinstance(_loop_result, dict) else "?"
                    _rlog.info("[reflection-loop] 완료: processed=%s", _processed)
            except Exception as _exc:
                _rlog.warning("[reflection-loop] 실패 — %s", _exc)
        await asyncio.sleep(INTERVAL)



# JSONL 파일 동시 쓰기 보호용 Lock — to_thread 로 호출되는 동기 함수의
# append 경합 방지. asyncio 환경이라도 같은 파일에 여러 태스크가 동시 진입 가능.
_jsonl_write_lock = asyncio.Lock()

async def _save_account_snapshot():
    """계좌 스냅샷을 account_history.jsonl에 저장."""
    try:
        if _auto_trader is None:
            return
        acct = await asyncio.to_thread(_auto_trader.get_account)
        equity = float(acct.get("equity", 0) or 0)
        if equity <= 0:
            return
        import time as _time
        snapshot = {
            "observed_ts":    _time.time(),
            "account_equity": equity,
            "available":      float(acct.get("available", 0) or 0),
            "today_total_pnl": float(acct.get("todayProfitLoss", 0) or 0),
            "today_pnl_pct":  0,
        }
        history_path = os.path.join(BASE_DIR, "data", "account_history.jsonl")
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        async with _jsonl_write_lock:
            with open(history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
    except Exception as e:
        print("[snapshot] 저장 실패:", e)

async def _run_auto_trade(analysis: dict, price: float | None, tf_data: dict | None):
    """분석 완료 후 Bitget 자동매매 실행 (비동기 wrapping)."""
    global _last_trade_result, _trade_log, _auto_trade_enabled

    if not _auto_trade_enabled or _auto_trader is None or price is None:
        return

    signal     = analysis.get("signal", "홀드")
    confidence = analysis.get("confidence", 0)
    trade_levels = analysis.get("trade_levels")

    # AI 권장 레버리지는 로그용 — 정확한 위치(claude_leverage)에서 읽음.
    # 기존엔 summary(한줄요약)에서 정규식으로 찾으려 했으나 거기엔 거의 없는 죽은 코드.
    ai_lev = analysis.get("claude_leverage")
    if ai_lev and _auto_trader and 1 <= int(ai_lev) <= 20:
        print(f"[AutoTrade] AI 권장 레버리지: {ai_lev}배 (현재 UI 설정: {_auto_trader.leverage}배 — 유지)")

    async with _auto_trade_lock:
        try:
            result = await asyncio.to_thread(
                _auto_trader.execute,
                signal, confidence, price, trade_levels, tf_data
            )
        except Exception as exc:
            import logging as _lg
            import traceback as _tb2; _tb2.print_exc()
            _lg.getLogger("auto-trader").error("자동매매 오류: %s", exc)
            result = {
                "action": "error", "reason": str(exc),
                "signal": signal, "confidence": confidence,
            }

        result["executed_at"] = _now_iso()
        result["price"] = price
        _last_trade_result = result

        # 텔레그램 매매 알림
        telegram_alert.alert_trade(result)
        telegram_alert.alert_trade_rejected({**result, "signal": signal, "confidence": confidence, "price": price or 0})

        # ── 실제 진입된 거래 메모리 기록 ──────────────────────────
        # 리플렉션이 실매매 결과를 학습할 수 있도록 analyst 메모리에 기록
        if result.get("action") in ("long", "short"):
            try:
                from agents.memory import get_memory as _gm
                _trade_mem = _gm("analyst")
                _action    = result.get("action")
                _tp        = result.get("tp")
                _sl        = result.get("sl")
                _rr        = result.get("rr")
                _entry     = result.get("entry") or price

                _trade_situation = (
                    f"[실매매 진입 — {_action.upper()}]\n"
                    f"진입가: ${_entry:,.2f} | 확신도: {confidence}% | 레짐: {analysis.get('regime','')}\n"
                    f"TP: {'$'+f'{_tp:,.2f}' if _tp else 'N/A'} | "
                    f"SL: {'$'+f'{_sl:,.2f}' if _sl else 'N/A'} | "
                    f"R:R: {_rr if _rr else 'N/A'}\n"
                    f"분석요약: {str(analysis.get('summary',''))[:300]}"
                )
                # TP/SL 가 None 일 때 f-string 포맷 에러 방지 — N/A 로 폴백.
                # 이게 있어야 진입은 성공했으나 메모리 기록이 깨져 reflection
                # 자료가 사라지는 사고를 막을 수 있음.
                _tp_str = f"${_tp:,.2f}" if _tp else "N/A"
                _sl_str = f"${_sl:,.2f}" if _sl else "N/A"
                _trade_advice = (
                    f"{_action.upper()} 진입 실행\n"
                    f"진입가 ${_entry:,.2f} | TP {_tp_str} | SL {_sl_str}"
                )
                _trade_mem.add_situation(
                    situation=_trade_situation,
                    advice=_trade_advice,
                    outcome="",
                    meta={
                        "price_at_analysis": float(_entry),
                        "trade_levels": trade_levels,
                        "signal": signal,
                        "confidence": confidence,
                        "action": _action,
                        "rr": _rr,
                        "tp": _tp,
                        "sl": _sl,
                        "missed": False,
                    }
                )
                import logging as _lg2
                _lg2.getLogger("auto-trader").info("[AutoTrade] 실매매 메모리 기록 완료")
            except Exception as _me2:
                import logging as _lg2
                _lg2.getLogger("auto-trader").warning("[AutoTrade] 실매매 메모리 기록 실패 — %s", _me2)

        # 진입 시 시장 스냅샷 저장 (학습용)
        if result.get("action") in ("long", "short"):
            try:
                # equity_at_entry 를 스냅샷에 기록해두어야
                # 청산 시 result_pct 계산이 정상 동작함
                try:
                    _snap_acct = await asyncio.to_thread(_auto_trader.get_account)
                    _entry_equity = float(_snap_acct.get("equity", 0) or 0)
                except Exception:
                    _entry_equity = 0.0
                snap = {
                    "timestamp": _now_iso(),
                    "action": result.get("action"),
                    "price": price,
                    "tp": result.get("tp"),
                    "sl": result.get("sl"),
                    "confidence": confidence,
                    "signal": signal,
                    "summary": analysis.get("summary", ""),
                    "trade_levels": trade_levels,
                    "regime": analysis.get("regime", ""),
                    "equity_at_entry": _entry_equity,
                }
                snap_path = os.path.join(BASE_DIR, "data", "trade_snapshots.jsonl")
                os.makedirs(os.path.dirname(snap_path), exist_ok=True)
                with open(snap_path, "a") as f:
                    f.write(json.dumps(snap, ensure_ascii=False) + "\n")
            except Exception:
                pass

        # 인메모리 로그 (최대 _TRADE_LOG_MAX 건)
        _trade_log.append(result)
        if len(_trade_log) > _TRADE_LOG_MAX:
            _trade_log = _trade_log[-_TRADE_LOG_MAX:]

        import logging as _lg
        _lg.getLogger("auto-trader").info(
            "[AutoTrade] action=%s reason=%s", result.get("action"), result.get("reason")
        )

        # ── 놓친 진입 메모리 기록 ──────────────────────────
        # R:R 거부 / SL 미설정 / 확신도 미달 → analyst 메모리에 missed_entry 로 기록
        # 리플렉션이 나중에 "왜 진입 못 했고 이후 가격이 어떻게 됐는지" 평가
        _reason = result.get("reason", "")
        _missed_keywords = ("R:R", "SL 미설정", "확신도", "중복 방지")
        if any(kw in _reason for kw in _missed_keywords):
            try:
                from agents.memory import get_memory as _gm
                _missed_mem = _gm("analyst")
                _entry_price = (trade_levels or {}).get("entry") or price
                _tp = result.get("tp")
                _sl = result.get("sl")
                _rr = result.get("rr")
                _missed_situation = (
                    f"[놓친 진입 — {_reason}]\n"
                    f"신호: {signal} | 확신도: {confidence}% | 현재가: ${price:,.2f}\n"
                    f"진입가: ${_entry_price:,.2f} | "
                    f"TP: {'$'+f'{_tp:,.2f}' if _tp else 'N/A'} | "
                    f"SL: {'$'+f'{_sl:,.2f}' if _sl else 'N/A'} | "
                    f"R:R: {_rr if _rr else 'N/A'}\n"
                    f"분석요약: {str(analysis.get('summary',''))[:200]}"
                )
                _missed_mem.add_situation(
                    situation=_missed_situation,
                    advice=f"진입 거부됨 — {_reason}",
                    outcome="",
                    meta={
                        "price_at_analysis": float(price),
                        "trade_levels": trade_levels,
                        "missed": True,
                        "signal": signal,
                        "confidence": confidence,
                        "rr": _rr,
                    }
                )
                _lg.getLogger("auto-trader").info("[AutoTrade] 놓친 진입 메모리 기록 완료")
            except Exception as _me:
                _lg.getLogger("auto-trader").warning("[AutoTrade] 놓친 진입 메모리 기록 실패 — %s", _me)


async def _migrate_history_to_memory():
    """과거 분석 히스토리를 reflection 메모리에 임포트 (서버 시작 후 1회)."""
    if _get_memory is None:
        return
    try:
        import json as _json
        _hist_path = os.path.join(BASE_DIR, "data", "analysis_history.jsonl")
        if not os.path.exists(_hist_path):
            return
        _mem = _get_memory("analyst")
        _existing = {r.timestamp for r in _mem.list_records()}
        _added = 0
        with open(_hist_path, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    _e = _json.loads(_line)
                except Exception:
                    continue
                _signal = _e.get("signal", "")
                _price  = _e.get("price")
                _ts     = _e.get("timestamp", "")
                if _signal in ("홀드", "HOLD", "hold"):
                    continue
                if not isinstance(_price, (int, float)) or _price <= 0:
                    continue
                if _ts in _existing:
                    continue
                _tl   = _e.get("trade_levels") or {}
                _conf = _e.get("confidence", 0)
                _situation = (
                    f"[과거 분석 — {_ts}]\n"
                    f"신호: {_signal} | 확신도: {_conf}% | 가격: ${_price:,.2f}\n"
                    f"레짐: {_e.get('regime', '')}\n"
                    f"요약: {str(_e.get('summary', ''))[:300]}"
                )
                _advice = f"{_signal} 신호 (확신도 {_conf}%)"
                if _tl.get("entry"):  _advice += f" | 진입가 ${_tl['entry']:,.2f}"
                if _tl.get("target"): _advice += f" | TP ${_tl['target']:,.2f}"
                if _tl.get("stop"):   _advice += f" | SL ${_tl['stop']:,.2f}"
                _rec = _mem.add_situation(
                    situation=_situation,
                    advice=_advice,
                    outcome="",
                    meta={
                        "price_at_analysis": float(_price),
                        "trade_levels": _tl,
                        "signal": _signal,
                        "confidence": _conf,
                        "missed": False,
                        "imported_from_history": True,
                    },
                    dedup_threshold=0.0,
                )
                if _rec:
                    # 원래 분석 타임스탬프로 교체 (4시간 경과 판정을 위해)
                    if _ts:
                        _rec.timestamp = _ts
                        _mem._rewrite_disk()
                    _added += 1
                    _existing.add(_ts)
        if _added > 0:
            print(f"[migration] 과거 분석 {_added}건 메모리 임포트 완료", flush=True)
        else:
            print(f"[migration] 새로 임포트할 기록 없음 (전체 {_mem.size()}건)", flush=True)
    except Exception as _me:
        print(f"[migration] 실패 (무시): {_me}", flush=True)


@app.on_event("startup")
async def on_startup():
    telegram_alert.init(os.environ.get("TELEGRAM_BOT_TOKEN",""), os.environ.get("TELEGRAM_CHAT_ID",""))
    _tg("🚀 서버 시작됨 - 텔레그램 연결 OK")
    await _market_stream.start()
    await _account_stream.start()
    await _macro_snapshot.start()
    await _schedule_manager.start()
    # asyncio.create_task(_reflection_loop(), name="reflection-loop")  # 매매 체결 시에만 실행

    # ── 과거 분석 히스토리 → 메모리 마이그레이션 ──────────────
    # server 싱글턴 초기화 후 실행해야 같은 인스턴스를 공유함
    asyncio.create_task(_migrate_history_to_memory())


@app.on_event("shutdown")
async def on_shutdown():
    await _market_stream.stop()
    await _account_stream.stop()
    await _macro_snapshot.stop()
    await _analysis_manager.stop()
    await _schedule_manager.stop()


@app.get("/api/market-stream")
async def market_stream():
    async def generate():
        queue = await _market_stream.subscribe()
        try:
            # ready 대기 중에도 keepalive를 5초마다 보내
            # → 브라우저가 연결이 끊겼다고 오해해서 재연결을 반복하는 것을 방지
            while not _market_stream.is_ready():
                try:
                    await asyncio.wait_for(
                        asyncio.shield(_market_stream._ready.wait()), timeout=5
                    )
                except asyncio.TimeoutError:
                    yield ": waiting\n\n"

            snapshot = await _market_stream.get_full_snapshot()
            yield f"data: {json.dumps({'type':'snapshot','data':snapshot})}\n\n"

            while True:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=15)
                    yield f"data: {json.dumps(message)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            await _market_stream.unsubscribe(queue)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/account-stream")
async def account_stream():
    async def generate():
        queue = await _account_stream.subscribe()
        try:
            await _account_stream.wait_until_ready()
            snapshot = await _account_stream.get_snapshot()
            yield f"data: {json.dumps({'type':'snapshot','data':snapshot})}\n\n"

            while True:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=20)
                    yield f"data: {json.dumps(message)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            await _account_stream.unsubscribe(queue)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/connections")
async def connections():
    """현재 market-stream 에 연결된 SSE 클라이언트 수 반환."""
    async with _market_stream._lock:
        count = len(_market_stream._listeners)
    return {"count": count}


@app.get("/api/debug/price")
async def debug_price():
    """WebSocket 연결 상태 및 현재 가격 진단용 엔드포인트."""
    import datetime as _dt
    async with _market_stream._lock:
        price = _market_stream._price
        last_update = _market_stream._last_update
        listeners = len(_market_stream._listeners)
        ready = _market_stream._ready.is_set()
        runner_alive = (
            _market_stream._runner_task is not None
            and not _market_stream._runner_task.done()
        )
        tick_alive = (
            _market_stream._price_tick_task is not None
            and not _market_stream._price_tick_task.done()
        )
    return {
        "price": price,
        "last_update": last_update,
        "server_time": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stream_ready": ready,
        "runner_task_alive": runner_alive,
        "price_tick_task_alive": tick_alive,
        "sse_listeners": listeners,
        # aggTrade/kline 수신 카운터 — 0이면 WebSocket이 데이터를 못 받는 것
        "trade_events_received": _market_stream._trade_count,
        "kline_events_received": _market_stream._kline_count,
        "ws_reconnect_count": _market_stream._ws_connect_count,
        "ws_source": "bybit",  # fstream.binance.com → stream.bybit.com 으로 교체
    }


@app.get("/api/schedule")
async def schedule_get():
    """현재 자동분석 스케줄 상태 반환."""
    return _schedule_manager.status()


class ScheduleSetRequest(BaseModel):
    enabled: bool


@app.post("/api/schedule")
async def schedule_set(body: ScheduleSetRequest):
    """자동분석 스케줄 설정. enabled=true/false 로 30분 타이머를 제어."""
    await _schedule_manager.set_schedule(body.enabled)
    return _schedule_manager.status()


class AnalyzeRequest(BaseModel):
    # 주인장 쿨다운 우회용 비밀번호 — 선택적. body 로만 받아 URL/액세스로그 노출 방지.
    password: str | None = None


@app.post("/api/analyze")
async def analyze_start(
    body: AnalyzeRequest | None = None,
    password: str | None = None,  # 하위 호환: 쿼리스트링으로도 일시적으로 허용
):
    # body 우선, 없으면 쿼리스트링 (레거시) — 둘 다 비어있으면 일반 분석
    supplied = (body.password if body else None) or password
    bypass = False
    if supplied:
        if not runtime_config.verify_owner_password(supplied):
            raise HTTPException(status_code=403, detail="비밀번호가 틀렸습니다.")
        bypass = True
    job, started = await _analysis_manager.start_job(bypass_cooldown=bypass)
    return {"job": job, "started": started}


@app.get("/api/analyze")
async def analyze_status(include_latest: bool = False):
    return await _analysis_manager.get_status(include_latest=include_latest)



@app.post("/api/reflect")
async def reflect_endpoint():
    """
    메모리에 누적된 과거 기록들 중 outcome 이 비어 있는 것들을
    백그라운드에서 비동기 처리 — 타임아웃 방지.
    """
    result = await _run_reflect_background()
    return result if isinstance(result, dict) else {"ok": True, "processed": 0}


async def _run_reflect_background():
    """리플렉션 백그라운드 실행 — 타임아웃 없이 처리."""
    if not _REFLECTION_ENABLED:
        print("[reflect-bg] REFLECTION_ENABLED=False — 건너뜀", flush=True)
        return
    import datetime as _dt
    loop = asyncio.get_event_loop()

    # 현재가 수집 (단일 호출) — 분석 executor 와 분리하여 경쟁 방지
    try:
        price_now = await asyncio.to_thread(fetch_current_price, DEFAULT_SYMBOL)
    except Exception as exc:
        return {"ok": False, "error": f"가격 수집 실패 — {exc}"}

    # 최근 청산 스냅샷에서 실제 청산가 가져오기
    try:
        snap_path = os.path.join(BASE_DIR, "data", "trade_snapshots.jsonl")
        if os.path.exists(snap_path):
            with open(snap_path) as f:
                lines = f.readlines()
            for line in reversed(lines):
                snap = json.loads(line)
                if snap.get("closed") and snap.get("close_equity"):
                    # 실제 청산된 스냅샷이 있으면 현재가 대신 사용
                    break
    except Exception:
        pass

    now_utc = _dt.datetime.now(_dt.timezone.utc)
    all_results = []
    total_skipped = 0

    # ── Analyst 메모리 리플렉션 ─────────────────────────
    analyst_memory = _get_memory("analyst")
    pending = analyst_memory.list_pending_reflections(min_age_seconds=7200.0, limit=5)
    print(f"[reflect-bg] analyst 전체={analyst_memory.size()}건 / analyst pending={len(pending)}건", flush=True)

    for rec in pending:
        try:
            ts = _dt.datetime.fromisoformat(rec.timestamp.replace("Z", "+00:00"))
        except Exception:
            continue
        elapsed = (now_utc - ts).total_seconds()

        meta = rec.meta or {}
        price_then = meta.get("price_at_analysis")
        if not isinstance(price_then, (int, float)) or price_then <= 0:
            trade_levels = meta.get("trade_levels") or {}
            entry = trade_levels.get("entry")
            if isinstance(entry, (int, float)) and entry > 0:
                price_then = float(entry)
            else:
                total_skipped += 1
                continue
        price_then = float(price_then)

        # 분석 이후 최고가/최저가 조회 (TP/SL 정확한 도달 여부 판단용)
        try:
            from data_fetcher import fetch_high_low_since as _fhls
            _hl = await asyncio.to_thread(_fhls, DEFAULT_SYMBOL, rec.timestamp)
            _price_high = _hl.get("high")
            _price_low  = _hl.get("low")
            _price_now_rec = _hl.get("current") or price_now
        except Exception:
            _price_high = None
            _price_low  = None
            _price_now_rec = price_now

        res = await loop.run_in_executor(
            _executor,
            lambda r=rec, pt=price_then, el=elapsed, pn=_price_now_rec, ph=_price_high, pl=_price_low: _reflect_for_role(
                role="analyst",
                record_ts=r.timestamp,
                situation=r.situation,
                advice=r.advice,
                price_then=pt,
                price_now=pn,
                elapsed_seconds=el,
                memory=analyst_memory,
                price_high=ph,
                price_low=pl,
            ),
        )
        all_results.append(res.to_dict())

    # ── 에이전트 역할 메모리 리플렉션 ─────────────────────
    if _get_agent_memories is not None and _reflect_for_role is not None:
        try:
            agent_mems = _get_agent_memories()
            for role in _AGENT_ROLES_FOR_REFLECT:
                role_mem = agent_mems.get(role)
                role_pending = role_mem.list_pending_reflections(min_age_seconds=7200.0, limit=3)
                for rec in role_pending:
                    try:
                        ts = _dt.datetime.fromisoformat(rec.timestamp.replace("Z", "+00:00"))
                    except Exception:
                        continue
                    elapsed = (now_utc - ts).total_seconds()
                    meta = rec.meta or {}
                    price_then = meta.get("price_at_analysis")
                    if not isinstance(price_then, (int, float)) or price_then <= 0:
                        total_skipped += 1
                        continue
                    price_then = float(price_then)

                    # 버그2 수정: 역할 리플렉션에도 구간 고가/저가 조회 (TP/SL 평가용)
                    try:
                        from data_fetcher import fetch_high_low_since as _fhls
                        _hl2 = await asyncio.to_thread(_fhls, DEFAULT_SYMBOL, rec.timestamp)
                        _r_price_high = _hl2.get("high")
                        _r_price_low  = _hl2.get("low")
                        _r_price_now  = _hl2.get("current") or price_now
                    except Exception:
                        _r_price_high = None
                        _r_price_low  = None
                        _r_price_now  = price_now

                    res = await loop.run_in_executor(
                        _executor,
                        lambda r=rec, pt=price_then, el=elapsed, rl=role, rm=role_mem,
                               pn=_r_price_now, ph=_r_price_high, pl=_r_price_low: _reflect_for_role(
                            role=rl,
                            record_ts=r.timestamp,
                            situation=r.situation,
                            advice=r.advice,
                            price_then=pt,
                            price_now=pn,
                            elapsed_seconds=el,
                            memory=rm,
                            price_high=ph,
                            price_low=pl,
                        ),
                    )
                    all_results.append(res.to_dict())
        except Exception as exc:
            import logging as _logging
            _logging.getLogger(__name__).warning("에이전트 역할 reflection 실패 — %s", exc)

    # 전체 기록 수 (outcome 유무 무관)
    total_records = analyst_memory.size()
    # 아직 리플렉션이 안 된 기록 수 — analyst + 전체 에이전트 역할 합산
    still_pending = len(analyst_memory.list_pending_reflections(min_age_seconds=0, limit=9999))
    if _get_agent_memories is not None:
        try:
            _am = _get_agent_memories()
            for _role in _AGENT_ROLES_FOR_REFLECT:
                still_pending += len(_am.get(_role).list_pending_reflections(min_age_seconds=0, limit=9999))
        except Exception:
            pass

    return {
        "ok": True,
        "price_now": price_now,
        "processed": len(all_results),
        "skipped_no_baseline": total_skipped,
        "memory_size": total_records,          # 프론트엔드 호환 필드
        "analyst_memory_size": total_records,  # 하위 호환
        "pending_count": still_pending,
        "results": all_results,
    }




@app.get("/api/macro")
async def macro_endpoint():
    """
    거시경제 지표 JSON 반환.
    yfinance 실시간 (^TNX/^FVX/DX-Y.NYB/HYG/LQD/IBIT) + DefiLlama + CoinGecko.
    _eth_btc / _trad_markets 는 eth_btc / trad_markets 키로 노출.
    """
    if _macro_snapshot.is_ready():
        data = await _macro_snapshot.get_snapshot()
    else:
        data = await asyncio.to_thread(fetch_macro_context)
    # _eth_btc, _trad_markets 를 공개 키로 포함
    out = {k: v for k, v in data.items() if not str(k).startswith("_")}
    out["eth_btc"]      = data.get("_eth_btc")
    out["trad_markets"] = data.get("_trad_markets")
    return out


# ─── 시장 심리 스냅샷 캐시 (5분 TTL) ─────────────────────────
_sentiment_cache: dict = {"data": None, "ts": 0.0}
_SENTIMENT_TTL = 300  # 5분

@app.get("/api/market-sentiment")
async def market_sentiment_endpoint():
    """
    Fear & Greed, 펀딩비, CVD, 오더북 불균형, Bybit OI 등 실시간 심리 지표.
    5분 캐시 적용.
    """
    import time as _t
    now = _t.time()
    if _sentiment_cache["data"] is None or now - _sentiment_cache["ts"] > _SENTIMENT_TTL:
        from market_context import fetch_market_context
        ctx = await asyncio.to_thread(fetch_market_context)
        _sentiment_cache["data"] = ctx
        _sentiment_cache["ts"]   = now
    return _sentiment_cache["data"]


@app.get("/api/account")
async def account_endpoint():
    print("[ACCT] auto_trader:", _auto_trader is not None, flush=True)
    """계좌 정보 — Bitget, 실패 시 캐시된 마지막 값 반환."""
    global _cached_account
    if _auto_trader is not None:
        try:
            acct = await asyncio.to_thread(_auto_trader.get_account)
            print("[ACCT-VAL]", acct, flush=True)
            positions = await asyncio.to_thread(_auto_trader.get_positions)
            print("[POS]", positions, flush=True)

            equity     = float(acct.get("equity",          0) or 0)
            available  = float(acct.get("available",       0) or 0)
            unrealized = float(acct.get("unrealizedPL",    0) or 0)
            # 포지션에서 미실현 손익 합산
            pos_unrealized = sum(float(p.get("unrealizedPL", 0) or 0) for p in positions)
            if pos_unrealized != 0: unrealized = pos_unrealized
            today_pnl  = float(acct.get("todayProfitLoss", 0) or 0)
            # ccxt 폴백: info 필드
            if equity == 0 and acct.get("info"):
                info = acct["info"]
                equity    = float(info.get("equity",    info.get("usdtEquity",    0)) or 0)
                available = float(info.get("available", info.get("available",     0)) or 0)

            pos_list = []
            for p in positions:
                hold   = p.get("holdSide", "")
                total  = float(p.get("total", 0) or 0)
                entry  = float(p.get("averageOpenPrice", 0) or 0)
                pnl    = float(p.get("unrealizedPL", 0) or 0)
                lev    = p.get("leverage", "–")

                # ROE 직접 계산: pnl / (size * entry / leverage)
                try:
                    margin = (total * entry) / float(lev) if float(str(lev)) > 0 else 0
                    pnl_r = (pnl / margin) if margin > 0 else 0
                except Exception:
                    pnl_r = 0
                pos_list.append({
                    "symbol":        DEFAULT_SYMBOL,
                    "side":          hold,
                    "size":          total,
                    "entry_price":   entry,
                    "entryPrice":    entry,
                    "unrealized_pnl": pnl,
                    "unrealizedPnl": pnl,
                    "roe":           round(pnl_r * 100, 2),
                    "leverage":      lev,
                    "notional":      total * entry,
                })

            result = {
                "source":                "bitget",
                # 프론트엔드 호환 필드명
                "wallet_balance":        equity,
                "available_balance":     available,
                "open_position_upnl":    unrealized,
                "today_total_pnl":       today_pnl,
                "today_cash_pnl":        today_pnl,
                "today_commission_fee":  0,
                "open_position_count":   len(pos_list),
                "open_position_notional":sum(float(p.get("size",0) or 0) * float(p.get("entryPrice", p.get("averageOpenPrice",0)) or 0) for p in pos_list),
                "open_positions":        pos_list,
                "effective_leverage":    pos_list[0].get("leverage") if pos_list else None,
                "updated_at":            _now_iso(),
            }
            # 성공 시 캐시 저장 (0 제외)
            if equity > 0:
                _cached_account = result
                return result
            if _cached_account:
                return _cached_account
            return result
        except Exception as e:
            import logging as _lg
            _lg.getLogger("account").warning("Bitget 계좌 조회 실패: %s", e)
            if _cached_account:
                return _cached_account
            return {"wallet_balance": 0, "available_balance": 0,
                    "open_positions": [], "open_position_count": 0,
                    "today_total_pnl": 0, "open_position_upnl": 0,
                    "updated_at": _now_iso(), "error": str(e)}

    return {"wallet_balance": 0, "available_balance": 0,
            "open_positions": [], "updated_at": _now_iso()}


@app.post("/api/symbol")
async def set_symbol(request: Request):
    body = await request.json()
    symbol = body.get("symbol", "BTCUSDT").upper()
    if symbol not in ("BTCUSDT", "ETHUSDT"):
        raise HTTPException(status_code=400, detail="지원하지 않는 심볼")
    _save_symbol(symbol)
    if _auto_trader:
        _auto_trader.symbol = symbol
    return {"ok": True, "symbol": symbol}

@app.get("/api/symbol")
async def get_symbol():
    return {"symbol": _current_symbol}

@app.get("/api/setup/status")
async def setup_status():
    """
    어떤 API 키가 빠져 있는지 반환.
    값이 채워져 있는지만 확인 — 실제 유효성 검증은 하지 않음.
    """
    import config as _cfg
    return {
        "claude":   bool(_cfg.CLAUDE_API_KEY),
        "binance":  bool(_cfg.BINANCE_API_KEY and _cfg.BINANCE_SECRET_KEY),
    }


class SetupSaveRequest(BaseModel):
    claude_api_key:     str = ""
    binance_api_key:    str = ""
    binance_secret_key: str = ""
    bitget_api_key:     str = ""
    bitget_secret_key:  str = ""
    bitget_passphrase:  str = ""


def _is_local_client(request: Request) -> bool:
    """요청이 로컬 루프백에서 왔는지 판정 — setup/save 같은 초기화 엔드포인트 보호용."""
    host = (request.client.host if request.client else "") or ""
    return host in ("127.0.0.1", "::1", "localhost")


@app.post("/api/setup/save")
async def setup_save(body: SetupSaveRequest, request: Request):
    """
    입력받은 키를 .env 파일에 저장하고 config 모듈을 재로드.

    보안:
      - 로컬 루프백 또는 올바른 OWNER_PASSWORD 가 있을 때만 허용.
        (원격에서 임의 사용자가 API 키를 덮어쓰는 것을 차단)
      - 입력 값은 CRLF 로 .env 파일을 오염시키지 못하도록 sanitize.
    """
    # 원격 호출이면 OWNER_PASSWORD 가 필요 (헤더로만 받음 → 로그 유출 방지)
    if not _is_local_client(request):
        _require_owner(request.headers.get("X-Owner-Password"))

    import config as _cfg
    from pathlib import Path

    env_path = Path(__file__).resolve().parent / ".env"

    # 기존 .env 읽기 (없으면 빈 dict)
    existing: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                existing[k.strip()] = v.strip()

    # 새 값 병합 — CRLF/NUL 제거하여 .env 파일 오염/환경변수 주입 방지
    updates = {
        "CLAUDE_API_KEY":     runtime_config.sanitize_env_value(body.claude_api_key),
        "BINANCE_API_KEY":    runtime_config.sanitize_env_value(body.binance_api_key),
        "BINANCE_SECRET_KEY": runtime_config.sanitize_env_value(body.binance_secret_key),
        "BITGET_API_KEY":     runtime_config.sanitize_env_value(body.bitget_api_key),
        "BITGET_SECRET_KEY":  runtime_config.sanitize_env_value(body.bitget_secret_key),
        "BITGET_PASSPHRASE":  runtime_config.sanitize_env_value(body.bitget_passphrase),
    }
    for k, v in updates.items():
        if v:
            existing[k] = v

    # .env 재작성
    lines = ["# API 키 — 절대 외부에 공유하지 마세요!\n"]
    for k, v in existing.items():
        lines.append(f"{k}={v}\n")
    env_path.write_text("".join(lines), encoding="utf-8")

    # os.environ + config 모듈 인메모리 갱신 (서버 재시작 없이 반영)
    for k, v in existing.items():
        os.environ[k] = v
    import importlib
    importlib.reload(_cfg)

    # 계좌 스트림은 API 키를 사용하므로 저장 직후 재시작해 새 값을 즉시 반영
    await _account_stream.stop()
    await _account_stream.start()

    # Bitget 키가 갱신되었을 수 있으므로 _auto_trader 재생성
    # (기존엔 시작 시 한 번만 읽어서 키 변경이 영원히 반영 안 되는 버그 있었음)
    global _auto_trader
    try:
        from config import (
            BITGET_API_KEY as _new_bk,
            BITGET_SECRET_KEY as _new_bs,
            BITGET_PASSPHRASE as _new_bp,
        )
        if _new_bk and _new_bs and _new_bp:
            _auto_trader = _make_auto_trader()
            _load_at_settings()  # 저장된 사용자 설정 다시 적용
            print("[setup] Bitget AutoTrader 재생성 완료", flush=True)
    except Exception as _exc:
        print(f"[setup] AutoTrader 재생성 실패: {_exc}", flush=True)

    return {"ok": True}


@app.get("/api/analysis-history")
async def analysis_history_endpoint(limit: int = 100):
    """저장된 분석 히스토리 반환 (최신순)."""
    # 방어적 경계 — 음수/비현실적 값은 상식적 범위로 clamp.
    limit = max(1, min(limit, ANALYSIS_HISTORY_MAX))
    from collections import deque
    entries: list = []
    try:
        if os.path.exists(ANALYSIS_HISTORY_PATH):
            # 파일 전체를 메모리에 올리지 않고 마지막 N 줄만 유지.
            with open(ANALYSIS_HISTORY_PATH, "r", encoding="utf-8") as f:
                tail = deque(f, maxlen=limit)
            for line in tail:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        pass
    except Exception as exc:
        print(f"[analysis-history] read failed: {exc}")
    return {"entries": list(reversed(entries))}


@app.get("/api/performance")
async def performance_endpoint(days: int = 30):
    """비트겟 거래 내역 기반 성과 지표 반환."""
    if _auto_trader is None:
        return {"error": "자동매매 미설정", "daily": [], "snapshots": []}
    days = max(1, min(days, 365))
    try:
        trades = await asyncio.to_thread(_auto_trader.client.get_trade_history, "BTCUSDT", days)
    except Exception as e:
        return {"error": str(e), "daily": [], "snapshots": []}

    from collections import defaultdict
    import datetime as _dt
    from time_utils import format_kst

    daily_map = defaultdict(lambda: {"pnl": 0.0, "trades": 0, "wins": 0})
    for t in trades:
        try:
            ts = _dt.datetime.fromtimestamp(int(t.get("timestamp", 0)) / 1000, tz=_dt.timezone.utc)
            day_key = format_kst(ts, "%Y-%m-%d")
            profit = float(t.get("info", {}).get("profit", 0) or 0)
            daily_map[day_key]["pnl"] += profit
            daily_map[day_key]["trades"] += 1
            if profit > 0:
                daily_map[day_key]["wins"] += 1
        except Exception:
            pass

    daily = []
    for day_key in sorted(daily_map.keys()):
        d = daily_map[day_key]
        daily.append({
            "date": day_key,
            "pnl": round(d["pnl"], 4),
            "trades": d["trades"],
            "wins": d["wins"],
        })

    total_pnl = sum(d["pnl"] for d in daily)
    total_trades = sum(d["trades"] for d in daily)
    total_wins = sum(d["wins"] for d in daily)

    return {
        "daily": daily,
        "snapshots": [],
        "total_pnl": round(total_pnl, 4),
        "total_trades": total_trades,
        "win_rate": round(total_wins / total_trades * 100, 1) if total_trades > 0 else 0,
    }
    from datetime import datetime, timezone, timedelta
    # 방어적 경계 — 최대 1년, 최소 1일.
    days = max(1, min(days, 365))
    history_path = os.path.join(BASE_DIR, "data", "account_history.jsonl")
    # cutoff 를 먼저 계산하여 파싱 단계에서 바로 필터링 → 메모리 사용량 절감
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()
    snapshots: list = []
    try:
        if os.path.exists(history_path):
            with open(history_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue
                    if (entry.get("observed_ts") or 0) >= cutoff:
                        snapshots.append(entry)
    except Exception as exc:
        return {"error": str(exc), "snapshots": [], "daily": []}

    # 일별 집계 (KST 기준 날짜로 그룹)
    from collections import defaultdict
    from time_utils import format_kst
    daily_map: dict = defaultdict(list)
    for snap in snapshots:
        try:
            dt = datetime.fromtimestamp(snap["observed_ts"], tz=timezone.utc)
            day_key = format_kst(dt, "%Y-%m-%d")
            daily_map[day_key].append(snap)
        except Exception:
            pass

    daily = []
    for day_key in sorted(daily_map.keys()):
        day_snaps = daily_map[day_key]
        # 당일 마지막 스냅샷 기준
        last = day_snaps[-1]
        first = day_snaps[0]
        pnl_vals = [s.get("today_total_pnl") for s in day_snaps if s.get("today_total_pnl") is not None]
        eq_vals  = [s.get("account_equity")  for s in day_snaps if s.get("account_equity")  is not None]
        daily.append({
            "date":           day_key,
            "pnl":            last.get("today_total_pnl"),
            "pnl_pct":        last.get("today_pnl_pct"),
            "equity_start":   first.get("account_equity"),
            "equity_end":     last.get("account_equity"),
            "equity_high":    max(eq_vals)  if eq_vals  else None,
            "equity_low":     min(eq_vals)  if eq_vals  else None,
            "pnl_high":       max(pnl_vals) if pnl_vals else None,
            "pnl_low":        min(pnl_vals) if pnl_vals else None,
            "risk_status":    last.get("risk_status"),
            "snap_count":     len(day_snaps),
        })

    return {"snapshots": snapshots, "daily": daily}




# ══════════════════════════════════════════════
# Bitget 자동매매 API
# ══════════════════════════════════════════════

class AutoTradeSettingsRequest(BaseModel):
    enabled:        bool  | None = None
    usdt_per_trade: float | None = None
    leverage:       int   | None = None
    min_confidence: int   | None = None
    use_tp:         bool  | None = None
    use_sl:         bool  | None = None


@app.get("/api/auto-trade/status")
async def auto_trade_status():
    """자동매매 현재 상태 + 마지막 거래 결과 + Bitget 포지션 조회."""
    positions = []
    account   = {}
    if _auto_trader is not None:
        try:
            positions = await asyncio.to_thread(_auto_trader.get_positions)
            print("[POS]", positions, flush=True)
        except Exception as e:
            positions = [{"error": str(e)}]
        try:
            account = await asyncio.to_thread(_auto_trader.get_account)
        except Exception as e:
            account = {"error": str(e)}

    cfg = _auto_trader
    return {
        "available":      _BITGET_AVAILABLE,
        "keys_set":       bool(BITGET_API_KEY and BITGET_SECRET_KEY and BITGET_PASSPHRASE),
        "enabled":        _auto_trade_enabled,
        "settings": {
            "symbol":         DEFAULT_SYMBOL,
            "usdt_per_trade": cfg.usdt_per_trade if cfg else AUTO_TRADE_USDT,
            "leverage":       cfg.leverage       if cfg else AUTO_TRADE_LEVERAGE,
            "min_confidence": cfg.min_confidence if cfg else AUTO_TRADE_MIN_CONF,
            "use_tp":         cfg.use_tp         if cfg else AUTO_TRADE_USE_TP,
            "use_sl":         cfg.use_sl         if cfg else AUTO_TRADE_USE_SL,
        },
        "last_trade":  _last_trade_result,
        "positions":   positions,
        "account":     account,
    }


@app.post("/api/auto-trade/settings")
async def auto_trade_settings(body: AutoTradeSettingsRequest):
    """자동매매 설정 변경 (런타임 즉시 반영)."""
    global _auto_trade_enabled, _auto_trader
    async with _auto_trade_lock:
        if body.enabled is not None:
            _auto_trade_enabled = body.enabled

        if _auto_trader is not None:
            if body.usdt_per_trade is not None:
                _auto_trader.usdt_per_trade = body.usdt_per_trade
            if body.leverage is not None:
                _auto_trader.leverage = body.leverage
            if body.min_confidence is not None:
                _auto_trader.min_confidence = body.min_confidence
            if body.use_tp is not None:
                _auto_trader.use_tp = body.use_tp
            if body.use_sl is not None:
                _auto_trader.use_sl = body.use_sl


    _save_at_settings()
    return {"ok": True, "enabled": _auto_trade_enabled}


@app.post("/api/auto-trade/close")
async def auto_trade_close():
    """수동 전체 포지션 청산 (TP/SL 포함)."""
    if _auto_trader is None:
        raise HTTPException(status_code=503, detail="Bitget 자동매매 모듈이 초기화되지 않았습니다.")
    async with _auto_trade_lock:
        results = await asyncio.to_thread(_auto_trader.close_all)
    return {"ok": True, "closed": results}


@app.get("/api/auto-trade/log")
async def auto_trade_log(limit: int = 50):
    """인메모리 자동매매 실행 로그 반환 (최신순)."""
    limit = max(1, min(limit, _TRADE_LOG_MAX))
    return {"log": list(reversed(_trade_log[-limit:]))}


@app.post("/api/admin/reset-migration")
async def reset_migration():
    """마이그레이션 플래그 초기화 — 다음 재시작 시 재실행."""
    try:
        _mem_dir = Path(os.environ.get("MEMORY_DIR", os.path.join(BASE_DIR, "data", "memory")))
        _flag = _mem_dir / ".history_imported"
        if _flag.exists():
            _flag.unlink()
            return {"ok": True, "message": "플래그 삭제 완료 — 서버 재시작하면 마이그레이션 재실행됩니다"}
        return {"ok": True, "message": "플래그 파일 없음 — 이미 초기화 상태"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/api/memory")
async def memory_endpoint(top_k: int = 3):
    """analyst 메모리 최신 기록 반환 — 리플렉션 완료 후 UI 갱신용."""
    if _get_memory is None:
        return {"ok": False, "memories": []}
    try:
        mem = _get_memory("analyst")
        all_recs = mem.list_records()
        recent = list(reversed(all_recs))[:top_k]
        records = [{"record": {
            "timestamp": r.timestamp,
            "situation": r.situation,
            "advice": r.advice,
            "outcome": r.outcome,
            "meta": r.meta,
        }, "score": 0} for r in recent]
        return {"ok": True, "memories": records}
    except Exception as e:
        return {"ok": False, "memories": [], "error": str(e)}

@app.get("/")
async def root():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path, encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/guide", include_in_schema=False)
async def guide():
    path = os.path.join(BASE_DIR, "static", "guide.html")
    with open(path, encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/favicon.ico", include_in_schema=False)
async def favicon_ico():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(BASE_DIR, "static", "favicon.ico"), media_type="image/x-icon")

@app.get("/favicon.svg", include_in_schema=False)
async def favicon_svg():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(BASE_DIR, "static", "favicon.svg"), media_type="image/svg+xml")

@app.get("/favicon-32x32.png", include_in_schema=False)
async def favicon_32():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(BASE_DIR, "static", "favicon-32x32.png"), media_type="image/png")

@app.get("/favicon-16x16.png", include_in_schema=False)
async def favicon_16():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(BASE_DIR, "static", "favicon-16x16.png"), media_type="image/png")

@app.get("/apple-touch-icon.png", include_in_schema=False)
async def apple_touch_icon():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(BASE_DIR, "static", "apple-touch-icon.png"), media_type="image/png")

@app.get("/og-image.png", include_in_schema=False)
async def og_image():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(BASE_DIR, "static", "og-image.png"), media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)