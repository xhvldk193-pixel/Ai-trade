"""
Bitget USDT-M Futures 자동매매 모듈 (ccxt 기반)
TP/SL: Claude AI 피보나치 기반 목표가/손절가 직접 전달
"""
from __future__ import annotations

import logging
import time
from typing import Optional

log = logging.getLogger(__name__)


# ──────────────────────────────────────────
# ccxt Bitget 클라이언트
# ──────────────────────────────────────────
class BitgetClient:
    def __init__(self, api_key: str, secret_key: str, passphrase: str):
        import ccxt
        self._ex = ccxt.bitget({
            "apiKey":     api_key,
            "secret":     secret_key,
            "password":   passphrase,   # Bitget passphrase
            "options": {
                "defaultType": "swap",  # USDT-M Futures
            },
        })

    def get_account(self, symbol="BTCUSDT"):
        for attempt in range(3):
            try:
                bal = self._ex.fetch_balance({"type": "swap"})
                usdt = bal.get("USDT") or {}
                total = float(usdt.get("total") or 0)
                free = float(usdt.get("free") or 0)
                if total > 0:
                    # 오늘 손익 조회
                    today_pnl = 0.0
                    try:
                        import datetime as _dt
                        today = _dt.datetime.utcnow().strftime("%Y-%m-%d")
                        pnl_data = self._ex.fetch_my_trades(
                            f"{symbol}/USDT:USDT",
                            params={"productType": "USDT-FUTURES", "startTime": today}
                        )
                        today_pnl = sum(float(t.get("info", {}).get("profit", 0) or 0) for t in pnl_data)
                    except:
                        pass
                    return {"equity": total, "available": free, "unrealizedPL": 0.0, "todayProfitLoss": today_pnl}
            except Exception as e:
                if attempt == 2: raise
                import time as _t; _t.sleep(1)
        return {"equity": 0, "available": 0, "unrealizedPL": 0.0, "todayProfitLoss": 0}


    def get_positions(self, symbol: str = "BTCUSDT") -> list[dict]:
        """현재 오픈 포지션 조회."""
        ccxt_symbol = f"{symbol[:3]}/{symbol[3:]}:{symbol[3:]}"  # BTC/USDT:USDT
        positions = self._ex.fetch_positions([ccxt_symbol])
        result = []
        for p in positions:
            contracts = float(p.get("contracts") or 0)
            if contracts <= 0:
                continue
            result.append({
                "holdSide":         p.get("side", ""),
                "total":            contracts,
                "averageOpenPrice": p.get("entryPrice", 0),
                "unrealizedPL":     p.get("unrealizedPnl", 0),
                "unrealizedPLR":    p.get("percentage", 0) / 100 if p.get("percentage") else 0,
                "leverage":         p.get("leverage", 1),
            })
        return result

    def set_leverage(self, symbol: str, leverage: int, hold_side: str = "long") -> dict:
        ccxt_symbol = f"{symbol[:3]}/{symbol[3:]}:{symbol[3:]}"
        try:
            return self._ex.set_leverage(leverage, ccxt_symbol, {"holdSide": hold_side})
        except Exception as e:
            log.warning("[Bitget] 레버리지 설정 실패: %s", e)
            return {}

    def place_order(self, symbol: str, side: str, size: float,
                    order_type: str = "market") -> dict:
        """
        side: "open_long" | "open_short" | "close_long" | "close_short"
        """
        ccxt_symbol = f"{symbol[:3]}/{symbol[3:]}:{symbol[3:]}"
        # ccxt side/positionSide 변환
        action_map = {
            "open_long":   ("buy",  "long"),
            "open_short":  ("sell", "short"),
            "close_long":  ("sell", "long"),
            "close_short": ("buy",  "short"),
        }
        ccxt_side, pos_side = action_map.get(side, ("buy", "long"))
        params = {
            "tdMode":       "cross",
            "posSide":      pos_side,
            "reduceOnly":   side.startswith("close"),
        }
        return self._ex.create_order(
            ccxt_symbol, order_type, ccxt_side, size, params=params
        )

    def close_all(self, symbol: str) -> list[dict]:
        """전체 포지션 시장가 청산."""
        positions = self.get_positions(symbol)
        results = []
        for p in positions:
            hold  = p.get("holdSide", "")
            total = float(p.get("total", 0))
            if total <= 0:
                continue
            side = "close_long" if hold == "long" else "close_short"
            try:
                results.append(self.place_order(symbol, side, total))
            except Exception as e:
                log.warning("[Bitget] 청산 실패: %s", e)
        return results

    def _rest_post(self, path: str, body: dict) -> dict:
        """Bitget REST API 직접 호출 (ccxt 우회)."""
        import hmac, hashlib, base64, time as _t, json as _json, requests as _req
        api_key = self._ex.apiKey
        secret  = self._ex.secret
        passphrase = self._ex.password
        ts = str(int(_t.time() * 1000))
        body_str = _json.dumps(body)
        pre = ts + "POST" + path + body_str
        sign = base64.b64encode(hmac.new(secret.encode(), pre.encode(), hashlib.sha256).digest()).decode()
        headers = {
            "ACCESS-KEY": api_key,
            "ACCESS-SIGN": sign,
            "ACCESS-TIMESTAMP": ts,
            "ACCESS-PASSPHRASE": passphrase,
            "Content-Type": "application/json",
        }
        r = _req.post("https://api.bitget.com" + path, headers=headers, data=body_str, timeout=10)
        d = r.json()
        if d.get("code") not in ("00000", "0"):
            err = f"Bitget: {d.get('msg')} ({d.get('code')})"
            self._tg_alert(f"⚠️ Bitget API 오류\n{err}")
            raise RuntimeError(err)
        return d

    def set_tp(self, symbol: str, trigger_price: float,
               hold_side: str, size: float) -> dict:
        """익절(TP) 주문 등록."""
        return self._rest_post("/api/v2/mix/order/placeTpslOrder", {
            "symbol":       f"{symbol}USDT",
            "productType":  "USDT-FUTURES",
            "marginCoin":   "USDT",
            "planType":     "profit_plan",
            "triggerPrice": str(trigger_price),
            "triggerType":  "fill_price",
            "size":         str(size),
        })

    def set_sl(self, symbol: str, trigger_price: float,
               hold_side: str, size: float) -> dict:
        """손절(SL) 주문 등록."""
        return self._rest_post("/api/v2/mix/order/placeTpslOrder", {
            "symbol":       f"{symbol}USDT",
            "productType":  "USDT-FUTURES",
            "marginCoin":   "USDT",
            "planType":     "loss_plan",
            "triggerPrice": str(trigger_price),
            "triggerType":  "fill_price",
            "size":         str(size),
        })

    def cancel_all_tpsl(self, symbol: str) -> dict:
        """TP/SL 플랜 주문 전체 취소."""
        try:
            ccxt_symbol = f"{symbol[:3]}/{symbol[3:]}:{symbol[3:]}"
            return self._ex.cancel_all_orders(ccxt_symbol, params={"planType": "profit_loss"})
        except Exception as e:
            log.warning("[Bitget] TP/SL 취소 실패(무시): %s", e)
            return {}


# ──────────────────────────────────────────
# TP/SL 유효성 검사
# ──────────────────────────────────────────
def _validate_tpsl(direction, price, tp, sl):
    if direction == "long":
        if tp is not None and tp <= price:
            log.warning("[TPSL] 롱 TP(%.2f) ≤ 현재가(%.2f) → 무시", tp, price); tp = None
        if sl is not None and sl >= price:
            log.warning("[TPSL] 롱 SL(%.2f) ≥ 현재가(%.2f) → 무시", sl, price); sl = None
    else:
        if tp is not None and tp >= price:
            log.warning("[TPSL] 숏 TP(%.2f) ≥ 현재가(%.2f) → 무시", tp, price); tp = None
        if sl is not None and sl <= price:
            log.warning("[TPSL] 숏 SL(%.2f) ≤ 현재가(%.2f) → 무시", sl, price); sl = None
    return tp, sl


# ──────────────────────────────────────────
# AI 신호 → 자동매매 실행기
# ──────────────────────────────────────────
class BitgetAutoTrader:
    def __init__(self, api_key, secret_key, passphrase,
                 symbol="BTCUSDT", usdt_per_trade=20.0, leverage=3,
                 min_confidence=65, use_tp=True, use_sl=True):
        self.client         = BitgetClient(api_key, secret_key, passphrase)
        self.symbol         = symbol.replace("/", "").upper()
        self.usdt_per_trade = usdt_per_trade
        self.leverage       = leverage
        self.min_confidence = min_confidence
        self.use_tp         = use_tp
        self.use_sl         = use_sl
        self._last: dict    = {}

    def _contracts(self, price: float) -> float:
        return round((self.usdt_per_trade * self.leverage) / price, 4)

    def _current_side(self) -> Optional[str]:
        positions = self.client.get_positions(self.symbol)
        if not positions:
            return None
        return positions[0].get("holdSide")

    def _extract_tpsl(self, direction, price, trade_levels):
        tp = sl = None
        if trade_levels:
            raw_tp = trade_levels.get("target")
            raw_sl = trade_levels.get("stop")
            if isinstance(raw_tp, (int, float)) and raw_tp > 0:
                tp = float(raw_tp)
            if isinstance(raw_sl, (int, float)) and raw_sl > 0:
                sl = float(raw_sl)
        return _validate_tpsl(direction, price, tp, sl)

    def execute(self, signal, confidence, price,
                trade_levels=None, tf_data=None) -> dict:
        result = {
            "action": "none", "reason": "", "order": None,
            "tp_order": None, "sl_order": None,
            "tp": None, "sl": None, "rr": None,
            "signal": signal, "confidence": confidence,
        }

        if confidence < self.min_confidence:
            result["reason"] = f"확신도 {confidence}% < {self.min_confidence}% → 패스"
            self._last = result; return result

        if signal == "홀드":
            result["reason"] = "홀드 신호 → 무동작"
            self._last = result; return result

        desired = "long" if signal == "매수" else "short"
        current = self._current_side()

        if current == desired:
            result["reason"] = f"이미 {desired} 포지션 → 중복 방지"
            self._last = result; return result

        if current and current != desired:
            log.info("[AutoTrader] 반대 포지션(%s) 청산", current)
            try: self.client.cancel_all_tpsl(self.symbol)
            except: pass
            self.client.close_all(self.symbol)
            time.sleep(0.8)

        for hs in ("long", "short"):
            self.client.set_leverage(self.symbol, self.leverage, hs)

        tp, sl = self._extract_tpsl(desired, price, trade_levels)
        result["tp"] = tp
        result["sl"] = sl
        if tp and sl:
            result["rr"] = round(abs(tp - price) / abs(sl - price), 2)

        size = self._contracts(price)
        order_side = "open_long" if desired == "long" else "open_short"
        order_resp = self.client.place_order(self.symbol, order_side, size)
        result["action"] = desired
        result["order"]  = order_resp
        result["reason"] = (
            f"{signal} (확신도 {confidence}%) → {desired} {size}계약 @ ${price:,.2f}"
            + (f" | TP ${tp:,.2f}" if tp else " | TP 없음")
            + (f" | SL ${sl:,.2f}" if sl else " | SL 없음")
        )
        log.info("[AutoTrader] %s", result["reason"])

        if self.use_tp and tp:
            try:
                result["tp_order"] = self.client.set_tp(self.symbol, tp, desired, size)
            except Exception as e:
                log.warning("[AutoTrader] TP 등록 실패: %s", e)
                result["tp_order"] = {"error": str(e)}
                self._tg_alert(f"⚠️ TP 등록 실패\n{type(e).__name__}: {str(e)[:200]}")

        if self.use_sl and sl:
            try:
                result["sl_order"] = self.client.set_sl(self.symbol, sl, desired, size)
            except Exception as e:
                log.warning("[AutoTrader] SL 등록 실패: %s", e)
                result["sl_order"] = {"error": str(e)}
                self._tg_alert(f"⚠️ SL 등록 실패\n{type(e).__name__}: {str(e)[:200]}")

        self._last = result
        return result

    def _tg_alert(self, msg):
        import os, requests
        t=os.environ.get("TELEGRAM_BOT_TOKEN","");c=os.environ.get("TELEGRAM_CHAT_ID","")
        if t and c:
            try: requests.post(f"https://api.telegram.org/bot{t}/sendMessage",json={"chat_id":c,"text":msg},timeout=5)
            except: pass

    def last_result(self):  return dict(self._last)
    def get_positions(self): return self.client.get_positions(self.symbol)
    def get_account(self):   return self.client.get_account(self.symbol)
    def close_all(self):
        try: self.client.cancel_all_tpsl(self.symbol)
        except: pass
        return self.client.close_all(self.symbol)
