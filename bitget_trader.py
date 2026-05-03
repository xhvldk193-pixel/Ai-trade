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
            "password":   passphrase,
            "options": {
                "defaultType": "swap",
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
                    today_pnl = 0.0
                    try:
                        import datetime as _dt
                        # 오늘 00:00 UTC 타임스탬프 (밀리초)
                        now = _dt.datetime.utcnow()
                        start_ts = int(_dt.datetime(now.year, now.month, now.day).timestamp() * 1000)
                        pnl_data = self._ex.fetch_my_trades(
                            f"BTC/USDT:USDT",
                            params={"productType": "USDT-FUTURES", "startTime": str(start_ts)}
                        )
                        today_pnl = sum(float(t.get("info", {}).get("profit", 0) or 0) for t in pnl_data)
                    except Exception as pnl_err:
                        print(f"[PNL-ERR] {pnl_err}", flush=True)
                    return {"equity": total, "available": free, "unrealizedPL": 0.0, "todayProfitLoss": today_pnl}
            except Exception as e:
                if attempt == 2: raise
                import time as _t; _t.sleep(1)
        return {"equity": 0, "available": 0, "unrealizedPL": 0.0, "todayProfitLoss": 0}

    def get_trade_history(self, symbol: str = "BTCUSDT", days: int = 30) -> list:
        """비트겟 거래 내역 조회 (days일치)."""
        import datetime as _dt
        now = _dt.datetime.utcnow()
        start_ts = int((now - _dt.timedelta(days=days)).timestamp() * 1000)
        try:
            trades = self._ex.fetch_my_trades(
                "BTC/USDT:USDT",
                params={"productType": "USDT-FUTURES", "startTime": str(start_ts)}
            )
            return trades
        except Exception as e:
            print(f"[TRADE-HISTORY-ERR] {e}", flush=True)
            return []

    def get_positions(self, symbol: str = "BTCUSDT") -> list[dict]:
        ccxt_symbol = f"{symbol[:3]}/{symbol[3:]}:{symbol[3:]}"
        positions = self._ex.fetch_positions([ccxt_symbol])
        result = []
        for p in positions:
            contracts = float(p.get("contracts") or 0)
            if contracts <= 0:
                continue

            side = p.get("side", "") or p.get("info", {}).get("holdSide", "")
            print(f"[RAW] side={side} unrealizedPnl={p.get('unrealizedPnl')} info_pnl={p.get('info',{}).get('unrealizedPL')}", flush=True)
            entry = float(p.get("entryPrice") or 0)
            mark  = float(p.get("markPrice") or 0)

            # ccxt unrealizedPnl 그대로 사용
            unrealized_pnl = float(p.get("unrealizedPnl") or 0)

            pct = p.get("percentage")
            unrealized_pnl_r = float(pct) / 100 if pct is not None else 0.0
            # 숏 percentage도 부호 보정
            if side == "short" and unrealized_pnl_r > 0 and unrealized_pnl < 0:
                unrealized_pnl_r = -unrealized_pnl_r

            result.append({
                "holdSide":         side,
                "total":            contracts,
                "averageOpenPrice": entry,
                "markPrice":        mark,
                "unrealizedPL":     unrealized_pnl,
                "unrealizedPLR":    unrealized_pnl_r,
                "leverage":         p.get("leverage", 1),
            })
        return result

    def set_leverage(self, symbol: str, leverage: int, hold_side: str = "long") -> dict:
        ccxt_symbol = f"{symbol[:3]}/{symbol[3:]}:{symbol[3:]}"
        try:
            return self._ex.set_leverage(leverage, ccxt_symbol, {
                "holdSide":    hold_side,
                "productType": "USDT-FUTURES",
            })
        except Exception as e:
            log.warning("[Bitget] 레버리지 설정 실패: %s", e)
            return {}

    def set_margin_mode(self, symbol: str, hold_side: str = "long") -> dict:
        """Isolated(독립) 마진 모드 설정 - Futures 전용"""
        ccxt_symbol = f"{symbol[:3]}/{symbol[3:]}:{symbol[3:]}"
        try:
            return self._ex.set_margin_mode(
                "isolated",
                ccxt_symbol,
                {
                    "productType": "USDT-FUTURES",
                    "holdSide":    hold_side,
                }
            )
        except Exception as e:
            log.warning("[Bitget] 마진모드 설정 실패(무시): %s", e)
            return {}

    def place_order(self, symbol: str, side: str, size: float,
                    order_type: str = "market", price: float = None,
                    tp: float = None, sl: float = None) -> dict:
        """
        side: "open_long" | "open_short" | "close_long" | "close_short"
        """
        action_map = {
            "open_long":   ("buy",  "long"),
            "open_short":  ("sell", "short"),
            "close_long":  ("sell", "long"),
            "close_short": ("buy",  "short"),
        }
        trade_side, hold_side = action_map.get(side, ("buy", "long"))
        body = {
            "symbol":      symbol if symbol.endswith("USDT") else f"{symbol}USDT",
            "productType": "USDT-FUTURES",
            "marginMode":  "isolated",
            "marginCoin":  "USDT",
            "size":        str(size),
            "side":        trade_side,
            "orderType":   order_type,
            "force":       "gtc",
        }
        if order_type == "limit" and price:
            body["price"] = str(price)
        if tp:
            body["presetStopSurplusPrice"] = str(tp)
        if sl:
            body["presetStopLossPrice"] = str(sl)
        return self._rest_post("/api/v2/mix/order/place-order", body)

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
        api_key    = self._ex.apiKey
        secret     = self._ex.secret
        passphrase = self._ex.password
        ts = str(int(_t.time() * 1000))
        body_str = _json.dumps(body)
        pre  = ts + "POST" + path + body_str
        sign = base64.b64encode(hmac.new(secret.encode(), pre.encode(), hashlib.sha256).digest()).decode()
        headers = {
            "ACCESS-KEY":        api_key,
            "ACCESS-SIGN":       sign,
            "ACCESS-TIMESTAMP":  ts,
            "ACCESS-PASSPHRASE": passphrase,
            "Content-Type":      "application/json",
        }
        print(f"[REST] {path} {body_str}", flush=True)
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
        try:
            return self._rest_post("/api/v2/mix/order/place-tpsl-order", {
                "symbol":       symbol if symbol.endswith("USDT") else f"{symbol}USDT",
                "productType":  "USDT-FUTURES",
                "marginCoin":   "USDT",
                "planType":     "profit_plan",
                "triggerPrice": str(trigger_price),
                "triggerType":  "mark_price",
                "size":         str(size),
            })
        except Exception as e:
            self._tg_alert(f"⚠️ TP 등록 실패\n{str(e)[:200]}")
            raise

    def set_sl(self, symbol: str, trigger_price: float,
               hold_side: str, size: float) -> dict:
        """손절(SL) 주문 등록."""
        try:
            return self._rest_post("/api/v2/mix/order/place-tpsl-order", {
                "symbol":       symbol if symbol.endswith("USDT") else f"{symbol}USDT",
                "productType":  "USDT-FUTURES",
                "marginCoin":   "USDT",
                "planType":     "loss_plan",
                "triggerPrice": str(trigger_price),
                "triggerType":  "mark_price",
                "size":         str(size),
            })
        except Exception as e:
            self._tg_alert(f"⚠️ SL 등록 실패\n{str(e)[:200]}")
            raise

    def cancel_all_tpsl(self, symbol: str) -> dict:
        """TP/SL 플랜 주문 전체 취소."""
        try:
            ccxt_symbol = f"{symbol[:3]}/{symbol[3:]}:{symbol[3:]}"
            return self._ex.cancel_all_orders(ccxt_symbol, params={
                "planType":    "profit_loss",
                "productType": "USDT-FUTURES",
            })
        except Exception as e:
            log.warning("[Bitget] TP/SL 취소 실패(무시): %s", e)
            return {}

    def _tg_alert(self, msg: str):
        import os, requests
        t = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        c = os.environ.get("TELEGRAM_CHAT_ID", "")
        if t and c:
            try:
                requests.post(
                    f"https://api.telegram.org/bot{t}/sendMessage",
                    json={"chat_id": c, "text": msg},
                    timeout=5
                )
            except:
                pass


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
    # 최소 손익비 — 이 값 미만이면 진입 거부
    # R:R 1.5 = 이길 때 버는 돈이 질 때 잃는 돈의 1.5배 이상
    MIN_RR: float = float(os.environ.get("AUTO_TRADE_MIN_RR", "1.5"))

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
        # usdt_per_trade가 1~100 사이면 잔고 비율(%)로 처리
        if 1 <= self.usdt_per_trade <= 100:
            try:
                acct = self.get_account()
                equity = float(acct.get("equity", 0) or 0)
                usdt = equity * (self.usdt_per_trade / 100) * 0.95  # 5% 수수료 여유
            except:
                usdt = 100  # 폴백
        else:
            usdt = self.usdt_per_trade
        return round((usdt * self.leverage) / price, 4)

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

        # 독립(Isolated) 마진 + 레버리지 설정
        for hs in ("long", "short"):
            self.client.set_margin_mode(self.symbol, hs)
            self.client.set_leverage(self.symbol, self.leverage, hs)

        tp, sl = self._extract_tpsl(desired, price, trade_levels)
        result["tp"] = tp
        result["sl"] = sl

        # ── SL 필수 체크 ──────────────────────────────
        # SL 없이 진입하면 레버리지 × 전체배분 규모가 무한손실 가능
        if sl is None:
            result["reason"] = "SL 미설정 → 진입 거부 (손실 무한 리스크)"
            self._last = result
            log.warning("[AutoTrader] SL 없음 → 진입 취소")
            return result

        # ── 최소 손익비(R:R) 체크 ────────────────────
        if tp is not None:
            rr = abs(tp - price) / abs(sl - price)
            result["rr"] = round(rr, 2)
            if rr < self.MIN_RR:
                result["reason"] = (
                    f"R:R {rr:.2f} < 최소 {self.MIN_RR} → 진입 거부"
                    f" (TP ${tp:,.2f} / SL ${sl:,.2f})"
                )
                self._last = result
                log.warning("[AutoTrader] R:R 불충분(%.2f) → 진입 취소", rr)
                return result
        else:
            # TP 없으면 R:R 계산 불가 → 진입 허용하되 경고 로그
            result["rr"] = None
            log.warning("[AutoTrader] TP 없음 — R:R 미검증 진입 (SL만 설정)")

        size = self._contracts(price)
        order_side = "open_long" if desired == "long" else "open_short"

        # 진입가 기반 시장가/지정가 결정
        entry_price = trade_levels.get("entry") if trade_levels else None
        use_limit = False
        if entry_price and price:
            if desired == "short" and price < entry_price:
                # 숏: 현재가가 진입가보다 낮으면 지정가
                use_limit = True
            elif desired == "long" and price > entry_price:
                # 롱: 현재가가 진입가보다 높으면 지정가
                use_limit = True

        if use_limit and entry_price:
            order_resp = self.client.place_order(self.symbol, order_side, size, order_type="limit", price=entry_price, tp=tp, sl=sl)
            result["reason"] = f"지정가 진입 @ ${entry_price:,.2f}"
        else:
            order_resp = self.client.place_order(self.symbol, order_side, size, tp=tp, sl=sl)
        result["action"] = desired
        result["order"]  = order_resp
        result["reason"] = (
            f"{signal} (확신도 {confidence}%) → {desired} {size}계약 @ ${price:,.2f}"
            + (f" | TP ${tp:,.2f}" if tp else " | TP 없음")
            + (f" | SL ${sl:,.2f}" if sl else " | SL 없음")
        )
        log.info("[AutoTrader] %s", result["reason"])

        # TP/SL은 진입 주문 시 presetStopSurplusPrice/presetStopLossPrice로 같이 등록됨
        result["tp_order"] = {"included_in_order": True} if tp else None
        result["sl_order"] = {"included_in_order": True} if sl else None

        self._last = result
        return result

    def _tg_alert(self, msg):
        import os, requests
        t = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        c = os.environ.get("TELEGRAM_CHAT_ID", "")
        if t and c:
            try:
                requests.post(
                    f"https://api.telegram.org/bot{t}/sendMessage",
                    json={"chat_id": c, "text": msg},
                    timeout=5
                )
            except:
                pass

    def last_result(self):   return dict(self._last)
    def get_positions(self): return self.client.get_positions(self.symbol)
    def get_account(self):   return self.client.get_account(self.symbol)
    def close_all(self):
        try: self.client.cancel_all_tpsl(self.symbol)
        except: pass
        return self.client.close_all(self.symbol)
