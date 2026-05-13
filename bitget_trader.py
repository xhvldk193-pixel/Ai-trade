"""
Bitget USDT-M Futures 자동매매 모듈 (ccxt 기반)
TP/SL: Claude AI 피보나치 기반 목표가/손절가 직접 전달
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

import telegram_alert as _tg

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
        """잔고 조회 — GET 요청으로 직접 호출 (40009 서명 오류 방지)."""
        for attempt in range(3):
            try:
                resp = self._rest_get(
                    "/api/v2/mix/account/account",
                    {
                        "symbol":      "BTCUSDT",
                        "productType": "USDT-FUTURES",
                        "marginCoin":  "USDT",
                    }
                )
                data = (resp or {}).get("data") or {}
                equity    = float(data.get("equity",    0) or 0)
                available = float(data.get("available", 0) or 0)
                # equity가 0이면 available로 폴백 (Isolated 모드에서 증거금 잠김 시)
                balance = equity if equity > 0 else available
                if balance > 0:
                    equity = balance  # 이하 로직에서 equity 사용
                    # 오늘 PnL 조회
                    today_pnl = 0.0
                    try:
                        import datetime as _dt
                        now_utc   = _dt.datetime.now(_dt.timezone.utc)
                        start_utc = _dt.datetime(now_utc.year, now_utc.month, now_utc.day, tzinfo=_dt.timezone.utc)
                        start_ts  = int(start_utc.timestamp() * 1000)
                        pnl_resp  = self._rest_get(
                            "/api/v2/mix/order/fill-history",
                            {
                                "symbol":      "BTCUSDT",
                                "productType": "USDT-FUTURES",
                                "startTime":   str(start_ts),
                            }
                        )
                        pnl_list = ((pnl_resp or {}).get("data") or {}).get("fillList") or []
                        today_pnl = sum(float(t.get("profit", 0) or 0) for t in pnl_list)
                    except Exception as pnl_err:
                        log.warning("[PNL-ERR] %s", pnl_err)
                    return {"equity": equity, "available": available, "unrealizedPL": 0.0, "todayProfitLoss": today_pnl}
            except Exception as e:
                log.warning("[get_account] 시도 %d 실패: %s", attempt+1, e)
                if attempt == 2:
                    raise
                import time as _t; _t.sleep(1)
        return {"equity": 0, "available": 0, "unrealizedPL": 0.0, "todayProfitLoss": 0}

    def get_trade_history(self, symbol: str = "BTCUSDT", days: int = 30) -> list:
        """비트겟 거래 내역 조회 (days일치)."""
        import datetime as _dt
        now_utc = _dt.datetime.now(_dt.timezone.utc)
        start_ts = int((now_utc - _dt.timedelta(days=days)).timestamp() * 1000)
        try:
            trades = self._ex.fetch_my_trades(
                "BTC/USDT:USDT",
                params={"productType": "USDT-FUTURES", "startTime": str(start_ts)}
            )
            return trades
        except Exception as e:
            log.warning("[TRADE-HISTORY-ERR] %s", e)
            return []

    def get_positions(self, symbol: str = "BTCUSDT") -> list[dict]:
        """포지션 조회 — CCXT 대신 _rest_get 직접 호출 (40009 서명 오류 방지)."""
        try:
            resp = self._rest_get(
                "/api/v2/mix/position/single-position",
                {
                    "symbol":      symbol if symbol.endswith("USDT") else f"{symbol}USDT",
                    "productType": "USDT-FUTURES",
                    "marginCoin":  "USDT",
                }
            )
            data_list = (resp or {}).get("data") or []
            result = []
            for p in data_list:
                total = float(p.get("total", 0) or 0)
                if total <= 0:
                    continue
                side           = p.get("holdSide", "")
                entry          = float(p.get("averageOpenPrice", 0) or 0)
                mark           = float(p.get("markPrice", 0) or 0)
                unrealized_pnl = float(p.get("unrealizedPL", 0) or 0)
                leverage       = float(p.get("leverage", 10) or 10)
                margin         = float(p.get("margin", 0) or 0)
                unrealized_pnl_r = (unrealized_pnl / margin) if margin > 0 else 0.0

                result.append({
                    "holdSide":         side,
                    "total":            total,
                    "averageOpenPrice": entry,
                    "markPrice":        mark,
                    "unrealizedPL":     unrealized_pnl,
                    "unrealizedPLR":    unrealized_pnl_r,
                    "leverage":         leverage,
                    "margin":           margin,
                })
            return result
        except Exception as e:
            log.warning("[get_positions] 실패: %s", e)
            return []

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

    def _rest_get(self, path: str, params: dict) -> dict:
        """Bitget REST API GET 요청 직접 호출 (쿼리스트링 서명)."""
        import hmac, hashlib, base64, time as _t, requests as _req
        from urllib.parse import urlencode
        api_key    = self._ex.apiKey
        secret     = self._ex.secret
        passphrase = self._ex.password
        ts = str(int(_t.time() * 1000))
        query_str = urlencode(params)
        full_path = path + "?" + query_str  # 서명에 쿼리스트링 포함
        pre  = ts + "GET" + full_path       # body 없음
        sign = base64.b64encode(hmac.new(secret.encode(), pre.encode(), hashlib.sha256).digest()).decode()
        headers = {
            "ACCESS-KEY":        api_key,
            "ACCESS-SIGN":       sign,
            "ACCESS-TIMESTAMP":  ts,
            "ACCESS-PASSPHRASE": passphrase,
            "Content-Type":      "application/json",
        }
        log.debug("[REST-GET] %s", full_path)
        r = _req.get("https://api.bitget.com" + full_path, headers=headers, timeout=10)
        d = r.json()
        if d.get("code") not in ("00000", "0"):
            err = f"Bitget: {d.get('msg')} ({d.get('code')})"
            log.warning("[REST-GET] %s 오류: %s", path, err)
            self._tg_alert(f"⚠️ Bitget API 오류\n{err}\n경로: {path}")
            raise RuntimeError(err)
        return d

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
        log.debug("[REST] %s %s", path, body_str)
        r = _req.post("https://api.bitget.com" + path, headers=headers, data=body_str, timeout=10)
        d = r.json()
        if d.get("code") not in ("00000", "0"):
            err = f"Bitget: {d.get('msg')} ({d.get('code')})"
            # 40009 서명 오류 시 어떤 엔드포인트인지 로그에 남김
            log.warning("[REST] %s 오류: %s", path, err)
            self._tg_alert(f"⚠️ Bitget API 오류\n{err}\n경로: {path}")
            raise RuntimeError(err)
        return d

    def set_tp(self, symbol: str, trigger_price: float,
               hold_side: str, size: float) -> dict:
        """익절(TP) 주문 등록. One-way 모드 기준 — holdSide 미전송."""
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
        """손절(SL) 주문 등록. One-way 모드 — holdSide 미전송."""
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

    def place_plan_order(self, symbol: str, side: str, size: float,
                         trigger_price: float, order_price: float = None,
                         tp: float = None, sl: float = None) -> dict:
        """
        트리거(조건부) 진입 주문 — place-plan-order API.
        trigger_price 도달 시 order_price(지정가) 또는 시장가로 체결.

        side: "open_long" | "open_short"
        order_price: None이면 시장가(market), 값 있으면 지정가(limit)
        """
        action_map = {
            "open_long":   ("buy",  "open"),
            "open_short":  ("sell", "open"),
        }
        trade_side, trade_type = action_map.get(side, ("buy", "open"))
        body = {
            "symbol":       symbol if symbol.endswith("USDT") else f"{symbol}USDT",
            "productType":  "USDT-FUTURES",
            "marginMode":   "isolated",
            "marginCoin":   "USDT",
            "planType":     "normal_plan",
            "size":         str(size),
            "side":         trade_side,
            "tradeSide":    trade_type,
            "triggerPrice": str(trigger_price),
            "triggerType":  "mark_price",
            "orderType":    "limit" if order_price else "market",
        }
        if order_price:
            body["price"] = str(order_price)
        if tp:
            body["presetStopSurplusPrice"] = str(tp)
        if sl:
            body["presetStopLossPrice"] = str(sl)
        return self._rest_post("/api/v2/mix/order/place-plan-order", body)

    def cancel_all_tpsl(self, symbol: str) -> dict:
        """TP/SL + 진입 트리거(normal_plan) 플랜 주문 전체 취소."""
        ccxt_symbol = f"{symbol[:3]}/{symbol[3:]}:{symbol[3:]}"
        results = {}
        # TP/SL 취소
        try:
            results["tpsl"] = self._ex.cancel_all_orders(ccxt_symbol, params={
                "planType":    "profit_loss",
                "productType": "USDT-FUTURES",
            })
        except Exception as e:
            log.warning("[Bitget] TP/SL 취소 실패(무시): %s", e)
        # 진입 트리거 주문(normal_plan) 전체 취소 — 중복 등록 방지
        # Bitget v2: 개별 orderId가 필요하므로 pending 조회 후 일괄 취소
        try:
            _sym = symbol if symbol.endswith("USDT") else f"{symbol}USDT"
            _pending = self._rest_get(
                "/api/v2/mix/order/orders-plan-pending",
                {
                    "symbol":      _sym,
                    "productType": "USDT-FUTURES",
                    "planType":    "normal_plan",
                }
            )
            _orders = (_pending or {}).get("data", {}).get("entrustedList", [])
            for _o in _orders:
                _oid = _o.get("orderId")
                if _oid:
                    try:
                        self._rest_post(
                            "/api/v2/mix/order/cancel-plan-order",
                            {
                                "symbol":      _sym,
                                "productType": "USDT-FUTURES",
                                "orderId":     _oid,
                            }
                        )
                    except Exception:
                        pass
            results["trigger"] = f"취소 완료 {len(_orders)}건"
        except Exception as e:
            log.warning("[Bitget] 진입 트리거 취소 실패(무시): %s", e)
        return results

    def _tg_alert(self, msg: str):
        _tg.send(msg)


# ──────────────────────────────────────────
# TP/SL 유효성 검사
# ──────────────────────────────────────────
def _validate_tpsl(direction, price, tp, sl, entry=None):
    """TP/SL 방향 검증. entry가 있으면 entry 기준, 없으면 현재가 기준."""
    ref = entry if entry is not None else price
    if direction == "long":
        if tp is not None and tp <= ref:
            log.warning("[TPSL] 롱 TP(%.2f) ≤ 기준가(%.2f) → 무시", tp, ref); tp = None
        if sl is not None and sl >= ref:
            log.warning("[TPSL] 롱 SL(%.2f) ≥ 기준가(%.2f) → 무시", sl, ref); sl = None
    else:
        if tp is not None and tp >= ref:
            log.warning("[TPSL] 숏 TP(%.2f) ≥ 기준가(%.2f) → 무시", tp, ref); tp = None
        if sl is not None and sl <= ref:
            log.warning("[TPSL] 숏 SL(%.2f) ≤ 기준가(%.2f) → 무시", sl, ref); sl = None
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
            except Exception as _e:
                log.warning("[AutoTrader] 잔고 조회 실패 — 진입 차단: %s", _e)
                usdt = 0  # 폴백 $100 대신 0으로 진입 차단
        else:
            usdt = self.usdt_per_trade
        raw_size = (usdt * self.leverage) / price
        return self._round_size(raw_size)

    def _round_size(self, raw_size: float) -> float:
        """거래소 사이즈 스텝(최소 단위)에 맞게 내림 라운딩.

        Bitget BTC USDT-M Futures 의 사이즈 스텝은 0.01 BTC, ETH 는 0.1 ETH 등
        심볼마다 다름. 잔고가 작거나 BTC 가격이 높으면 raw_size 가 0.0073 같은
        값이 되어 거래소가 거부함 → step 단위로 내림(올림 아님 — 잔고 초과 방지).
        env BITGET_SIZE_STEP, BITGET_MIN_SIZE 로 오버라이드 가능.
        """
        import math
        step = float(os.environ.get("BITGET_SIZE_STEP", "0.01"))
        min_size = float(os.environ.get("BITGET_MIN_SIZE", "0.01"))
        # 내림 라운딩 (잔고 초과 방지)
        rounded = math.floor(raw_size / step) * step
        # 부동소수점 노이즈 정리 (0.01 -> 0.01, 0.30000000000000004 -> 0.3)
        # step 의 소수 자릿수만큼 round
        decimals = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
        rounded = round(rounded, decimals)
        if rounded < min_size:
            log.warning(
                "[AutoTrader] 계산 사이즈 %.6f < 최소 %.4f → 0 으로 차단",
                raw_size, min_size,
            )
            return 0.0
        return rounded

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
        # entry 기준으로 검증 (지정가 진입 시 현재가와 다를 수 있음)
        entry = float(trade_levels.get("entry") or price) if trade_levels else price
        return _validate_tpsl(direction, price, tp, sl, entry=entry)

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
            except Exception:
                pass
            self.client.close_all(self.symbol)
            # 청산 검증 — 0.8초 고정 sleep 대신 polling 으로 확인.
            # 비트겟이 청산 처리하기 전에 신규 진입하면 헤지 모드 충돌 또는 거부 가능.
            cleared = False
            for _ in range(10):  # 최대 5초 (10 × 0.5초)
                time.sleep(0.5)
                if not self._current_side():
                    cleared = True
                    break
            if not cleared:
                log.warning("[AutoTrader] 반대 포지션 청산 미확인 → 진입 보류")
                result["reason"] = "반대 포지션 청산 미완료 → 다음 사이클 재시도"
                self._last = result
                return result

        # 독립(Isolated) 마진 + 레버리지 설정
        for hs in ("long", "short"):
            self.client.set_margin_mode(self.symbol, hs)
            self.client.set_leverage(self.symbol, self.leverage, hs)

        tp, sl = self._extract_tpsl(desired, price, trade_levels)
        result["tp"] = tp
        result["sl"] = sl

        # R:R 계산 기준가 — entry 파싱 성공 시 entry 사용, 없으면 현재가
        _entry_for_rr = float((trade_levels or {}).get("entry") or price)
        result["entry"] = _entry_for_rr

        # ── SL 필수 체크 ──────────────────────────────
        if sl is None:
            result["reason"] = "SL 미설정 → 진입 거부 (손실 무한 리스크)"
            self._last = result
            log.warning("[AutoTrader] SL 없음 → 진입 취소")
            return result

        # ── 최소 손익비(R:R) 체크 ────────────────────
        # TP 누락 시 자동 보정: AUTO_TRADE_AUTO_TP_RR 환경변수로 SL 거리의 N배를
        # 자동 TP 로 사용 (기본 1.5). 0 이면 기능 비활성화 → TP 없이는 진입 거부.
        _auto_tp_rr = float(os.environ.get("AUTO_TRADE_AUTO_TP_RR", "1.5"))
        if tp is None and _auto_tp_rr > 0:
            sl_dist = abs(sl - _entry_for_rr)
            if sl_dist > 0:
                if desired == "long":
                    tp = _entry_for_rr + sl_dist * _auto_tp_rr
                else:
                    tp = _entry_for_rr - sl_dist * _auto_tp_rr
                result["tp"] = tp
                log.info("[AutoTrader] TP 자동 보정 — SL 거리×%.2f → TP $%.2f", _auto_tp_rr, tp)

        if tp is not None:
            rr = abs(tp - _entry_for_rr) / abs(sl - _entry_for_rr)
            result["rr"] = round(rr, 2)
            # 부동소수점 epsilon: 자동 보정 비율(예 1.5)이 MIN_RR(1.5)와 같으면
            # 부동소수점 오차로 1.4999... 가 나와 거부되는 경우가 있음 → 0.01 허용.
            if rr < self.MIN_RR - 0.01:
                result["reason"] = (
                    f"R:R {rr:.2f} < 최소 {self.MIN_RR} → 진입 거부"
                    f" (TP ${tp:,.2f} / SL ${sl:,.2f} / 진입가 ${_entry_for_rr:,.2f})"
                )
                self._last = result
                log.warning("[AutoTrader] R:R 불충분(%.2f) → 진입 취소", rr)
                return result
        else:
            # TP 없고 자동 보정도 비활성 → 진입 거부 (R:R 검증 불가)
            result["rr"] = None
            result["reason"] = (
                "TP 미설정 + 자동 보정 비활성 → 진입 거부 (R:R 검증 불가)"
            )
            self._last = result
            log.warning("[AutoTrader] TP 없음 + 자동 보정 off → 진입 취소")
            return result

        size = self._contracts(price)
        if size <= 0:
            # 사유: 잔고 부족 / 라운딩 후 최소 사이즈 미만
            result["reason"] = (
                "사이즈 0 → 진입 차단 (잔고 부족 또는 거래소 최소 단위 미만)"
            )
            self._last = result
            log.warning("[AutoTrader] size=0 → 진입 취소")
            return result

        order_side = "open_long" if desired == "long" else "open_short"

        # ── 진입가 기반 시장가 / 트리거 주문 결정 ──────────────────────────────
        # ENTRY_BUFFER(기본 0.2%): 돌파 직후 소폭 위에서도 시장가 허용하는 슬리피지 여유
        ENTRY_BUFFER = float(os.environ.get("AUTO_TRADE_ENTRY_BUFFER", "0.002"))
        entry_price = trade_levels.get("entry") if trade_levels else None
        use_limit   = False
        use_trigger = False   # 미돌파 → 거래소에 트리거 주문 등록

        # entry_price 없으면 현재가 기준으로 즉시 진입하지 않고
        # 분석 텍스트에서 "돌파" "눌림" 키워드 확인 후 트리거 등록
        if not entry_price and trade_levels:
            _raw_entry = str(trade_levels.get("bull_trigger") or "")
            if _raw_entry and float(_raw_entry) > 0:
                entry_price = float(_raw_entry)
                log.info("[AutoTrader] entry 미파싱 → bull_trigger $%.2f 사용", entry_price)

        if entry_price and price:
            if desired == "long":
                if price >= entry_price:
                    # 이미 진입가 이상 — 되돌림 대기 (지정가)
                    use_limit = True
                    log.info("[AutoTrader] 롱 — 현재가($%.2f) >= 진입가($%.2f) → 지정가 되돌림 대기",
                             price, entry_price)
                else:
                    # 미돌파 → 트리거 주문 등록
                    use_trigger = True
                    log.info("[AutoTrader] 롱 — 미돌파(현재가 $%.2f < 진입가 $%.2f) → 트리거 주문 등록",
                             price, entry_price)

            elif desired == "short":
                if price <= entry_price:
                    # 이미 진입가 이하 — 되돌림 대기 (지정가)
                    use_limit = True
                    log.info("[AutoTrader] 숏 — 현재가($%.2f) <= 진입가($%.2f) → 지정가 되돌림 대기",
                             price, entry_price)
                else:
                    # 미이탈 → 트리거 주문 등록
                    use_trigger = True
                    log.info("[AutoTrader] 숏 — 미이탈(현재가 $%.2f > 진입가 $%.2f) → 트리거 주문 등록",
                             price, entry_price)

        if use_trigger and entry_price:
            # 트리거 가격 도달 시 시장가 즉시 체결
            # 지정가로 하면 돌파 순간 가격이 올라가 미체결 가능성 있음
            order_resp = self.client.place_plan_order(
                self.symbol, order_side, size,
                trigger_price=entry_price,
                order_price=None,   # 시장가 체결
                tp=tp, sl=sl,
            )
            result["action"] = desired
            result["order"]  = order_resp
            result["reason"] = (
                f"{signal} (확신도 {confidence}%) → 트리거 주문 등록 @ ${entry_price:,.2f} "
                f"(현재가 ${price:,.2f} 미돌파, 도달 시 자동 체결)"
                + (f" | TP ${tp:,.2f}" if tp else " | TP 없음")
                + (f" | SL ${sl:,.2f}" if sl else " | SL 없음")
            )
            log.info("[AutoTrader] %s", result["reason"])
            result["tp_order"] = {"included_in_order": True} if tp else None
            result["sl_order"] = {"included_in_order": True} if sl else None
            self._last = result
            return result

        if use_limit and entry_price:
            order_resp = self.client.place_order(self.symbol, order_side, size, order_type="limit", price=entry_price, tp=tp, sl=sl)
            result["reason"] = f"지정가 진입 @ ${entry_price:,.2f}"
        elif not entry_price:
            # 진입가 없으면 시장가 진입 차단 — 반드시 되돌림/트리거 진입
            result["reason"] = "진입가 미설정 → 시장가 진입 차단 (되돌림 대기 필요)"
            log.info("[AutoTrader] 진입가 없음 → 시장가 차단")
            self._last = result
            return result
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
        _tg.send(msg)

    def last_result(self):   return dict(self._last)
    def get_positions(self): return self.client.get_positions(self.symbol)
    def get_account(self):   return self.client.get_account(self.symbol)
    def close_all(self):
        try: self.client.cancel_all_tpsl(self.symbol)
        except Exception:
            pass
        return self.client.close_all(self.symbol)

    def update_position_tpsl(self, new_tp=None, new_sl=None, breakeven_at_pct: float = 1.0) -> dict:
        """현재 보유 포지션의 거래소 TP/SL 업데이트.

        - 분석 사이클마다 호출되어 AI 권고 SL/TP 를 거래소에 반영.
        - 본전 이동(breakeven): 진입 후 breakeven_at_pct 이상 이익 시 SL 을 진입가로 이동.
          new_sl 이 None 이거나 본전보다 불리하면 본전이 우선 적용됨.

        반환: {updated: bool, reason: str, side, entry, applied_tp, applied_sl}
        """
        result = {"updated": False, "reason": "", "side": None, "entry": None,
                  "applied_tp": None, "applied_sl": None}
        try:
            positions = self.get_positions(self.symbol) if False else self.client.get_positions(self.symbol)
        except Exception as e:
            result["reason"] = f"포지션 조회 실패: {e}"
            return result
        if not positions:
            result["reason"] = "보유 포지션 없음"
            return result

        # 첫 번째 포지션만 처리 (단일 심볼 단일 포지션 가정)
        p = positions[0]
        side  = p.get("holdSide") or p.get("side")
        entry = float(p.get("averageOpenPrice", 0) or 0)
        size  = float(p.get("size", 0) or 0)
        cur   = float(p.get("markPrice", 0) or 0)
        if not side or entry <= 0 or size <= 0:
            result["reason"] = "포지션 정보 불완전"
            return result

        result["side"] = side
        result["entry"] = entry

        # 본전 이동 판정 — 현재가가 진입가 대비 breakeven_at_pct 이상 이익이면 SL 을 본전으로
        if cur > 0:
            profit_pct = ((cur - entry) / entry * 100) * (1 if side == "long" else -1)
            if profit_pct >= breakeven_at_pct:
                # 새 SL 후보가 본전보다 불리(더 멀리)하면 본전으로 강제
                if new_sl is None:
                    new_sl = entry
                else:
                    if side == "long" and new_sl < entry:
                        new_sl = entry
                    elif side == "short" and new_sl > entry:
                        new_sl = entry
                log.info("[AutoTrader] 본전 이동 적용 — profit %.2f%% → SL=$%.2f",
                         profit_pct, new_sl)

        # 새 TP/SL 방향 검증 — 진입가 기준으로 방향 오류 사전 차단
        new_tp, new_sl = _validate_tpsl(side, cur, new_tp, new_sl, entry=entry)

        # SL이 무효화돼도 TP는 독립적으로 처리
        # (SL이 진입가보다 위에 있는 경우 SL만 None, TP는 유효할 수 있음)
        if new_tp is None and new_sl is None:
            result["reason"] = "변경할 TP/SL 없음 (방향 검증 후 모두 무효)"
            return result
        # TP만 있고 SL 없어도 → TP만 업데이트
        # SL만 있고 TP 없어도 → SL만 업데이트

        # place-pos-tpsl: TP/SL 동시 등록 API 사용
        # cancel 후 재등록 방식 대신 단일 API로 처리 → 기존 SL/TP 소실 방지
        # TP 또는 SL 중 하나만 있으면 pending 조회 후 나머지 유지
        try:
            # 기존 pending tpsl 조회
            _pending = self.client._rest_get(
                "/api/v2/mix/order/orders-plan-pending",
                {
                    "symbol":      self.symbol if self.symbol.endswith("USDT") else f"{self.symbol}USDT",
                    "productType": "USDT-FUTURES",
                    "planType":    "profit_loss",
                }
            )
            _pending_list = (_pending or {}).get("data", {}).get("entrustedList", [])
            for _po in _pending_list:
                _pt = _po.get("planType", "")
                _pp = float(_po.get("triggerPrice", 0) or 0)
                if _pt == "profit_plan" and new_tp is None and _pp > 0:
                    new_tp = _pp
                    log.info("[AutoTrader] 기존 TP $%.2f 유지", new_tp)
                elif _pt == "loss_plan" and new_sl is None and _pp > 0:
                    new_sl = _pp
                    log.info("[AutoTrader] 기존 SL $%.2f 유지", new_sl)
        except Exception as _pe:
            log.warning("[AutoTrader] 기존 tpsl 조회 실패: %s", _pe)

        # 기존 취소 후 재등록
        try:
            self.client.cancel_all_tpsl(self.symbol)
        except Exception:
            pass

        try:
            if new_tp is not None:
                self.client.set_tp(self.symbol, float(new_tp), side, size)
                result["applied_tp"] = float(new_tp)
            if new_sl is not None:
                self.client.set_sl(self.symbol, float(new_sl), side, size)
                result["applied_sl"] = float(new_sl)
            result["updated"] = True
            result["reason"] = "TP/SL 거래소 반영 완료"
            log.info("[AutoTrader] TP/SL 갱신 완료 — side=%s entry=%.2f tp=%s sl=%s",
                     side, entry, result["applied_tp"], result["applied_sl"])
        except Exception as e:
            result["reason"] = f"TP/SL 등록 실패: {e}"
            log.warning("[AutoTrader] %s", result["reason"])
        return result
