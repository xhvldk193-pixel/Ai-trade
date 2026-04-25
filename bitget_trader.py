"""
=============================================
Bitget USDT-M Futures 자동매매 모듈
=============================================
TP/SL 전략:
  Claude AI가 피보나치 저항/지지 기반으로 제안한
  '목표가(TP)'와 '손절가(SL)'를 그대로 거래소에 전달.
  AI 값이 없을 경우에만 ATR 폴백 사용.
"""
from __future__ import annotations

import hashlib
import hmac as _hmac
import json
import logging
import time
from typing import Optional

import requests

log = logging.getLogger(__name__)

BITGET_BASE = "https://api.bitget.com"


# ──────────────────────────────────────────
# REST 클라이언트
# ──────────────────────────────────────────
class BitgetClient:
    def __init__(self, api_key: str, secret_key: str, passphrase: str):
        self.api_key    = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self._s = requests.Session()

    def _sign(self, ts: str, method: str, path: str, body: str = "") -> str:
        pre = ts + method.upper() + path + body
        return _hmac.new(
            self.secret_key.encode(), pre.encode(), hashlib.sha256
        ).hexdigest()

    def _headers(self, method: str, path: str, body: str = "") -> dict:
        ts = str(int(time.time() * 1000))
        return {
            "ACCESS-KEY":        self.api_key,
            "ACCESS-SIGN":       self._sign(ts, method, path, body),
            "ACCESS-TIMESTAMP":  ts,
            "ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type":      "application/json",
            "locale":            "zh-CN",
        }

    def _get(self, path: str, params: dict | None = None) -> dict:
        qs = ("?" + "&".join(f"{k}={v}" for k, v in params.items())) if params else ""
        fp = path + qs
        r = self._s.get(BITGET_BASE + fp, headers=self._headers("GET", fp), timeout=10)
        r.raise_for_status()
        d = r.json()
        if d.get("code") not in ("00000", "0"):
            raise RuntimeError(f"Bitget API 오류: {d.get('msg')} (code={d.get('code')})")
        return d

    def _post(self, path: str, body: dict) -> dict:
        bs = json.dumps(body)
        r = self._s.post(BITGET_BASE + path, headers=self._headers("POST", path, bs),
                         data=bs, timeout=10)
        r.raise_for_status()
        d = r.json()
        if d.get("code") not in ("00000", "0"):
            raise RuntimeError(f"Bitget API 오류: {d.get('msg')} (code={d.get('code')})")
        return d

    # ── 계좌 ────────────────────────────────
    def get_account(self, symbol: str = "BTCUSDT", margin_coin: str = "USDT") -> dict:
        # v2: symbol은 필수, marginCoin은 필수
        d = self._get("/api/v2/mix/account/account",
                      {"symbol": symbol, "productType": "USDT-FUTURES", "marginCoin": margin_coin})
        data = d.get("data")
        if isinstance(data, list) and data:
            return data[0]
        if isinstance(data, dict):
            return data
        return {}

    def get_positions(self, symbol: str = "BTCUSDT", margin_coin: str = "USDT") -> list[dict]:
        d = self._get("/api/v2/mix/position/singlePosition",
                      {"symbol": symbol, "productType": "USDT-FUTURES", "marginCoin": margin_coin})
        raw = d.get("data") or []
        if isinstance(raw, dict):
            raw = [raw]
        return [p for p in raw if float(p.get("total", 0) or 0) > 0]

    # ── 레버리지 ────────────────────────────
    def set_leverage(self, symbol: str, leverage: int,
                     hold_side: str = "long", margin_coin: str = "USDT") -> dict:
        return self._post("/api/v2/mix/account/setLeverage", {
            "symbol":      symbol,
            "productType": "USDT-FUTURES",
            "marginCoin":  margin_coin,
            "leverage":    str(leverage),
            "holdSide":    hold_side,
        })

    # ── 주문 ─────────────────────────────────
    def place_order(self, symbol: str, side: str, size: float,
                    order_type: str = "market", price: Optional[float] = None,
                    margin_coin: str = "USDT") -> dict:
        body: dict = {
            "symbol":      symbol,
            "productType": "USDT-FUTURES",
            "marginCoin":  margin_coin,
            "size":        str(size),
            "side":        side,
            "orderType":   order_type,
            "force":       "normal",
        }
        if order_type == "limit" and price is not None:
            body["price"] = str(price)
        return self._post("/api/v2/mix/order/placeOrder", body)

    def close_all(self, symbol: str, margin_coin: str = "USDT") -> list[dict]:
        """보유 포지션 전체 시장가 청산."""
        positions = self.get_positions(symbol, margin_coin)
        results = []
        for p in positions:
            hold  = p.get("holdSide", "")
            total = float(p.get("total", 0))
            if total <= 0:
                continue
            side = "close_long" if hold == "long" else "close_short"
            results.append(self.place_order(symbol, side, total, "market", margin_coin=margin_coin))
        return results

    # ── TP / SL 플랜 주문 ────────────────────
    def set_tp(self, symbol: str, trigger_price: float,
               hold_side: str, size: float, margin_coin: str = "USDT") -> dict:
        """익절(Take Profit) 플랜 주문 등록."""
        return self._post("/api/v2/mix/order/placeTpslOrder", {
            "symbol":      symbol,
            "productType": "USDT-FUTURES",
            "marginCoin":   margin_coin,
            "planType":     "profit_plan",
            "triggerPrice": str(trigger_price),
            "triggerType":  "fill_price",
            "holdSide":     hold_side,
            "size":         str(size),
        })

    def set_sl(self, symbol: str, trigger_price: float,
               hold_side: str, size: float, margin_coin: str = "USDT") -> dict:
        """손절(Stop Loss) 플랜 주문 등록."""
        return self._post("/api/v2/mix/order/placeTpslOrder", {
            "symbol":      symbol,
            "productType": "USDT-FUTURES",
            "marginCoin":   margin_coin,
            "planType":     "loss_plan",
            "triggerPrice": str(trigger_price),
            "triggerType":  "fill_price",
            "holdSide":     hold_side,
            "size":         str(size),
        })

    def cancel_all_tpsl(self, symbol: str, margin_coin: str = "USDT") -> dict:
        """기존 TP/SL 플랜 주문 전체 취소 (포지션 전환 시 호출)."""
        try:
            return self._post("/api/v2/mix/order/cancelAllTpslOrder", {
                "symbol":      symbol,
                "productType": "USDT-FUTURES",
                "marginCoin": margin_coin,
            })
        except Exception as e:
            log.warning("[Bitget] TP/SL 취소 실패(무시): %s", e)
            return {}


# ──────────────────────────────────────────
# TP/SL 유효성 검사
# ──────────────────────────────────────────
def _validate_tpsl(
    direction: str,
    price:     float,
    tp:        Optional[float],
    sl:        Optional[float],
) -> tuple[Optional[float], Optional[float]]:
    """
    AI가 제안한 TP/SL의 방향 유효성을 검사합니다.

    롱:  TP > 현재가 > SL  이어야 유효
    숏:  TP < 현재가 < SL  이어야 유효

    방향이 잘못된 값은 None 처리 (거래소에 전달하지 않음).
    """
    if direction == "long":
        if tp is not None:
            if tp <= price:
                log.warning("[TPSL] 롱 TP(%.2f) ≤ 현재가(%.2f) → 무시", tp, price)
                tp = None
        if sl is not None:
            if sl >= price:
                log.warning("[TPSL] 롱 SL(%.2f) ≥ 현재가(%.2f) → 무시", sl, price)
                sl = None
    else:  # short
        if tp is not None:
            if tp >= price:
                log.warning("[TPSL] 숏 TP(%.2f) ≥ 현재가(%.2f) → 무시", tp, price)
                tp = None
        if sl is not None:
            if sl <= price:
                log.warning("[TPSL] 숏 SL(%.2f) ≤ 현재가(%.2f) → 무시", sl, price)
                sl = None
    return tp, sl


# ──────────────────────────────────────────
# AI 신호 → 자동매매 실행기
# ──────────────────────────────────────────
class BitgetAutoTrader:
    """
    Claude AI의 피보나치 기반 TP/SL을 그대로 Bitget 거래소에 전달합니다.

    TP/SL 결정 우선순위
    -------------------
    1. AI trade_levels["target"]  → TP  (피보나치 저항선 / 직전 고점)
    2. AI trade_levels["stop"]    → SL  (피보나치 지지선 / 직전 저점)
    3. AI 값이 None이면 해당 주문 생략 (ATR 폴백 없음)

    매매 실행 조건
    --------------
    - signal == "매수"  → 롱 진입 (반대 숏 청산 후)
    - signal == "매도"  → 숏 진입 (반대 롱 청산 후)
    - signal == "홀드"  → 무동작
    - confidence < min_confidence → 무동작
    - 이미 동일 방향 포지션 → 무동작 (중복 방지)
    """

    def __init__(
        self,
        api_key:        str,
        secret_key:     str,
        passphrase:     str,
        symbol:         str   = "BTCUSDT",
        usdt_per_trade: float = 20.0,
        leverage:       int   = 3,
        min_confidence: int   = 65,
        use_tp:         bool  = True,
        use_sl:         bool  = True,
    ):
        self.client         = BitgetClient(api_key, secret_key, passphrase)
        self.symbol         = symbol.replace("/", "").upper()
        self.usdt_per_trade = usdt_per_trade
        self.leverage       = leverage
        self.min_confidence = min_confidence
        self.use_tp         = use_tp
        self.use_sl         = use_sl
        self._last: dict    = {}

    # ── 헬퍼 ────────────────────────────────
    def _contracts(self, price: float) -> float:
        """USDT 거래금액 × 레버리지를 계약 수로 환산."""
        notional = self.usdt_per_trade * self.leverage
        return round(notional / price, 4)

    def _current_side(self) -> Optional[str]:
        """현재 보유 포지션 방향. 없으면 None."""
        positions = self.client.get_positions(self.symbol)
        if not positions:
            return None
        return positions[0].get("holdSide")  # "long" | "short"

    # ── TP/SL 추출 ───────────────────────────
    def _extract_tpsl(
        self,
        direction:    str,
        price:        float,
        trade_levels: dict | None,
    ) -> tuple[Optional[float], Optional[float]]:
        """
        AI trade_levels에서 TP(목표가)·SL(손절가)을 추출하고
        방향 유효성 검사를 거쳐 반환합니다.

        - trade_levels["target"] → TP
        - trade_levels["stop"]   → SL
        """
        tp: Optional[float] = None
        sl: Optional[float] = None

        if trade_levels:
            raw_tp = trade_levels.get("target")
            raw_sl = trade_levels.get("stop")
            if isinstance(raw_tp, (int, float)) and raw_tp > 0:
                tp = float(raw_tp)
            if isinstance(raw_sl, (int, float)) and raw_sl > 0:
                sl = float(raw_sl)

        # 방향 유효성 검사 (잘못된 AI 출력 방어)
        tp, sl = _validate_tpsl(direction, price, tp, sl)

        if tp is None and sl is None:
            log.warning(
                "[AutoTrader] AI TP/SL 모두 없음 (trade_levels=%s) → TP/SL 주문 생략",
                trade_levels,
            )
        elif tp is None:
            log.info("[AutoTrader] AI TP 없음 → TP 주문 생략, SL=%.2f만 등록", sl)
        elif sl is None:
            log.info("[AutoTrader] AI SL 없음 → SL 주문 생략, TP=%.2f만 등록", tp)
        else:
            rr = abs(tp - price) / abs(sl - price) if abs(sl - price) > 0 else 0
            log.info(
                "[AutoTrader] AI TP=%.2f / SL=%.2f | 손익비=%.2f:1",
                tp, sl, rr,
            )

        return tp, sl

    # ── 메인 실행 ────────────────────────────
    def execute(
        self,
        signal:       str,
        confidence:   int,
        price:        float,
        trade_levels: dict | None = None,
        tf_data:      dict | None = None,  # 서명 호환용 (미사용)
    ) -> dict:
        """
        AI 신호를 받아 Bitget에 주문을 실행합니다.

        Parameters
        ----------
        signal       : "매수" | "매도" | "홀드"
        confidence   : 0~100
        price        : 현재 체결가
        trade_levels : analyzer.parse_trade_levels() 결과
                       → trade_levels["target"]  : 피보나치 기반 TP (목표가)
                       → trade_levels["stop"]    : 피보나치 기반 SL (손절가)
        tf_data      : 미사용 (하위 호환 서명)
        """
        result: dict = {
            "action":     "none",
            "reason":     "",
            "order":      None,
            "tp_order":   None,
            "sl_order":   None,
            "tp":         None,
            "sl":         None,
            "signal":     signal,
            "confidence": confidence,
            "rr":         None,
        }

        # ── 확신도 필터 ──────────────────────
        if confidence < self.min_confidence:
            result["reason"] = (
                f"확신도 {confidence}% < 최솟값 {self.min_confidence}% → 패스"
            )
            self._last = result
            log.info("[AutoTrader] %s", result["reason"])
            return result

        # ── 홀드 신호 ────────────────────────
        if signal == "홀드":
            result["reason"] = "홀드 신호 → 무동작"
            self._last = result
            return result

        desired  = "long" if signal == "매수" else "short"
        current  = self._current_side()

        # ── 중복 방지 ────────────────────────
        if current == desired:
            result["reason"] = f"이미 {desired} 포지션 보유 → 중복 진입 방지"
            self._last = result
            log.info("[AutoTrader] %s", result["reason"])
            return result

        # ── 반대 포지션 청산 ─────────────────
        if current and current != desired:
            log.info("[AutoTrader] 반대 포지션(%s) 청산 + 기존 TP/SL 취소", current)
            self.client.cancel_all_tpsl(self.symbol)
            self.client.close_all(self.symbol)
            time.sleep(0.8)

        # ── 레버리지 설정 (롱·숏 양쪽) ──────
        for hs in ("long", "short"):
            try:
                self.client.set_leverage(self.symbol, self.leverage, hold_side=hs)
            except Exception as e:
                log.warning("[AutoTrader] 레버리지(%s) 설정 실패: %s", hs, e)

        # ── AI TP/SL 추출 (피보나치 기반) ───
        tp, sl = self._extract_tpsl(desired, price, trade_levels)
        result["tp"] = tp
        result["sl"] = sl
        if tp and sl:
            rr = round(abs(tp - price) / abs(sl - price), 2) if abs(sl - price) > 0 else None
            result["rr"] = rr

        # ── 진입 주문 ────────────────────────
        size       = self._contracts(price)
        order_side = "open_long" if desired == "long" else "open_short"
        order_resp = self.client.place_order(self.symbol, order_side, size)
        result["action"] = desired
        result["order"]  = order_resp
        result["reason"] = (
            f"{signal} (확신도 {confidence}%) → {desired} {size}계약 @ ${price:,.2f}"
            + (f" | TP ${tp:,.2f}" if tp else " | TP 없음")
            + (f" | SL ${sl:,.2f}" if sl else " | SL 없음")
            + (f" | RR {result['rr']}:1" if result.get("rr") else "")
        )
        log.info("[AutoTrader] %s", result["reason"])

        # ── TP 주문 등록 ─────────────────────
        if self.use_tp and tp:
            try:
                result["tp_order"] = self.client.set_tp(
                    self.symbol, tp, desired, size
                )
                log.info("[AutoTrader] TP 등록 완료 → $%.2f", tp)
            except Exception as e:
                log.warning("[AutoTrader] TP 등록 실패: %s", e)
                result["tp_order"] = {"error": str(e)}

        # ── SL 주문 등록 ─────────────────────
        if self.use_sl and sl:
            try:
                result["sl_order"] = self.client.set_sl(
                    self.symbol, sl, desired, size
                )
                log.info("[AutoTrader] SL 등록 완료 → $%.2f", sl)
            except Exception as e:
                log.warning("[AutoTrader] SL 등록 실패: %s", e)
                result["sl_order"] = {"error": str(e)}

        self._last = result
        return result

    # ── 편의 메서드 ─────────────────────────
    def last_result(self) -> dict:
        return dict(self._last)

    def get_positions(self) -> list[dict]:
        return self.client.get_positions(self.symbol)

    def get_account(self) -> dict:
        return self.client.get_account(self.symbol)

    def close_all(self) -> list[dict]:
        """수동 전체 청산: TP/SL 취소 후 포지션 청산."""
        self.client.cancel_all_tpsl(self.symbol)
        return self.client.close_all(self.symbol)
