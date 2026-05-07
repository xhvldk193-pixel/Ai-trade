import logging, requests
log = logging.getLogger(__name__)
_T = ""; _C = ""

def init(t, c):
    global _T, _C
    _T = t; _C = c

def send(m):
    if not _T or not _C:
        return
    try:
        # parse_mode 제거 — 메시지에 < 들어가면 HTML 파싱 실패로 전송 자체가 실패함.
        # 일반 텍스트로 보내면 어떤 문자가 들어와도 안전.
        requests.post(
            f"https://api.telegram.org/bot{_T}/sendMessage",
            json={"chat_id": _C, "text": m},
            timeout=5
        )
    except Exception:
        pass

def alert_trade(r):
    """실제 진입 성공 알림."""
    a = r.get("action", "")
    if a not in ("long", "short"):
        return
    e = "🟢 롱" if a == "long" else "🔴 숏"
    rr = r.get("rr")
    rr_str = f"\nR:R {rr:.2f}" if rr else ""
    entry = r.get("entry") or r.get("price", 0)
    send(
        f"{e} 진입\n"
        f"확신도 {r.get('confidence')}%\n"
        f"진입가 ${entry:,.0f}\n"
        f"TP ${r.get('tp') or 0:,.0f}\n"
        f"SL ${r.get('sl') or 0:,.0f}"
        f"{rr_str}"
    )

def alert_trade_rejected(r):
    """진입 거부 알림 — R:R 미달 / SL 없음 / 확신도 미달 / TP 미설정 / 사이즈 0."""
    reason = r.get("reason", "")
    # 새 거부 사유들도 잡도록 키워드 확장
    _reject_keywords = (
        "R:R", "SL 미설정", "확신도", "중복 방지",
        "TP 미설정",   # 자동 TP 보정 비활성 + AI 가 TP 안 줄 때
        "사이즈 0",    # 잔고 부족 또는 거래소 최소 단위 미만
        "최소 단위",   # 보강 키워드
    )
    if not any(kw in reason for kw in _reject_keywords):
        return
    signal = r.get("signal", "")
    confidence = r.get("confidence", 0)
    price = r.get("price", 0)
    tp = r.get("tp")
    sl = r.get("sl")
    rr = r.get("rr")

    icon = "⛔"
    lines = [
        f"{icon} 진입 거부",
        f"사유: {reason}",
        f"신호: {signal} | 확신도: {confidence}%",
        f"현재가: ${price:,.0f}",
    ]
    if tp:
        lines.append(f"TP: ${tp:,.0f}")
    if sl:
        lines.append(f"SL: ${sl:,.0f}")
    if rr:
        lines.append(f"R:R: {rr:.2f}")
    send("\n".join(lines))

def alert_analysis(a, p, min_confidence=65, **k):
    """분석 결과 알림."""
    if a.get("confidence", 0) < min_confidence:
        return
    s = a.get("signal", "")
    e = "📈" if s == "매수" else "📉" if s == "매도" else "⏸"
    send(f"{e} AI분석\n{s} 확신도{a.get('confidence')}%\n현재가 ${p:,.0f}")
