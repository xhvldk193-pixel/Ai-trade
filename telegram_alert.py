import logging, requests, time
log = logging.getLogger(__name__)
_T = ""; _C = ""

# 거부 알림 쓰로틀 — 같은 사유로 4시간마다 거부될 때 알림 반복되는 문제 해결.
# {reason_key: last_sent_ts} — 같은 사유 1시간 내 1번만 발송.
_last_reject_alerts: dict[str, float] = {}
_REJECT_THROTTLE_SECS = 3600  # 1시간

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
    """진입 거부 알림 — R:R 미달 / SL 없음 / 확신도 미달 / TP 미설정 / 사이즈 0.

    같은 사유가 반복될 때 알림 폭격 방지 — 1시간에 1번만 발송.
    """
    reason = r.get("reason", "")
    # 새 거부 사유들도 잡도록 키워드 확장
    _reject_keywords = (
        "R:R", "SL 미설정", "확신도", "중복 방지",
        "TP 미설정",   # 자동 TP 보정 비활성 + AI 가 TP 안 줄 때
        "사이즈 0",    # 잔고 부족 또는 거래소 최소 단위 미만
        "최소 단위",   # 보강 키워드
        "반대 포지션", # 청산 미완료
    )
    if not any(kw in reason for kw in _reject_keywords):
        return

    # 쓰로틀 체크 — 같은 사유 1시간 내 재발송 방지.
    # 사유 앞 30자로 그룹핑 (같은 사유면 가격만 다른 경우가 많아 prefix 매칭).
    now = time.time()
    throttle_key = reason[:30]
    last_sent = _last_reject_alerts.get(throttle_key, 0)
    if now - last_sent < _REJECT_THROTTLE_SECS:
        return  # 1시간 내 같은 사유 → 스킵
    _last_reject_alerts[throttle_key] = now

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
    lines.append("(1시간 내 같은 사유 알림은 스킵됩니다)")
    send("\n".join(lines))

def alert_analysis(a, p, min_confidence=65, **k):
    """분석 결과 알림."""
    if a.get("confidence", 0) < min_confidence:
        return
    s = a.get("signal", "")
    e = "📈" if s == "매수" else "📉" if s == "매도" else "⏸"
    send(f"{e} AI분석\n{s} 확신도{a.get('confidence')}%\n현재가 ${p:,.0f}")
