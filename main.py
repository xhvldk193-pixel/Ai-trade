# 배포 플랫폼(Railway / Render / Fly.io)이 main:app 을 찾을 때 사용되는 진입점
from server import app  # noqa: F401

# ── 서버 시작 시 1회 마이그레이션 ──────────────────────
# analysis_history.jsonl → reflection 메모리 (이미 실행됐으면 스킵)
import os
from pathlib import Path

_BASE = Path(__file__).resolve().parent
_MEMORY_DIR = Path(os.getenv("MEMORY_DIR", str(_BASE / "data" / "memory")))

try:
    import json, sys
    sys.path.insert(0, str(_BASE))
    from agents.memory import get_memory as _gm

    _hist_path = _BASE / "data" / "analysis_history.jsonl"
    if _hist_path.exists():
        _mem = _gm("analyst")
        _existing = {r.timestamp for r in _mem.list_records()}
        _added = 0
        for _line in _hist_path.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if not _line:
                continue
            try:
                _e = json.loads(_line)
            except:
                continue
            _signal = _e.get("signal", "")
            _price  = _e.get("price")
            _ts     = _e.get("timestamp", "")
            if _signal in ("홀드", "HOLD", "hold"):
                continue
            if not isinstance(_price, (int, float)) or _price <= 0:
                continue
            # 이미 있는 타임스탬프면 스킵
            if _ts in _existing:
                continue
            _tl   = _e.get("trade_levels") or {}
            _conf = _e.get("confidence", 0)
            _situation = (
                f"[과거 분석 — {_ts}]\n"
                f"신호: {_signal} | 확신도: {_conf}% | 가격: ${_price:,.2f}\n"
                f"레짐: {_e.get('regime','')}\n"
                f"요약: {str(_e.get('summary',''))[:300]}"
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
                _added += 1
                _existing.add(_ts)

        if _added > 0:
            print(f"[migration] 과거 분석 {_added}건 메모리 임포트 완료", flush=True)
        else:
            print(f"[migration] 새로 임포트할 기록 없음 (전체 {_mem.size()}건 이미 저장됨)", flush=True)
except Exception as _me:
    print(f"[migration] 실패 (무시): {_me}", flush=True)

