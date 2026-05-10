# =============================================
# Fib Stats — 피보나치 연장선별 TP 도달률 통계 & 자동 선택
#           + SL ATR 배수별 손절 발생률 통계 & 자동 조정
# =============================================
# 역할:
#   - Reflection 결과에서 TP 도달 여부 + 사용된 Fib 연장선을 추출
#   - 연장선별(1.272 / 1.618 / 2.0) TP 도달률을 누적 통계로 관리
#   - SL 손절 발생 여부 + 사용된 ATR 배수를 누적 통계로 관리
#   - 손절률이 높은 ATR 배수 → 다음 분석에서 더 넓은 배수 권고
#
# 저장 위치: ~/.crypto_analyzer/memory/fib_stats.json
# =============================================
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

_DEFAULT_STATS_PATH = Path.home() / ".crypto_analyzer" / "memory" / "fib_stats.json"

# 지원하는 피보나치 연장선
FIB_EXTENSIONS = (1.272, 1.618, 2.0)

# 지원하는 SL ATR 배수 후보
SL_ATR_MULTIPLIERS = (1.0, 1.2, 1.5, 2.0)

# 통계 신뢰 최소 샘플 수
MIN_SAMPLES = 5

# SL 손절률 임계값 — 이 이상이면 더 넓은 배수로 올림
SL_HIT_RATE_THRESHOLD = 0.4   # 40% 이상 손절 → ATR 배수 확대 권고

_DIRECTIONS = ("long", "short")


class FibStats:
    """
    피보나치 연장선별 TP 도달률 + ATR 배수별 SL 손절률을 누적하고
    각각 최적값을 반환한다.

    구조:
        {
          "tp": {
            "long":  { "1.272": {"hit": 7, "total": 10}, ... },
            "short": { ... }
          },
          "sl": {
            "long":  { "1.0": {"hit": 3, "total": 10}, ... },
            "short": { ... }
          }
        }
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else _DEFAULT_STATS_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data: dict = self._load()

    # ── 디스크 I/O ──────────────────────────────────────
    def _load(self) -> dict:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                for d in _DIRECTIONS:
                    raw.setdefault("tp", {}).setdefault(d, {})
                    raw.setdefault("sl", {}).setdefault(d, {})
                    for ext in FIB_EXTENSIONS:
                        raw["tp"][d].setdefault(str(ext), {"hit": 0, "total": 0})
                    for mult in SL_ATR_MULTIPLIERS:
                        raw["sl"][d].setdefault(str(mult), {"hit": 0, "total": 0})
                return raw
            except Exception:
                pass
        return {
            "tp": {d: {str(e): {"hit": 0, "total": 0} for e in FIB_EXTENSIONS}  for d in _DIRECTIONS},
            "sl": {d: {str(m): {"hit": 0, "total": 0} for m in SL_ATR_MULTIPLIERS} for d in _DIRECTIONS},
        }

    def _save(self) -> None:
        try:
            self.path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    # ── TP 기록 ─────────────────────────────────────────
    def record_tp(self, direction: str, fib_ext: float, tp_hit: bool) -> None:
        if direction not in _DIRECTIONS:
            return
        key = str(fib_ext)
        if key not in {str(e) for e in FIB_EXTENSIONS}:
            return
        with self._lock:
            b = self._data["tp"][direction].setdefault(key, {"hit": 0, "total": 0})
            b["total"] += 1
            if tp_hit:
                b["hit"] += 1
            self._save()

    # ── SL 기록 ─────────────────────────────────────────
    def record_sl(self, direction: str, atr_mult: float, sl_hit: bool) -> None:
        """
        SL 결과 기록.

        Parameters
        ----------
        direction : "long" | "short"
        atr_mult  : 사용된 ATR 배수 (1.0 / 1.2 / 1.5 / 2.0)
        sl_hit    : 손절 발생 여부 (True=손절 터치)
        """
        if direction not in _DIRECTIONS:
            return
        # 가장 가까운 지원 배수로 반올림
        closest = min(SL_ATR_MULTIPLIERS, key=lambda m: abs(m - atr_mult))
        key = str(closest)
        with self._lock:
            b = self._data["sl"][direction].setdefault(key, {"hit": 0, "total": 0})
            b["total"] += 1
            if sl_hit:
                b["hit"] += 1
            self._save()

    # ── TP 조회 ─────────────────────────────────────────
    def tp_hit_rate(self, direction: str, fib_ext: float) -> Optional[float]:
        key = str(fib_ext)
        with self._lock:
            b = self._data.get("tp", {}).get(direction, {}).get(key, {})
        total = b.get("total", 0)
        return b["hit"] / total if total >= MIN_SAMPLES else None

    def preferred_extensions(self, direction: str) -> tuple[float, ...]:
        """도달률 높은 순으로 정렬된 연장선 튜플."""
        rates = [(ext, self.tp_hit_rate(direction, ext)) for ext in FIB_EXTENSIONS]
        has_data = any(r is not None for _, r in rates)
        if not has_data:
            return FIB_EXTENSIONS
        with_data    = sorted([(e, r) for e, r in rates if r is not None], key=lambda x: (-x[1], x[0]))
        without_data = [e for e, r in rates if r is None]
        return tuple(e for e, _ in with_data) + tuple(without_data)

    # ── SL 조회 ─────────────────────────────────────────
    def sl_hit_rate(self, direction: str, atr_mult: float) -> Optional[float]:
        key = str(atr_mult)
        with self._lock:
            b = self._data.get("sl", {}).get(direction, {}).get(key, {})
        total = b.get("total", 0)
        return b["hit"] / total if total >= MIN_SAMPLES else None

    def recommended_atr_mult(self, direction: str) -> float:
        """
        손절률 기반 권장 ATR 배수 반환.

        - 현재 배수(1.0)의 손절률이 임계값(40%) 초과 → 더 넓은 배수 권고
        - 샘플 부족 시 기본값 1.0
        - 모든 배수가 임계값 초과 시 최대값(2.0) 반환
        """
        for mult in SL_ATR_MULTIPLIERS:
            rate = self.sl_hit_rate(direction, mult)
            if rate is None:
                return mult   # 샘플 부족 → 이 배수 사용
            if rate <= SL_HIT_RATE_THRESHOLD:
                return mult   # 손절률 허용 범위 → 이 배수 사용
        return SL_ATR_MULTIPLIERS[-1]  # 전부 초과 → 최대 배수

    def sl_summary(self, direction: str) -> str:
        """SL 통계 요약 문자열."""
        lines = [f"  SL {direction}:"]
        for mult in SL_ATR_MULTIPLIERS:
            b = self._data.get("sl", {}).get(direction, {}).get(str(mult), {})
            total = b.get("total", 0)
            hit   = b.get("hit", 0)
            rate  = f"{hit/total*100:.1f}% 손절" if total >= MIN_SAMPLES else f"샘플부족({total}건)"
            lines.append(f"    ATR×{mult}: {hit}/{total}  손절률={rate}")
        return "\n".join(lines)

    def log_summary(self) -> str:
        """로그용 전체 요약 문자열."""
        lines = ["[FibStats 도달률 요약]"]
        for d in _DIRECTIONS:
            lines.append(f"  TP {d}:")
            for ext in FIB_EXTENSIONS:
                b = self._data.get("tp", {}).get(d, {}).get(str(ext), {})
                total = b.get("total", 0)
                hit   = b.get("hit", 0)
                rate  = f"{hit/total*100:.1f}%" if total >= MIN_SAMPLES else f"샘플부족({total}건)"
                lines.append(f"    Fib {ext}: {hit}/{total}  도달률={rate}")
        lines.append("")
        lines.append("[SL ATR 배수별 손절률]")
        for d in _DIRECTIONS:
            lines.append(self.sl_summary(d))
            rec = self.recommended_atr_mult(d)
            lines.append(f"    → 권장 배수: ATR×{rec}")
        return "\n".join(lines)


# ── 싱글턴 ──────────────────────────────────────────────
_instance: Optional[FibStats] = None
_inst_lock = threading.Lock()


def get_fib_stats() -> FibStats:
    global _instance
    if _instance is None:
        with _inst_lock:
            if _instance is None:
                _instance = FibStats()
    return _instance


# ── Reflection 연동 헬퍼 ────────────────────────────────
def record_from_reflection(
    direction: str,
    fib_ext: Optional[float],
    tp_hit: bool,
) -> None:
    """TP 결과 기록 (fib_ext=None이면 폴백 케이스로 기록 안 함)."""
    if fib_ext is None:
        return
    get_fib_stats().record_tp(direction, fib_ext, tp_hit)


def record_sl_from_reflection(
    direction: str,
    atr_mult: Optional[float],
    sl_hit: bool,
) -> None:
    """SL 결과 기록 (atr_mult=None이면 기록 안 함)."""
    if atr_mult is None:
        return
    get_fib_stats().record_sl(direction, atr_mult, sl_hit)

