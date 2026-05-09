import json
import os
import re
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except Exception:
    BM25Okapi = None
    _BM25_AVAILABLE = False

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _THIS_DIR.parent
DEFAULT_MEMORY_DIR = Path(os.getenv("MEMORY_DIR", str(_PROJECT_DIR / "data" / "memory")))

@dataclass
class MemoryRecord:
    timestamp: str
    timestamp_unix: float
    situation: str
    advice: str
    outcome: str = ""
    meta: dict = None

class FinancialSituationMemory:
    def __init__(self, name: str, memory_dir: Optional[Path] = None):
        self.name = name
        self.memory_dir = memory_dir or DEFAULT_MEMORY_DIR
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.memory_dir / f"{name}.jsonl"
        self._lock = threading.Lock()
        self.records: List[MemoryRecord] = []
        self._load()

    def _load(self):
        if not self.path.exists(): return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                if "timestamp_unix" not in d:
                    d["timestamp_unix"] = time.time()
                self.records.append(MemoryRecord(**d))

    def _save(self):
        with self.path.open("w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    def add_situation(self, situation: str, advice: str, outcome: str = "", meta: dict = None):
        with self._lock:
            rec = MemoryRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                timestamp_unix=time.time(),
                situation=situation.strip(),
                advice=advice.strip(),
                outcome=outcome.strip(),
                meta=meta or {}
            )
            self.records.append(rec)
            self._save()
            return rec

    def update_outcome(self, timestamp: str, outcome: str) -> bool:
        with self._lock:
            for r in self.records:
                if r.timestamp == timestamp:
                    r.outcome = outcome
                    self._save()
                    return True
        return False

    def cleanup_old_no_outcome_records(self, days_threshold=3):
        """결과(outcome)가 없는 3일 이상 된 노이즈 데이터만 삭제"""
        now = time.time()
        threshold = days_threshold * 24 * 3600
        with self._lock:
            initial_count = len(self.records)
            self.records = [
                r for r in self.records 
                if (r.outcome and len(r.outcome.strip()) > 0) or (now - r.timestamp_unix < threshold)
            ]
            if len(self.records) != initial_count:
                self._save()
        return initial_count - len(self.records)

def get_memory(name: str = "analyst") -> FinancialSituationMemory:
    return FinancialSituationMemory(name)
