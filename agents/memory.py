import json
import os
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    BM25Okapi = None
    _BM25_AVAILABLE = False

@dataclass
class MemoryRecord:
    timestamp: str
    timestamp_unix: float
    situation: str
    advice: str
    outcome: str = ""
    meta: dict = None

class FinancialSituationMemory:
    def __init__(self, name: str):
        self.name = name
        self._lock = threading.Lock()
        self.path = Path(f"data/memory/{name}.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: List[MemoryRecord] = []
        self._load()

    def _load(self):
        if not self.path.exists(): return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if "timestamp_unix" not in d: d["timestamp_unix"] = 0.0
                    self.records.append(MemoryRecord(**d))
                except: continue

    def add_situation(self, situation: str, advice: str, outcome: str = "", meta: dict = None):
        with self._lock:
            rec = MemoryRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                timestamp_unix=datetime.now().timestamp(),
                situation=situation, advice=advice, outcome=outcome, meta=meta or {}
            )
            self.records.append(rec)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            return rec

    def update_outcome(self, timestamp: str, outcome: str):
        with self._lock:
            updated = False
            for r in self.records:
                if r.timestamp == timestamp:
                    r.outcome = outcome
                    updated = True
            if updated:
                with self.path.open("w", encoding="utf-8") as f:
                    for r in self.records:
                        f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
            return updated

    def cleanup_old_no_outcome_records(self, days_threshold=3):
        now = datetime.now().timestamp()
        limit = days_threshold * 86400
        with self._lock:
            original_count = len(self.records)
            self.records = [r for r in self.records if r.outcome or (now - r.timestamp_unix < limit)]
            if len(self.records) != original_count:
                with self.path.open("w", encoding="utf-8") as f:
                    for r in self.records:
                        f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        return original_count - len(self.records)

def get_memory(name="analyst"):
    return FinancialSituationMemory(name)
