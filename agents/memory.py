import json
import os
import re
import time
import threading
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
DEFAULT_MEMORY_DIR = Path(os.getenv("MEMORY_DIR", _PROJECT_DIR / "data" / "memory"))

@dataclass
class MemoryRecord:
    timestamp: str
    timestamp_unix: float
    situation: str
    advice: str
    outcome: str
    meta: dict

class FinancialSituationMemory:
    def __init__(self, role: str, memory_dir: Path = DEFAULT_MEMORY_DIR):
        self.role = role
        self.file_path = memory_dir / f"{role}.jsonl"
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: List[MemoryRecord] = []
        self.bm25 = None
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if not self.file_path.exists(): return
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                self.records.append(MemoryRecord(**d))
        self._rebuild_bm25()

    def _rebuild_bm25(self):
        if not _BM25_AVAILABLE or not self.records: return
        tokenized_corpus = [self._tokenize(r.situation) for r in self.records]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _tokenize(self, text: str):
        return re.findall(r'[\w\d]+', text.lower())

    def _save(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    def add_situation(self, situation: str, advice: str, outcome: str = "", meta: dict = None):
        with self._lock:
            for r in self.records[-5:]:
                if r.situation == situation: return None
            now = datetime.now(timezone.utc)
            rec = MemoryRecord(timestamp=now.isoformat(), timestamp_unix=now.timestamp(),
                               situation=situation, advice=advice, outcome=outcome, meta=meta or {})
            self.records.append(rec)
            self._save()
            self._rebuild_bm25()
            return rec

    def update_outcome(self, timestamp: str, outcome: str):
        with self._lock:
            for r in self.records:
                if r.timestamp == timestamp:
                    r.outcome = outcome
                    self._save()
                    return True
        return False

    def cleanup_old_no_outcome_records(self, days_threshold=3):
        """결과가 없는 3일 이상 된 노이즈 삭제 (손절/익절 등 결과가 있는 기록은 영구 보존)"""
        now = time.time()
        threshold_seconds = days_threshold * 24 * 3600
        with self._lock:
            initial_count = len(self.records)
            self.records = [
                rec for rec in self.records 
                if (rec.outcome and len(rec.outcome.strip()) > 0) or (now - rec.timestamp_unix < threshold_seconds)
            ]
            if len(self.records) != initial_count:
                self._rebuild_bm25()
                self._save()
        return initial_count - len(self.records)

    def get_memories(self, situation: str, top_k: int = 3):
        if not self.bm25 or not self.records: return []
        query = self._tokenize(situation)
        scores = self.bm25.get_scores(query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [{"record": asdict(self.records[i]), "score": scores[i]} for i in top_indices if scores[i] > 0]

_memories = {}
def get_memory(role: str) -> FinancialSituationMemory:
    if role not in _memories:
        _memories[role] = FinancialSituationMemory(role)
    return _memories[role]
