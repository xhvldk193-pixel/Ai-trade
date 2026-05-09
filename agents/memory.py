from __future__ import annotations
import json
import os
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except Exception:
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
    def __init__(self, name: str, dirname: Path):
        self.name = name
        self.path = dirname / f"{name}.jsonl"
        self._lock = threading.Lock()
        self.records: List[MemoryRecord] = []
        self._load()

    def _load(self):
        if not self.path.exists(): return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if "timestamp_unix" not in d: d["timestamp_unix"] = time.time()
                    self.records.append(MemoryRecord(**d))
                except: continue

    def add_situation(self, situation: str, advice: str, outcome: str = "", meta: dict = None):
        with self._lock:
            rec = MemoryRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                timestamp_unix=time.time(),
                situation=situation, advice=advice, outcome=outcome, meta=meta or {}
            )
            self.records.append(rec)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    def update_outcome(self, timestamp: str, outcome: str):
        with self._lock:
            for r in self.records:
                if r.timestamp == timestamp:
                    r.outcome = outcome
                    self._save_all()
                    return True
        return False

    def _save_all(self):
        with self.path.open("w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    def cleanup_old_records(self, days=3):
        limit = time.time() - (days * 86400)
        with self._lock:
            self.records = [r for r in self.records if r.outcome or r.timestamp_unix > limit]
            self._save_all()

    def get_memories(self, situation: str, top_k: int = 3):
        if not _BM25_AVAILABLE or not self.records: return []
        # 원본 BM25 검색 로직 (사용자 파일 내용 유지)
        tokenized_corpus = [re.sub(r'[^가-힣a-zA-Z0-9\s]', '', r.situation).split() for r in self.records]
        bm25 = BM25Okapi(tokenized_corpus)
        query = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', situation).split()
        scores = bm25.get_scores(query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [{"record": self.records[i], "score": scores[i]} for i in top_indices if scores[i] > 0]

class AgentMemories:
    def __init__(self, dirname: Optional[Path] = None):
        if dirname is None:
            dirname = Path("data/memory")
        dirname.mkdir(parents=True, exist_ok=True)
        self.dirname = dirname
        self._mems: Dict[str, FinancialSituationMemory] = {}

    def get(self, role: str) -> FinancialSituationMemory:
        if role not in self._mems:
            self._mems[role] = FinancialSituationMemory(role, self.dirname)
        return self._mems[role]

    def get_memories_text(self, role: str, situation: str, top_k: int = 3) -> str:
        mems = self.get(role).get_memories(situation, top_k)
        if not mems: return ""
        # 원본 출력 형식 유지
        lines = ["[과거 유사 상황 판단 이력]"]
        for i, item in enumerate(mems, 1):
            r = item["record"]
            lines.append(f"\n- 사례 {i} ({r.timestamp}):\n  주장: {r.advice[:200]}\n  결과: {r.outcome[:300]}")
        return "\n".join(lines)

_global_memories = None
def get_memory():
    global _global_memories
    if _global_memories is None:
        _global_memories = AgentMemories()
    return _global_memories
