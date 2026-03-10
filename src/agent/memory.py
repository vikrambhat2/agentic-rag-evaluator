import json
import os
from datetime import datetime
from typing import Optional

from src.models.schemas import RetrievalPlan

MEMORY_MAX_ENTRIES = 50
MEMORY_SAVE_THRESHOLD = 0.75
MEMORY_WORD_OVERLAP_THRESHOLD = 0.60


class MemoryStore:
    """Persistent memory for caching successful retrieval plans."""

    def __init__(self, memory_file: str = None):
        if memory_file is None:
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            memory_file = os.path.join(base, "data", "memory.json")

        self.memory_file = memory_file
        self._ensure_file()

    def _ensure_file(self):
        """Create memory.json if it does not exist."""
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        if not os.path.exists(self.memory_file):
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _load(self) -> list:
        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_all(self, entries: list):
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(entries, f, indent=2)
        except Exception as e:
            print(f"[MemoryStore] Error saving memory: {e}")

    @staticmethod
    def _tokenize(text: str):
        """Lowercase, strip punctuation, return set of words longer than 2 chars."""
        import re
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return set(t for t in tokens if len(t) > 2)

    def _word_overlap(self, q1: str, q2: str) -> float:
        """
        Compute symmetric word overlap: |intersection| / max(|A|, |B|).
        Less strict than Jaccard; captures "most words in common" even when
        one query is shorter. Punctuation-stripped before comparison.
        """
        words1 = self._tokenize(q1)
        words2 = self._tokenize(q2)
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        return len(intersection) / max(len(words1), len(words2))

    def check(self, query: str) -> Optional[RetrievalPlan]:
        """Return a cached RetrievalPlan if a similar query exists, else None."""
        entries = self._load()
        for entry in entries:
            stored_query = entry.get("query", "")
            overlap = self._word_overlap(query, stored_query)
            if overlap > MEMORY_WORD_OVERLAP_THRESHOLD:
                try:
                    plan = RetrievalPlan(**entry["plan"])
                    return plan
                except Exception as e:
                    print(f"[MemoryStore] Error parsing cached plan: {e}")
        return None

    def save(self, query: str, plan: RetrievalPlan, avg_relevance: float):
        """Save a plan to memory if avg_relevance meets the threshold."""
        if avg_relevance < MEMORY_SAVE_THRESHOLD:
            return

        entries = self._load()

        new_entry = {
            "query": query,
            "plan": plan.model_dump(),
            "avg_relevance": avg_relevance,
            "timestamp": datetime.utcnow().isoformat(),
        }

        entries.append(new_entry)

        # Cap at max entries (drop oldest)
        if len(entries) > MEMORY_MAX_ENTRIES:
            entries = entries[-MEMORY_MAX_ENTRIES:]

        self._save_all(entries)
