import json
import os
from typing import List


class APITool:
    """Retrieves entries from a mock API JSON file using keyword matching."""

    def __init__(self, api_file: str = None):
        if api_file is None:
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            api_file = os.path.join(base, "data", "api_mock.json")

        self.api_file = api_file
        self.data: dict = {}
        self._load()

    def _load(self):
        """Load the mock API JSON file."""
        try:
            with open(self.api_file, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except Exception as e:
            print(f"[APITool] Error loading api_mock.json: {e}")
            self.data = {}

    def _entry_to_str(self, key: str, entry: dict) -> str:
        """Format a JSON entry as a readable string."""
        lines = [f"[API:{key}]"]
        for k, v in entry.items():
            if isinstance(v, list):
                v_str = ", ".join(str(i) for i in v)
                lines.append(f"  {k}: {v_str}")
            else:
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def _keyword_score(self, query: str, text: str) -> int:
        """Count keyword matches between query and text (case-insensitive)."""
        query_words = set(w.lower() for w in query.split() if len(w) > 2)
        text_lower = text.lower()
        return sum(1 for w in query_words if w in text_lower)

    def retrieve(self, query: str) -> List[str]:
        """Return matching API entries as formatted strings."""
        if not self.data:
            return []

        results = []

        # Search across all top-level list fields
        for field_name, field_value in self.data.items():
            if not isinstance(field_value, list):
                continue

            for entry in field_value:
                if not isinstance(entry, dict):
                    continue

                entry_str = self._entry_to_str(field_name, entry)
                score = self._keyword_score(query, entry_str)
                if score > 0:
                    results.append((score, entry_str))

        # Sort by relevance score, return top 5
        results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in results[:5]]
