import os
from typing import List, Tuple


class FileTool:
    """Retrieves internal notes files using keyword overlap matching."""

    def __init__(self, notes_dir: str = None):
        if notes_dir is None:
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            notes_dir = os.path.join(base, "data", "notes")

        self.notes_dir = notes_dir
        self.files: List[Tuple[str, str]] = []  # (filename, content)
        self._load_files()

    def _load_files(self):
        """Load all .txt files from the notes directory."""
        if not os.path.isdir(self.notes_dir):
            print(f"[FileTool] Notes directory not found: {self.notes_dir}")
            return

        for fname in os.listdir(self.notes_dir):
            if fname.endswith(".txt"):
                fpath = os.path.join(self.notes_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()
                    self.files.append((fname, content))
                except Exception as e:
                    print(f"[FileTool] Error loading {fname}: {e}")

    def _keyword_overlap(self, query: str, content: str) -> int:
        """Count how many unique query words appear in the content (case-insensitive)."""
        query_words = set(w.lower() for w in query.split() if len(w) > 2)
        content_lower = content.lower()
        return sum(1 for w in query_words if w in content_lower)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Return top-k file contents ranked by keyword overlap with the query."""
        if not self.files:
            return []

        scored = []
        for fname, content in self.files:
            score = self._keyword_overlap(query, content)
            scored.append((score, fname, content))

        # Sort descending by overlap score
        scored.sort(key=lambda x: x[0], reverse=True)

        # Return top_k contents (only files with at least 1 keyword match)
        results = []
        for score, fname, content in scored[:top_k]:
            if score > 0:
                header = f"[File: {fname}]\n"
                results.append(header + content)

        return results
