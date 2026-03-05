import os
from typing import List

import chromadb
from langchain_ollama import OllamaEmbeddings


class VectorTool:
    """Retrieves document chunks from ChromaDB using semantic similarity."""

    def __init__(self, collection_name: str = "agentic_rag_docs", persist_dir: str = None):
        if persist_dir is None:
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            persist_dir = os.path.join(base, "chroma_db")

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Return top-k most semantically similar chunks for the given query."""
        try:
            query_embedding = self.embeddings.embed_query(query)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, max(self.collection.count(), 1)),
                include=["documents"],
            )
            docs = results.get("documents", [[]])[0]
            return [d for d in docs if d]
        except Exception as e:
            print(f"[VectorTool] Retrieval error: {e}")
            return []
