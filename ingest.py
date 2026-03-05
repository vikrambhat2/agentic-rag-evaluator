"""
ingest.py — Index documents into ChromaDB and generate test_set.json.

Usage:
    python ingest.py
"""

import json
import os
import sys
import time

import chromadb
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "data", "docs")
NOTES_DIR = os.path.join(BASE_DIR, "data", "notes")
API_FILE = os.path.join(BASE_DIR, "data", "api_mock.json")
TEST_SET_FILE = os.path.join(BASE_DIR, "data", "test_set.json")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

COLLECTION_NAME = "agentic_rag_docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_TEST_CASES = 20


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def load_markdown_files(docs_dir: str) -> list[dict]:
    """Load all .md files and return list of {text, source} dicts."""
    docs = []
    for fname in sorted(os.listdir(docs_dir)):
        if fname.endswith(".md"):
            fpath = os.path.join(docs_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read()
            docs.append({"text": text, "source": fname})
    return docs


def chunk_documents(docs: list[dict]) -> list[dict]:
    """Split documents into chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = []
    for doc in docs:
        splits = splitter.split_text(doc["text"])
        for i, split in enumerate(splits):
            chunks.append({
                "text": split,
                "source": doc["source"],
                "chunk_index": i,
            })
    return chunks


def index_chunks(chunks: list[dict], embeddings: OllamaEmbeddings) -> int:
    """Embed and store chunks in ChromaDB. Returns count stored."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection to start fresh
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Cleared existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c["text"] for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks]

    # Embed in batches to avoid memory issues
    batch_size = 20
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        print(f"  Embedding batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}...")
        try:
            embs = embeddings.embed_documents(batch)
            all_embeddings.extend(embs)
        except Exception as e:
            print(f"  [ERROR] Embedding batch failed: {e}")
            sys.exit(1)

    collection.add(
        embeddings=all_embeddings,
        documents=texts,
        metadatas=metadatas,
        ids=ids,
    )
    return len(chunks)


def generate_qa_from_chunk(chunk_text: str, source_type: str, llm: ChatOllama) -> dict | None:
    """Use ChatOllama to generate a (query, ground_truth) pair from a chunk."""
    prompt = f"""Given the following text excerpt, generate ONE factual question that can be answered from the text.
Then provide the concise answer.
Return ONLY a JSON object with keys "query" and "ground_truth". Nothing else.

Text: {chunk_text[:800]}

JSON:"""
    try:
        response = llm.invoke([("human", prompt)])
        raw = response.content if hasattr(response, "content") else str(response)
        # Strip markdown fences
        import re
        raw = re.sub(r"```(?:json)?\s*", "", raw)
        raw = re.sub(r"```\s*", "", raw).strip()
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return {
                "query": data.get("query", "").strip(),
                "ground_truth": data.get("ground_truth", "").strip(),
                "expected_sources": [source_type],
            }
    except Exception as e:
        print(f"  [WARN] QA generation failed: {e}")
    return None


def generate_qa_from_notes(notes_file: str, llm: ChatOllama) -> dict | None:
    """Generate QA from a notes file."""
    try:
        with open(notes_file, "r", encoding="utf-8") as f:
            content = f.read()
        return generate_qa_from_chunk(content[:1000], "filesystem", llm)
    except Exception as e:
        print(f"  [WARN] Notes QA generation failed: {e}")
        return None


def generate_qa_from_api(api_data: dict, llm: ChatOllama) -> dict | None:
    """Generate QA from api_mock.json content."""
    # Build a text snippet from the first trending technique
    try:
        techniques = api_data.get("trending_techniques", [])
        models = api_data.get("latest_models", [])
        snippets = []
        if techniques:
            t = techniques[0]
            snippets.append(f"Technique: {t['name']}. {t['description']}")
        if models:
            m = models[0]
            snippets.append(f"Model: {m['name']} by {m['provider']}. {m['highlights']}")
        text = "\n".join(snippets)
        return generate_qa_from_chunk(text, "api", llm)
    except Exception as e:
        print(f"  [WARN] API QA generation failed: {e}")
        return None


# ─────────────────────────────────────────────
# Hardcoded multi-source test cases
# ─────────────────────────────────────────────

MULTI_SOURCE_CASES = [
    {
        "query": "How does chunking strategy affect retrieval quality and what have internal experiments found about optimal chunk sizes?",
        "ground_truth": "Chunk size significantly impacts precision and recall. Internal experiments found 512 tokens with 50-token overlap performs best, balancing precision at 0.72 and recall at 0.74.",
        "expected_sources": ["vector_db", "filesystem"],
    },
    {
        "query": "What embedding model is recommended for local RAG and how does it compare to the latest models available?",
        "ground_truth": "nomic-embed-text is recommended for local RAG via Ollama with 768 dimensions. Recent model releases like Llama 3.2 and Qwen 2.5 are available but focus on generation, not embeddings.",
        "expected_sources": ["vector_db", "api"],
    },
    {
        "query": "What are the internal evaluation findings about memory systems and how does that relate to current RAG techniques?",
        "ground_truth": "Internal experiments show memory hit rate of 34% with 0.09 relevance improvement. Agentic RAG, a trending technique, benefits from memory to cache successful retrieval plans.",
        "expected_sources": ["filesystem", "api"],
    },
    {
        "query": "Compare ChromaDB for local RAG deployments with what our internal notes say about vector database selection and recent benchmark results.",
        "ground_truth": "ChromaDB suits small-to-medium deployments with Apache 2.0 license and simple Python setup. Internal notes confirm it as the choice for the agentic-rag-evaluator project. RAGBench shows open-source RAG scoring 0.78.",
        "expected_sources": ["vector_db", "filesystem", "api"],
    },
]


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("INGEST.PY — Agentic RAG Evaluator Setup")
    print("=" * 60)

    # 1. Load and chunk documents
    print("\n[1/4] Loading and chunking documents...")
    docs = load_markdown_files(DOCS_DIR)
    print(f"  Loaded {len(docs)} markdown files")
    chunks = chunk_documents(docs)
    print(f"  Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # 2. Embed and index
    print("\n[2/4] Embedding and indexing into ChromaDB...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    stored = index_chunks(chunks, embeddings)
    print(f"  Stored {stored} chunks in collection '{COLLECTION_NAME}' at {CHROMA_DIR}")

    # 3. Generate test set
    print("\n[3/4] Generating test_set.json...")
    llm = ChatOllama(model="llama3.2", temperature=0.3)
    test_cases = []

    # Vector DB queries — from doc chunks (sample every N chunks)
    step = max(1, len(chunks) // 8)
    sample_chunks = chunks[::step][:8]
    print(f"  Generating vector_db queries from {len(sample_chunks)} sample chunks...")
    for i, chunk in enumerate(sample_chunks):
        if len(test_cases) >= MAX_TEST_CASES - len(MULTI_SOURCE_CASES) - 3:
            break
        qa = generate_qa_from_chunk(chunk["text"], "vector_db", llm)
        if qa and qa["query"] and qa["ground_truth"]:
            test_cases.append(qa)
            print(f"    [{len(test_cases)}] {qa['query'][:60]}...")
        time.sleep(0.2)

    # Filesystem queries — from notes files
    print(f"\n  Generating filesystem queries from notes...")
    for fname in sorted(os.listdir(NOTES_DIR)):
        if not fname.endswith(".txt"):
            continue
        if len(test_cases) >= MAX_TEST_CASES - len(MULTI_SOURCE_CASES):
            break
        fpath = os.path.join(NOTES_DIR, fname)
        qa = generate_qa_from_notes(fpath, llm)
        if qa and qa["query"] and qa["ground_truth"]:
            test_cases.append(qa)
            print(f"    [{len(test_cases)}] {qa['query'][:60]}...")
        time.sleep(0.2)

    # API queries — from api_mock.json
    print(f"\n  Generating api queries from api_mock.json...")
    with open(API_FILE, "r", encoding="utf-8") as f:
        api_data = json.load(f)

    # Generate from different sections of the API data
    api_sources = []
    for technique in api_data.get("trending_techniques", [])[:2]:
        text = f"Technique: {technique['name']}. {technique['description']}"
        api_sources.append(text)
    for model in api_data.get("latest_models", [])[:2]:
        text = f"Model {model['name']} by {model['provider']}: {model['highlights']}"
        api_sources.append(text)
    for bench in api_data.get("recent_benchmarks", [])[:1]:
        text = f"Benchmark {bench['name']}: {bench['description']}. Top score: {bench['top_score']} by {bench['top_model']}."
        api_sources.append(text)

    for text_snippet in api_sources:
        if len(test_cases) >= MAX_TEST_CASES - len(MULTI_SOURCE_CASES):
            break
        qa = generate_qa_from_chunk(text_snippet, "api", llm)
        if qa and qa["query"] and qa["ground_truth"]:
            test_cases.append(qa)
            print(f"    [{len(test_cases)}] {qa['query'][:60]}...")
        time.sleep(0.2)

    # Add multi-source cases
    print(f"\n  Adding {len(MULTI_SOURCE_CASES)} hardcoded multi-source test cases...")
    test_cases.extend(MULTI_SOURCE_CASES)

    # Cap at MAX_TEST_CASES
    test_cases = test_cases[:MAX_TEST_CASES]

    # 4. Save test set
    print(f"\n[4/4] Saving test_set.json ({len(test_cases)} cases)...")
    with open(TEST_SET_FILE, "w", encoding="utf-8") as f:
        json.dump(test_cases, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"INGESTION COMPLETE")
    print(f"  Chunks stored in ChromaDB: {stored}")
    print(f"  Test cases generated: {len(test_cases)}")
    print(f"  Test set saved to: {TEST_SET_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
