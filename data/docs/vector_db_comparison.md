# Vector Database Comparison for RAG Systems

## What is a Vector Database?

A vector database stores high-dimensional numerical vectors (embeddings) and supports efficient approximate nearest neighbor (ANN) search. In RAG, the vector DB serves as the retrieval index where document chunks are stored and queried by semantic similarity.

## Key Features to Evaluate

1. **Query latency** — time to return top-k results
2. **Indexing throughput** — vectors indexed per second
3. **Scalability** — behavior at millions/billions of vectors
4. **Metadata filtering** — pre-filter by document attributes before ANN search
5. **Persistence** — disk-backed storage vs in-memory
6. **Deployment mode** — embedded, self-hosted, or managed cloud
7. **License** — open-source vs proprietary

## ChromaDB

**Type:** Embedded / self-hosted
**License:** Apache 2.0
**Language:** Python-native
**ANN Index:** HNSW (via hnswlib)

ChromaDB is optimized for ease of use in Python applications. It supports:
- Persistent storage to disk (SQLite + parquet)
- Metadata filtering via simple dictionary syntax
- Multiple distance metrics (cosine, L2, IP)
- Collections for namespace isolation

**Strengths:**
- Zero-dependency local setup (no Docker required)
- Native LangChain and LlamaIndex integration
- Excellent for prototyping and small-to-medium deployments (up to ~1M vectors)

**Weaknesses:**
- Not designed for distributed multi-node deployments
- Limited horizontal scalability compared to Weaviate or Qdrant

**Usage Example:**
```python
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("my_docs")
collection.add(embeddings=[[...]], documents=["text"], ids=["id1"])
results = collection.query(query_embeddings=[[...]], n_results=5)
```

## Qdrant

**Type:** Self-hosted / managed cloud
**License:** Apache 2.0
**Language:** Rust core, Python/Go/JS clients
**ANN Index:** HNSW

Qdrant is a high-performance vector DB designed for production workloads. Key features:
- Payload (metadata) filtering with complex conditions (must, should, must_not)
- Scalar and product quantization for memory efficiency
- Sparse vector support for hybrid dense+sparse search
- On-disk index option for memory-constrained deployments

**Performance:** Industry benchmarks show Qdrant consistently topping ANN benchmarks for recall at fixed latency budgets.

## Weaviate

**Type:** Self-hosted / managed cloud
**License:** BSD-3-Clause
**Language:** Go core, Python client
**ANN Index:** HNSW

Weaviate introduces a **GraphQL-based query interface** and built-in support for:
- Multi-modal embeddings (text + image)
- Generative search (RAG built-in)
- Module system for plugging in different embedding providers
- BM25 hybrid search

Weaviate is best suited for enterprise setups requiring rich query expressiveness.

## Pinecone

**Type:** Managed cloud only
**License:** Proprietary
**ANN Index:** Proprietary (pod-based or serverless)

Pinecone is a fully managed SaaS vector DB. It abstracts away all infrastructure concerns and offers:
- Serverless tier with usage-based pricing
- Namespaces for multi-tenancy
- Sparse-dense hybrid search

Trade-off: complete vendor lock-in and no local deployment option.

## FAISS

**Type:** Library (not a full DB)
**License:** MIT
**Language:** C++ core, Python bindings
**Organization:** Meta AI

FAISS (Facebook AI Similarity Search) is the foundational ANN library underlying many vector DBs. It supports dozens of index types (Flat, IVF, HNSW, PQ) and is optimized for GPU-accelerated search.

FAISS requires building your own storage, persistence, and metadata layers on top. Best for research or highly custom deployments.

## Comparison Table

| Database | Local? | Persistence | Metadata Filter | Scale | License |
|----------|--------|-------------|-----------------|-------|---------|
| ChromaDB | Yes | Yes (SQLite) | Basic | <1M | Apache 2.0 |
| Qdrant | Yes | Yes (RocksDB) | Advanced | 100M+ | Apache 2.0 |
| Weaviate | Yes | Yes | GraphQL | 100M+ | BSD-3 |
| Pinecone | No | Managed | Yes | 1B+ | Proprietary |
| FAISS | Yes | Manual | None | Unlimited | MIT |
| pgvector | Yes | PostgreSQL | SQL | 10M+ | Apache 2.0 |

## Choosing for Local RAG

For fully local, offline RAG with Python:
- **Small scale (< 100K chunks):** ChromaDB — simplest setup, no Docker needed
- **Medium scale (100K–10M chunks):** Qdrant local mode — best recall/latency
- **Need SQL-style metadata:** pgvector — embed in existing PostgreSQL

The `agentic-rag-evaluator` project uses **ChromaDB** with a persistent client for maximum simplicity and zero external service dependencies.

## Hybrid Search

Modern RAG systems combine dense (embedding) and sparse (BM25/TF-IDF) retrieval. Dense retrieval excels at semantic matching; sparse retrieval excels at exact keyword matching.

Vectors DBs with hybrid support:
- Qdrant: sparse + dense in single query
- Weaviate: BM25 + vector with RRF fusion
- Elasticsearch: kNN + BM25

Hybrid search typically improves recall by 5–15% over pure dense retrieval on diverse query sets.
