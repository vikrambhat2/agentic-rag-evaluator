# Embedding Models for RAG Systems

## What Are Embeddings?

Embeddings are dense numerical representations of text that capture semantic meaning. Two semantically similar texts will have embeddings with high cosine similarity. Embeddings are the foundation of semantic search and retrieval in RAG systems.

## Embedding Model Taxonomy

### API-Based Models
Hosted by cloud providers; require internet access and API keys.
- **OpenAI text-embedding-3-small** — 1536 dimensions, fast, cost-effective
- **OpenAI text-embedding-3-large** — 3072 dimensions, higher quality
- **Cohere Embed v3** — multilingual, supports search and classification tasks

### Open-Source Models (Self-Hosted)
Run locally without external dependencies.
- **nomic-embed-text** — 768 dimensions, Apache 2.0 license, strong MTEB scores, runs via Ollama
- **all-MiniLM-L6-v2** — 384 dimensions, extremely fast, good baseline for English
- **BGE-M3** — multilingual, supports dense, sparse, and multi-vector retrieval
- **E5-large-v2** — strong general-purpose encoder, instruction-following variants available
- **Instructor-XL** — task-specific embeddings via instruction prefixes

## nomic-embed-text

`nomic-embed-text` is a particularly strong choice for local RAG deployments:
- **Architecture:** Transformer encoder, 137M parameters
- **Context length:** 8192 tokens (much larger than most open models)
- **Dimensions:** 768
- **License:** Apache 2.0 (fully open)
- **MTEB benchmark:** Competitive with OpenAI text-embedding-ada-002
- **Ollama support:** Available as `ollama pull nomic-embed-text`

The model supports matryoshka representation learning, allowing truncation of embedding dimensions without major quality loss.

## Evaluation Benchmarks: MTEB

The Massive Text Embedding Benchmark (MTEB) is the standard evaluation suite for embedding models. It covers:
- Retrieval (15 datasets)
- Classification
- Clustering
- Semantic textual similarity
- Reranking
- Summarization

Top performers as of 2024 (retrieval-focused):
1. BGE-M3 (dense component): 54.9 nDCG@10
2. E5-mistral-7b-instruct: 56.9 nDCG@10
3. nomic-embed-text-v1.5: 53.1 nDCG@10
4. all-MiniLM-L6-v2: 41.9 nDCG@10

## Embedding Dimensions and Trade-offs

| Dimensions | Speed | Storage | Quality |
|-----------|-------|---------|---------|
| 384 | Very fast | Small | Moderate |
| 768 | Fast | Medium | Good |
| 1536 | Moderate | Large | High |
| 3072 | Slow | Very large | Highest |

For most local RAG deployments, 768-dimension models (nomic-embed-text, BGE-base) offer the best quality-to-performance ratio.

## Chunking and Embedding Alignment

The embedding model's maximum token context length determines the maximum viable chunk size:
- `all-MiniLM-L6-v2`: 256 tokens max
- `nomic-embed-text`: 8192 tokens max
- `text-embedding-3-small`: 8191 tokens max

Exceeding the model's context window causes silent truncation, degrading retrieval quality. Always set chunk size below the model's limit.

## Bi-Encoder vs. Cross-Encoder

**Bi-encoder (embedding model):**
- Encodes query and document independently
- Fast at retrieval time (pre-computed document embeddings)
- Used for initial recall

**Cross-encoder (reranker):**
- Encodes query and document jointly
- Much higher precision but slower (cannot pre-compute)
- Used after bi-encoder to reorder top-k results

The standard production pattern: bi-encoder retrieves top-100, cross-encoder reranks to top-5.

## Domain Adaptation

General-purpose embeddings may underperform on specialized domains (legal, medical, scientific). Options for adaptation:
1. **Fine-tune on domain pairs** — contrastive learning with (query, relevant-doc) pairs
2. **Instruction-following models** — prefix queries with domain-specific instructions (E5-instruct, Instructor-XL)
3. **Domain-specific models** — BioLinkBERT (biomedical), LegalBERT (legal)

## Embedding Caching

Embeddings are deterministic for a given model and input. Cache strategies:
- **Document cache** — persist embeddings in the vector DB; only re-embed on document changes
- **Query cache** — cache embeddings for repeated or similar queries
- **Redis/DiskCache** — fast lookup layers over expensive embedding calls

## Local Embedding with Ollama

```python
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector = embeddings.embed_query("What is retrieval-augmented generation?")
# Returns list of 768 floats
```

Ollama serves the embedding model as a local API, removing all external dependencies. Combined with ChromaDB, this enables a fully offline RAG pipeline.
