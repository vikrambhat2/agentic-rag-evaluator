# Chunking Strategies for RAG Systems

## Why Chunking Matters

Chunking is the process of splitting documents into smaller units before embedding and indexing. The chunk size and method significantly impact retrieval quality. Too-large chunks reduce precision; too-small chunks lose context. The optimal strategy depends on document type, query style, and embedding model.

## Fixed-Size Chunking

The simplest approach: split text into chunks of N tokens with optional overlap.

**Parameters:**
- `chunk_size`: number of tokens per chunk (commonly 256–1024)
- `chunk_overlap`: token overlap between consecutive chunks (commonly 10–20% of chunk_size)

**Advantages:**
- Predictable chunk count
- Fast and deterministic
- Works well for uniform text (e.g., articles, documentation)

**Disadvantages:**
- May split sentences or paragraphs mid-way
- Does not respect semantic boundaries
- Performance degrades with highly variable document structures

## Semantic Chunking

Uses sentence embeddings to detect topic shifts and create chunks at natural semantic boundaries.

**How it works:**
1. Embed each sentence individually
2. Compute cosine similarity between adjacent sentence pairs
3. Insert chunk boundaries where similarity drops below a threshold

**Advantages:**
- Preserves semantic coherence within chunks
- Better retrieval precision for conceptually dense documents

**Disadvantages:**
- Computationally expensive (requires embedding every sentence)
- Chunk sizes are variable and unpredictable
- Requires tuning of the similarity threshold

## Recursive Character Text Splitting

A pragmatic approach used by LangChain's `RecursiveCharacterTextSplitter`. Tries to split on progressively smaller separators in order: `["\n\n", "\n", " ", ""]`.

This preserves paragraph structure when possible, falls back to sentence splitting, then word splitting.

**Best for:** General-purpose text, mixed content, code documentation.

## Document-Aware Chunking

Splits along structural boundaries inherent to the document format:
- Markdown headings (`#`, `##`, `###`)
- HTML tags (`<section>`, `<p>`, `<article>`)
- JSON/XML schema boundaries
- Code function/class boundaries

**Advantages:**
- Chunks align with human-perceived document units
- Ideal for structured documentation and codebases

## Sliding Window Chunking

Creates overlapping chunks where each chunk starts M tokens after the previous one.

```
Chunk 1: tokens 0–511
Chunk 2: tokens 256–767
Chunk 3: tokens 512–1023
```

Higher overlap means more context redundancy at the cost of increased storage and retrieval noise.

## Sentence-Window Retrieval

Store individual sentences as chunks (for precise retrieval) but at query time, return the surrounding N sentences as context for the LLM. Balances retrieval granularity with generation context quality.

## Agentic Chunking (Proposition-Level)

Emerging technique: use an LLM to extract self-contained "propositions" from documents (atomic factual statements). Each proposition is stored as a chunk.

**Advantages:**
- Maximum semantic precision
- Each chunk is independently meaningful

**Disadvantages:**
- Very high ingestion cost (LLM call per document segment)
- Not suitable for large-scale document sets

## Choosing a Chunk Size

| Use Case | Recommended Chunk Size | Overlap |
|----------|----------------------|---------|
| FAQ / short answers | 128–256 tokens | 10–20 tokens |
| Technical documentation | 512–768 tokens | 50–100 tokens |
| Legal / academic papers | 768–1024 tokens | 100–150 tokens |
| Code files | Function-level | None |

## Impact on RAG Performance

Experiments consistently show that chunk size and overlap jointly affect:
- **Precision@k** — smaller chunks improve precision
- **Recall** — larger chunks improve recall
- **Faithfulness** — moderate chunks (512) tend to maximize answer faithfulness

There is no universally optimal chunk size. Always evaluate on your specific query distribution.

## Chunk Metadata

Attach metadata to every chunk for downstream filtering:
- `source_file` — origin document
- `page_number` — for PDFs
- `section_heading` — structural location
- `timestamp` — document version/date

Metadata-filtered retrieval can substantially improve precision without sacrificing recall.
