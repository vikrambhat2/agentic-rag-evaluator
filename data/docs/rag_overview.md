# Retrieval-Augmented Generation (RAG) Overview

## What is RAG?

Retrieval-Augmented Generation (RAG) is an AI framework that enhances large language model (LLM) outputs by incorporating relevant information retrieved from external knowledge sources. Rather than relying solely on the parametric knowledge baked into model weights during training, RAG augments generation with dynamic, up-to-date, or domain-specific context.

The core RAG pipeline consists of three stages:
1. **Indexing** — documents are chunked, embedded, and stored in a vector database.
2. **Retrieval** — given a user query, the most semantically relevant chunks are fetched.
3. **Generation** — the LLM produces an answer conditioned on the retrieved context.

## Why RAG?

LLMs have a knowledge cutoff and cannot access proprietary or real-time data. RAG bridges this gap by grounding responses in retrieved documents. Key advantages include:
- Reduced hallucination through factual grounding
- Lower cost compared to fine-tuning
- Easy knowledge base updates without retraining
- Auditability via source attribution

## Types of RAG

### Naive RAG
The simplest form: embed the query, retrieve top-k chunks by cosine similarity, concatenate them into a prompt, and generate. Fast and easy to implement but limited by retrieval precision.

### Advanced RAG
Improves upon naive RAG with techniques such as:
- **Query rewriting** — reformulate the query for better retrieval
- **HyDE (Hypothetical Document Embeddings)** — generate a hypothetical answer, embed it, and use that for retrieval
- **Re-ranking** — apply a cross-encoder to reorder initial results
- **Sentence-window retrieval** — store small chunks but retrieve surrounding context

### Modular RAG
Treats each component (retriever, ranker, reader) as an independent module. Allows mixing and matching strategies. Enables multi-hop reasoning across documents.

### Agentic RAG
Uses an LLM agent to orchestrate the retrieval process dynamically. The agent can:
- Decide which tools or data sources to query
- Issue multiple sub-queries
- Refine retrieval iteratively based on intermediate results
- Combine information from heterogeneous sources (vector DB, filesystem, APIs)

## RAG vs Fine-tuning

| Aspect | RAG | Fine-tuning |
|--------|-----|-------------|
| Knowledge updates | Easy (re-index) | Requires retraining |
| Hallucination | Lower (grounded) | Higher (parametric) |
| Latency | Higher (retrieval step) | Lower |
| Cost | Low operational | High upfront |
| Interpretability | High (sources visible) | Low |

## Evaluation Metrics

Common RAG evaluation dimensions:
- **Answer faithfulness** — is the answer supported by retrieved context?
- **Answer relevance** — does the answer address the query?
- **Context precision** — are retrieved chunks relevant?
- **Context recall** — are all necessary chunks retrieved?

## RAG in Production

Production-grade RAG systems require attention to:
- Chunking strategy (size, overlap, semantic boundaries)
- Embedding model quality and domain fit
- Vector database selection (scalability, filtering, metadata support)
- Prompt engineering for the reader LLM
- Continuous evaluation and monitoring pipelines
