# LLM Evaluation for RAG Systems

## Why Evaluation Matters

Evaluating RAG systems is notoriously difficult. Unlike classification tasks, there is no single ground-truth label. Answers can be partially correct, stylistically varied, or factually accurate but incomplete. Systematic evaluation is critical to:
- Detect regressions when changing components
- Compare retrieval strategies objectively
- Identify failure modes (hallucination, poor recall, irrelevant answers)
- Guide production monitoring

## Evaluation Dimensions

### 1. Retrieval Quality
Measures how well the retrieval component finds relevant information.

**Metrics:**
- **Precision@k** — fraction of retrieved chunks that are relevant
- **Recall@k** — fraction of all relevant chunks that are retrieved
- **MRR (Mean Reciprocal Rank)** — position of the first relevant result
- **NDCG@k** — normalized discounted cumulative gain, accounts for rank position

### 2. Answer Faithfulness
Measures whether the generated answer is factually consistent with the retrieved context. A faithful answer does not introduce information not present in the context.

**Approach:** Decompose the answer into atomic claims; verify each claim against the context chunks.

**Scale:** 0.0 (fully hallucinated) to 1.0 (fully grounded)

### 3. Answer Relevance
Measures how well the answer addresses the original question. An answer may be faithful but not relevant (e.g., answering a different question than asked).

**Approach:** Generate synthetic questions from the answer and measure similarity to the original query.

### 4. Context Relevance / Precision
Fraction of the retrieved context that is actually relevant to answering the question. High context relevance means less noise in the LLM's input.

### 5. Context Recall
Given a ground-truth answer, measures whether all necessary information is present in the retrieved chunks. Requires ground-truth annotations.

## Evaluation Frameworks

### RAGAS
The most widely used RAG evaluation framework. Computes answer faithfulness, answer relevance, context precision, and context recall using LLM-as-judge scoring. Requires an LLM (often GPT-4) as the judge.

### TruLens
Offers a feedback function framework for RAG evaluation. Supports local LLMs as judges via Hugging Face integration.

### DeepEval
Testing framework for LLM applications. Provides out-of-the-box RAG metrics and integrates with pytest.

### Custom LLM-as-Judge
For fully local setups, implement custom judge prompts using a local LLM (e.g., llama3.2 via Ollama). Design prompts that output a score between 0 and 1 based on specific criteria.

## LLM-as-Judge Pattern

LLM-as-judge uses a language model to evaluate outputs. It works because:
- Strong LLMs have emergent capability for text quality assessment
- Flexible rubrics can be expressed in natural language
- Cost-effective alternative to human annotation at scale

**Best practices:**
- Use temperature=0 for deterministic and reproducible scores
- Prompt the judge to return only a numeric score, not prose
- Validate judge scores against human annotations periodically
- Use chain-of-thought before scoring for higher accuracy (slow but better)

**Example Judge Prompt:**
```
You are an evaluation judge. Given a query, context chunks, and an answer,
score how well the answer is supported by the context on a scale of 0 to 1.
Return only a decimal number between 0.0 and 1.0. Nothing else.

Query: {query}
Context: {context}
Answer: {answer}
Score:
```

## Agentic RAG Evaluation Layers

Standard RAG metrics focus on the final answer. Agentic RAG requires evaluating the entire decision process:

**Layer 1 — Plan Quality:** Did the agent select the right retrieval sources and strategy?

**Layer 2 — Retrieval Quality:** Were the retrieved chunks relevant to the query?

**Layer 3 — Refinement Quality:** When the agent refined its retrieval, did relevance improve?

**Layer 4 — Memory Efficiency:** Was cached/memory information used correctly?

**Layer 5 — Answer Alignment:** Is the final answer faithful and relevant?

## Ground Truth Construction

Building a reliable test set is the foundation of RAG evaluation.

**Strategies:**
1. **Manual annotation** — domain experts write (query, answer, sources) triples
2. **LLM-generated** — use a strong LLM to generate query-answer pairs from chunks
3. **Synthetic adversarial** — generate hard negatives and edge cases

**Test set properties:**
- Cover all retrieval sources in the system
- Include multi-hop queries requiring multiple chunks
- Include queries that should fail (out-of-scope)
- Balance query types (factual, comparative, procedural)

## Metrics Aggregation

For a multi-query evaluation run, report:
- Per-query scores for debugging
- Mean scores per layer for trend analysis
- Weak spot detection (flag layers below threshold, e.g., < 0.70)
- Worst-performing query trace for root cause analysis

## Continuous Evaluation in Production

- **Shadow evaluation** — run evaluator on a sample of live traffic
- **Drift detection** — alert if rolling average drops below baseline
- **Human-in-the-loop** — flag low-confidence answers for expert review
- **A/B testing** — compare RAG configurations on real users

## Common Failure Modes

| Failure Mode | Root Cause | Fix |
|-------------|-----------|-----|
| Hallucination | Weak grounding | Stricter faithfulness prompt |
| Incomplete answers | Poor recall | Increase k, better chunking |
| Off-topic answers | Bad retrieval plan | Improve planner prompt |
| Slow responses | Too many iterations | Tune confidence threshold |
| Memory pollution | Low-quality plan cached | Raise memory save threshold |
