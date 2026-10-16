# RAG System Fundamentals

## Overview

Retrieval-Augmented Generation (RAG) combines information retrieval with text generation to provide grounded, factual answers. A hybrid RAG system uses both sparse (BM25) and dense (vector) retrieval, fused via Reciprocal Rank Fusion (RRF), followed by cross-encoder reranking.

## Architecture Components

### 1. Retrieval Layer

The retrieval layer has two parallel branches:

**Sparse Retrieval (BM25)**: 
- Tokenizes queries and documents using word-level tokenization
- Scores using term frequency (TF) and inverse document frequency (IDF)
- Formula: BM25(q, d) = Σ IDF(qi) * (TF(qi, d) * (k1 + 1)) / (TF(qi, d) + k1 * (1 - b + b * |d|/avgdl))
- Parameters: k1=1.2, b=0.75 (typical defaults)
- Fast, exact keyword matching, good for rare terms

**Dense Retrieval (Vector Search)**:
- Uses sentence transformers (e.g., nomic-embed-text) to embed queries and chunks into 768-dimensional vector space
- Similarity computed via cosine similarity: cos(θ) = (A·B) / (||A|| ||B||)
- Captures semantic meaning beyond exact keyword matches
- Slower but handles synonyms and paraphrases

### 2. Fusion (RRF)

Results from both retrievers are combined using Reciprocal Rank Fusion:

RRF(d) = Σ 1 / (C + rank(d)) where C=60 (smoothing constant)

RRF is parameter-free, robust, and doesn't require score calibration between retrievers.

### 3. Reranking

A cross-encoder (BAAI/bge-reranker-base) re-scores the top candidates jointly:
- Input: [query, document] pairs
- Output: relevance score (higher = more relevant)
- If unavailable, falls back to weighted blend: 0.6 * RRF + 0.4 * (normalized dense + BM25)

### 4. Generation with Routing

A router (qwen2.5:3b) classifies queries into routes:
- **code**: qwen2.5-coder:7b (Python, SQL, regex, debugging, shell)
- **math**: qwen2.5:7b (equations, proofs, formulas, derivations)
- **multimodal**: qwen2.5:7b (tables, charts, images, visual content)
- **fast**: qwen2.5:3b (short queries, chit-chat, < 5 tokens)
- **general**: qwen2.5:7b (default for complex RAG queries)

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| chunk_size | 900 | Token window for chunking |
| chunk_overlap | 150 | Overlap between adjacent chunks |
| top_k_bm25 | 25 | Sparse retrieval fan-out (RAG route) |
| top_k_dense | 25 | Dense retrieval fan-out (RAG route) |
| top_k_rerank | 6 | Final context size after reranking |
| rrf_c | 60 | RRF smoothing constant |
| max_context_tokens | 6000 | Context budget for LLM prompt |

## Equations

Cosine similarity: $cos(\\theta) = \\frac{A \\cdot B}{\\|A\\| \\|B\\|}$

BM25 score: $BM25(q, d) = \\sum_{i=1}^{n} IDF(q_i) \\cdot \\frac{TF(q_i, d) \\cdot (k_1 + 1)}{TF(q_i, d) + k_1 \\cdot (1 - b + b \\cdot \\frac{|d|}{avgdl})}$

RRF score: $RRF(d) = \\sum \\frac{1}{C + rank(d)}$ where $C=60$

## Code Examples

```python
def cosine_similarity(a, b):
    import numpy as np
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def bm25_score(query_terms, doc_terms, k1=1.2, b=0.75):
    # Simplified BM25 implementation
    pass
```

```sql
SELECT * FROM documents 
WHERE content LIKE '%RAG%' 
ORDER BY bm25_score(content, 'hybrid retrieval') DESC
LIMIT 10;
```

## Tables

| Route | Model | Use Case |
|-------|-------|----------|
| code | qwen2.5-coder:7b | Programming, debugging, SQL, regex |
| math | qwen2.5:7b | Equations, proofs, scientific formulas |
| multimodal | qwen2.5:7b | Tables, charts, images, diagrams |
| fast | qwen2.5:3b | Short questions, chit-chat, greetings |
| general | qwen2.5:7b | Default for complex RAG queries |

## References

- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", 2020
- Gao et al., "Rerankers for Hybrid Retrieval", 2023
- Rao et al., "Reciprocal Rank Fusion for Hybrid Search", 2009
- Robertson et al., "Okapi BM25: a non-probabilistic model", 1994