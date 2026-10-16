# Cross-Encoder Reranking

## Overview

Cross-encoders are neural models that take a query-document pair as input and output a relevance score. Unlike bi-encoders (which embed query and document separately), cross-encoders attend to both simultaneously, enabling richer interaction modeling.

## Architecture

```
Query: "What is BM25?"          Document: "BM25 is a ranking function..."
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
            [CLS] Query [SEP] Document [SEP]
                       │
                       ▼
              Transformer Encoder
                       │
                       ▼
              [CLS] → Linear → Score
```

## Popular Cross-Encoder Models

| Model | Base | Parameters | Languages | Performance |
|-------|------|------------|-----------|-------------|
| bge-reranker-base | BERT-base | 110M | English | High quality |
| bge-reranker-large | BERT-large | 330M | English | Highest quality |
| bge-reranker-v2-m3 | BERT-base | 110M | 100+ | Multilingual |
| ms-marco-MiniLM-L-6-v2 | MiniLM | 22M | English | Fast, good quality |
| cross-encoder/ms-marco-TinyBERT-L-6 | TinyBERT | 14M | English | Very fast |

## Cross-Encoder vs Bi-Encoder

| Aspect | Bi-Encoder | Cross-Encoder |
|--------|------------|---------------|
| Inference | Embed once, search many | Joint encoding per pair |
| Speed | Very fast (ANN search) | Slow (O(N) per query) |
| Quality | Lower (no interaction) | Higher (full attention) |
| Use Case | First-stage retrieval | Second-stage reranking |

## Reranking Pipeline

1. **First-stage retrieval**: Retrieve top-K (e.g., 50-100) candidates using bi-encoder + ANN or BM25
2. **Cross-encoder scoring**: Score query against each candidate
3. **Re-rank**: Sort by cross-encoder score
4. **Select top-N**: Pass to generator (e.g., top-5)

## Score Calibration

Cross-encoder raw scores are not probabilities. Common approaches:

- **Sigmoid**: σ(score) → [0, 1]
- **Softmax**: exp(score) / Σ exp(scores) → distribution
- **Min-max normalization**: (score - min) / (max - min) → [0, 1]
- **Rank-based**: Use ranks instead of scores

This system uses min-max normalization followed by weighted fusion: `0.6 * RRF + 0.4 * normalized_rerank`

## Implementation in This System

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("BAAI/bge-reranker-base")
pairs = [[query, doc] for doc in candidate_docs]
scores = model.predict(pairs)  # Returns raw logits
normalized = (scores - scores.min()) / (scores.max() - scores.min())
```

## Performance Considerations

- **Batch inference**: Process multiple pairs together (batch_size=32-64)
- **GPU acceleration**: 10-50x speedup on GPU vs CPU
- **Model quantization**: INT8 reduces memory 4x with minimal quality loss
- **Caching**: Cache scores for repeated queries
- **Early exit**: Stop after top-K confirmed (speculative)

## Evaluation Metrics

- **MAP (Mean Average Precision)**: Overall ranking quality
- **NDCG@k**: Discounted cumulative gain at k
- **Recall@k**: Fraction of relevant docs in top-k
- **MRR (Mean Reciprocal Rank)**: 1/rank of first relevant doc

## Code Example

```python
from sentence_transformers import CrossEncoder

# Load model (lazy-loaded on first use)
reranker = CrossEncoder("BAAI/bge-reranker-base")

query = "How does BM25 work?"
candidates = [
    "BM25 uses TF-IDF with document length normalization",
    "BM25 is a probabilistic retrieval model",
    "Vector databases store embeddings for semantic search",
    "RAG combines retrieval with generation"
]

# Score all pairs
scores = reranker.predict([[query, c] for c in candidates])

# Rank by score
ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
for doc, score in ranked[:3]:
    print(f"{score:.4f}: {doc[:60]}...")
```

## Multilingual Reranking

For multilingual RAG, use `bge-reranker-v2-m3` which supports 100+ languages and can handle cross-lingual queries (query in one language, docs in another).

## References

- Nogueira & Cho, "Passage Re-ranking with BERT", 2019
- Gao et al., "BGE: Embedding and Reranking Models", 2023
- Xiao et al., "BGE-M3: Multi-Functionality, Multi-Linguality, Multi-Granularity", 2024