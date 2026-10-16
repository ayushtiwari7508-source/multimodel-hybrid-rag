# Vector Databases and Embeddings

## Introduction

Vector databases are specialized databases designed to store, index, and query high-dimensional vectors (embeddings). They are essential for semantic search, recommendation systems, and RAG applications.

## Popular Vector Databases

### Chroma
- Open-source, lightweight, embeddable
- Supports metadata filtering
- Persistent storage with SQLite backend
- Used in this hybrid RAG system

### Pinecone
- Fully managed cloud service
- High-scale, low-latency
- Enterprise features (namespaces, pod scaling)

### Weaviate
- Open-source with GraphQL API
- Built-in modules for embeddings (OpenAI, Cohere, HuggingFace)
- Hybrid search (BM25 + vector)

### Qdrant
- Rust-based, high performance
- Supports payload filtering
- Horizontal scaling

### Milvus
- Cloud-native, distributed
- Multiple index types (HNSW, IVF, FLAT)
- Kubernetes deployment

## Embedding Models

| Model | Dimensions | Context | License | Use Case |
|-------|------------|---------|---------|----------|
| nomic-embed-text | 768 | 8192 | Apache 2.0 | General purpose, code |
| bge-large-en-v1.5 | 1024 | 512 | MIT | High quality retrieval |
| e5-large-v2 | 1024 | 512 | MIT | Multilingual, asymmetric tasks |
| text-embedding-3-large | 3072 | 8192 | OpenAI | Highest quality, paid API |
| all-MiniLM-L6-v2 | 384 | 256 | Apache 2.0 | Fast, lightweight |

## Indexing Algorithms

### HNSW (Hierarchical Navigable Small World)
- Graph-based approximate nearest neighbor
- Logarithmic search complexity
- Parameters: M (connections), ef_construction, ef_search
- Best for high-recall requirements

### IVF (Inverted File Index)
- Partition vectors into clusters (centroids)
- Search only relevant clusters
- Faster build, lower recall than HNSW
- Parameters: nlist (clusters), nprobe (clusters to search)

### FLAT (Brute Force)
- Exact search, no approximation
- O(N) complexity
- Only for small datasets (< 100k vectors)

## Quantization

| Method | Memory Reduction | Quality Loss | Speedup |
|--------|-----------------|--------------|---------|
| None (FP32) | 1x | 0% | 1x |
| FP16 | 2x | < 0.1% | 1.5-2x |
| INT8 | 4x | 0.5-1% | 2-4x |
| PQ (Product Quantization) | 8-16x | 1-3% | 4-8x |
| Binary | 32x | 5-10% | 10-20x |

## Hybrid Search

Combining keyword (BM25) and vector search:
1. **Score fusion**: Normalize and combine scores
2. **RRF**: Rank-based fusion (parameter-free)
3. **Learned fusion**: Train a model to combine scores

This system uses RRF with cross-encoder reranking for best quality.

## Code Example: Chroma Usage

```python
import chromadb
from chromadb.utils import embedding_functions

# Initialize client
client = chromadb.PersistentClient(path="./data/chroma")

# Create collection with embedding function
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="nomic-embed-text"
)
collection = client.create_collection("documents", embedding_function=ef)

# Add documents
collection.add(
    ids=["doc1", "doc2"],
    documents=["RAG combines retrieval and generation", "BM25 is sparse retrieval"],
    metadatas=[{"source": "intro.md"}, {"source": "bm25.md"}]
)

# Query
results = collection.query(
    query_texts=["What is RAG?"],
    n_results=5
)
```

## Performance Tuning

- **Batch embeddings**: Process 32-128 texts per batch
- **Persist index**: Save HNSW graph to disk
- **Warm up**: Pre-load index into memory
- **Monitor**: Track query latency, recall@k, index size

## References

- Johnson et al., "Billion-scale similarity search with GPUs", 2019
- Malkov & Yashunin, "HNSW: Efficient Approximate Nearest Neighbor", 2018
- Babenko & Lempitsky, "Product Quantization for Nearest Neighbor Search", 2014