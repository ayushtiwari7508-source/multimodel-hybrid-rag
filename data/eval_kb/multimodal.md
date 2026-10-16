# Multimodal RAG: Tables, Images, and Equations

## Overview

Multimodal RAG extends text-only retrieval to handle structured data (tables), visual content (images, charts), and mathematical notation. This document demonstrates the multimodal capabilities of the hybrid RAG system.

## Table Extraction

The system extracts tables from PDFs using multiple methods:

1. **pdfplumber**: Built-in table extraction from PDF text layer
2. **camelot**: Advanced table extraction (lattice/stream modes)
3. **img2table**: Image-based table detection + OCR

### Example Table: Model Comparison

| Model | Parameters | Context Length | Languages | License | Best For |
|-------|------------|----------------|-----------|---------|----------|
| nomic-embed-text | 137M | 8192 | English | Apache 2.0 | General retrieval |
| bge-large-en-v1.5 | 330M | 512 | English | MIT | High-quality English |
| e5-large-v2 | 330M | 512 | 100+ | MIT | Multilingual |
| text-embedding-3-large | 3072 | 8192 | 100+ | Proprietary | Maximum quality |

### Example Table: RRF Parameters

| C Value | Behavior | Use Case |
|---------|----------|----------|
| 10 | Strong rank preference | When top ranks are very reliable |
| 60 | Balanced (default) | General purpose hybrid search |
| 100 | More score-like | When scores are well-calibrated |

## Image Processing Pipeline

1. **Image Detection**: Identify images in PDF pages using pdfplumber
2. **Cropping**: Extract image regions with 200 DPI resolution
3. **Vision Description**: LLaVA-7b describes image content in detail
4. **OCR**: Tesseract extracts text from images
5. **Indexing**: Both description and OCR text are indexed as separate elements

### Supported Image Types

- Charts and graphs (bar, line, pie, scatter)
- Diagrams and flowcharts
- Screenshots and UI captures
- Photographs with text
- Mathematical figures

## Equation Extraction

The system detects LaTeX-style equations in text:

- **Inline**: `$E = mc^2$` or `\(E = mc^2\)`
- **Display**: `$$E = mc^2$$` or `\[E = mc^2\]`

### Example Equations

**Cosine Similarity**:
$$cos(\\theta) = \\frac{A \\cdot B}{\\|A\\| \\|B\\|}$$

**BM25 Formula**:
$$BM25(q, d) = \\sum_{i=1}^{n} IDF(q_i) \\cdot \\frac{TF(q_i, d) \\cdot (k_1 + 1)}{TF(q_i, d) + k_1 \\cdot (1 - b + b \\cdot \\frac{|d|}{avgdl})}$$

**RRF Score**:
$$RRF(d) = \\sum_{r \\in R} \\frac{1}{C + rank_r(d)}$$

**Cross-Entropy Loss**:
$$H(p, q) = -\\sum_{x} p(x) \\log q(x)$$

## Multimodal Query Routing

Queries referencing visual elements route to the multimodal path:

| Query | Route | Reason |
|-------|-------|--------|
| "What does the table show?" | multimodal | References table |
| "Describe the chart on page 3" | multimodal | References chart |
| "What does the diagram illustrate?" | multimodal | References diagram |
| "Explain the formula" | math | Mathematical content |
| "Show me the image" | multimodal | Visual reference |

## Code Example: Multimodal Ingestion

```python
from app.multimodal.loaders import load_pdf
from app.generation.ollama_client import OllamaClient

ollama = OllamaClient()
doc = load_pdf("report.pdf", ollama)

# Document contains mixed elements
for el in doc.elements:
    print(f"Type: {el.type}, Page: {el.page}, Section: {el.section}")
    print(f"Content: {el.content[:100]}...")
    print("---")
```

## Evaluation: Table Question Answering

| Question | Expected Answer Source | Element Type |
|----------|------------------------|--------------|
| What model has 3072 dimensions? | text-embedding-3-large | table |
| Which model is best for multilingual? | e5-large-v2 | table |
| What is the default RRF C value? | 60 | table |
| What equation defines cosine similarity? | Equation section | equation |

## References

- Chen et al., "Table Extraction from PDFs", 2022
- Liu et al., "LLaVA: Large Language and Vision Assistant", 2023
- Smith, "Tesseract OCR Engine", 2007