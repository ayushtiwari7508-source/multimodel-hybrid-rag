# Mathematical Foundations of RAG

## Vector Space Mathematics

### Cosine Similarity

The cosine similarity between two vectors A and B measures the cosine of the angle between them:

$$cos(\\theta) = \\frac{A \\cdot B}{\\|A\\| \\|B\\|} = \\frac{\\sum_{i=1}^{n} A_i B_i}{\\sqrt{\\sum_{i=1}^{n} A_i^2} \\sqrt{\\sum_{i=1}^{n} B_i^2}}$$

Properties:
- Range: [-1, 1] (typically [0, 1] for normalized embeddings)
- 1 = identical direction (maximum similarity)
- 0 = orthogonal (no similarity)
- -1 = opposite direction

For normalized embeddings: $cos(\\theta) = A \\cdot B$ since $\\|A\\| = \\|B\\| = 1$

### Euclidean Distance

$$d(A, B) = \\sqrt{\\sum_{i=1}^{n} (A_i - B_i)^2} = \\sqrt{\\|A\\|^2 + \\|B\\|^2 - 2(A \\cdot B)}$$

For normalized vectors: $d(A, B) = \\sqrt{2(1 - cos(\\theta))}$

### Dot Product

$$A \\cdot B = \\sum_{i=1}^{n} A_i B_i = \\|A\\| \\|B\\| cos(\\theta)$$

For normalized embeddings, dot product equals cosine similarity.

## BM25 Mathematical Formulation

### Term Frequency (TF)

$$TF(t, d) = \\frac{f_{t,d}}{\\max_{t'} f_{t',d}}$$

Or raw count: $TF(t, d) = f_{t,d}$

### Inverse Document Frequency (IDF)

$$IDF(t) = \\log \\frac{N - n_t + 0.5}{n_t + 0.5} + 1$$

Where:
- $N$ = total number of documents
- $n_t$ = number of documents containing term $t$

Alternative (Robertson-Spärck Jones):
$$IDF(t) = \\log \\frac{N}{n_t}$$

### BM25 Score

$$BM25(q, d) = \\sum_{t \\in q} IDF(t) \\cdot \\frac{TF(t, d) \\cdot (k_1 + 1)}{TF(t, d) + k_1 \\cdot (1 - b + b \\cdot \\frac{|d|}{avgdl})}$$

Parameters:
- $k_1$ = term frequency saturation (typically 1.2-2.0)
- $b$ = document length normalization (typically 0.75)
- $|d|$ = document length
- $avgdl$ = average document length in collection

### BM25 Variants

**BM25F** (field-weighted):
$$BM25F(q, d) = \\sum_{t \\in q} IDF(t) \\cdot \\frac{\\sum_f w_f \\cdot TF_f(t, d) \\cdot (k_1 + 1)}{\\sum_f w_f \\cdot TF_f(t, d) + k_1 \\cdot (1 - b + b \\cdot \\frac{|d|_f}{avgdl_f})}$$

**BM25+** (adds constant to avoid zero scores):
$$BM25+(q, d) = BM25(q, d) + \\delta$$

## Reciprocal Rank Fusion (RRF)

### RRF Formula

$$RRF(d) = \\sum_{r \\in R} \\frac{1}{C + rank_r(d)}$$

Where:
- $R$ = set of rankers (e.g., BM25, dense)
- $rank_r(d)$ = rank of document $d$ in ranker $r$ (1-indexed)
- $C$ = smoothing constant (typically 60)

### Properties

- Parameter-free (only $C$ which is robust)
- No score calibration needed between rankers
- Robust to outliers in individual rankers
- Theoretically grounded in rank aggregation

### Example

| Doc | BM25 Rank | Dense Rank | RRF (C=60) |
|-----|-----------|------------|------------|
| A   | 1         | 3          | 1/61 + 1/63 = 0.0323 |
| B   | 2         | 1          | 1/62 + 1/61 = 0.0322 |
| C   | 3         | 2          | 1/63 + 1/62 = 0.0311 |
| D   | 4         | -          | 1/64 = 0.0156 |

## Cross-Encoder Scoring

### Model Architecture

Cross-encoder computes: $score = f_{\\theta}([q; d])$ where $f_{\\theta}$ is a transformer.

Unlike bi-encoder: $score = cos(embed_{\\theta}(q), embed_{\\theta}(d))$

### Score Interpretation

Raw output is typically a logit. For ranking:
- Higher score = more relevant
- Scores not calibrated across queries
- Use within-query ranking only

### Normalization

Min-max within query:
$$score_{norm} = \\frac{score - \\min(scores)}{\\max(scores) - \\min(scores)}$$

Softmax within query:
$$score_{softmax} = \\frac{e^{score}}{\\sum_j e^{score_j}}$$

## RRF + Reranker Fusion

This system uses:
$$final\_score = 0.6 \\times RRF_{norm} + 0.4 \\times rerank_{norm}$$

Where both components are min-max normalized to [0, 1].

## Information Theory Metrics

### Entropy

$$H(X) = -\\sum_{x} p(x) \\log_2 p(x)$$

### Cross-Entropy

$$H(p, q) = -\\sum_{x} p(x) \\log_2 q(x)$$

### KL Divergence

$$D_{KL}(p \\| q) = \\sum_{x} p(x) \\log_2 \\frac{p(x)}{q(x)}$$

## Evaluation Metrics

### Precision@k

$$P@k = \\frac{|relevant \\cap top\_k|}{k}$$

### Recall@k

$$R@k = \\frac{|relevant \\cap top\_k|}{|relevant|}$$

### F1@k

$$F1@k = 2 \\cdot \\frac{P@k \\cdot R@k}{P@k + R@k}$$

### NDCG@k

$$DCG@k = \\sum_{i=1}^{k} \\frac{2^{rel_i} - 1}{\\log_2(i+1)}$$
$$NDCG@k = \\frac{DCG@k}{IDCG@k}$$

### MAP

$$AP = \\sum_{k=1}^{n} P(k) \\cdot rel(k)$$
$$MAP = \\frac{1}{|Q|} \\sum_{q \\in Q} AP(q)$$

### MRR

$$MRR = \\frac{1}{|Q|} \\sum_{q \\in Q} \\frac{1}{rank_q}$$

Where $rank_q$ is the rank of the first relevant document for query $q$.

## Probability and Statistics

### Bayes Theorem

$$P(A|B) = \\frac{P(B|A) P(A)}{P(B)}$$

### Law of Total Probability

$$P(A) = \\sum_i P(A|B_i) P(B_i)$$

### Expectation

$$E[X] = \\sum_x x \\cdot P(X=x)$$

### Variance

$$Var(X) = E[(X - \\mu)^2] = E[X^2] - (E[X])^2$$

## Linear Algebra for Embeddings

### Vector Norm

$$\\|v\\|_p = \\left(\\sum_i |v_i|^p\\right)^{1/p}$$
L2 norm: $\\|v\\|_2 = \\sqrt{\\sum_i v_i^2}$

### Orthogonal Projection

$$proj_u(v) = \\frac{v \\cdot u}{\\|u\\|^2} u$$

### Gram-Schmidt Orthogonalization

Given vectors $v_1, ..., v_n$:
$u_1 = v_1$
$u_k = v_k - \\sum_{j=1}^{k-1} proj_{u_j}(v_k)$

### Singular Value Decomposition (SVD)

$$A = U \\Sigma V^T$$

Where $U, V$ are orthogonal, $\\Sigma$ is diagonal.

## Optimization

### Gradient Descent

$$\\theta_{t+1} = \\theta_t - \\eta \\nabla L(\\theta_t)$$

### Adam Optimizer

$$m_t = \\beta_1 m_{t-1} + (1-\\beta_1) g_t$$
$$v_t = \\beta_2 v_{t-1} + (1-\\beta_2) g_t^2$$
$$\\hat{m}_t = m_t / (1-\\beta_1^t)$$
$$\\hat{v}_t = v_t / (1-\\beta_2^t)$$
$$\\theta_{t+1} = \\theta_t - \\eta \\hat{m}_t / (\\sqrt{\\hat{v}_t} + \\epsilon)$$

## References

- Robertson et al., "Okapi BM25 at TREC", 1994
- Craswell et al., "Reciprocal Rank Fusion", 2009
- Nogueira & Cho, "Passage Re-ranking with BERT", 2019
- Vaswani et al., "Attention Is All You Need", 2017
- Kingma & Ba, "Adam: A Method for Stochastic Optimization", 2015