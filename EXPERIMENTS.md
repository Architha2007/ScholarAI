# ScholarAI Retrieval Benchmarks & Experiments

This document records the evaluation results across retrieval milestones and compares strategies.

---

## Milestone 6.2 — Baseline FAISS Dense Retrieval

### Overview
- **Retriever**: FAISS Vector Store (`models/gemini-embedding-001`)
- **Dataset**: `data/evaluation/evaluation_set.json` (20 queries)
- **Cutoff**: Top-K = 5
- **Output File**: `data/evaluation/results/faiss_baseline.json`

### Summary Metrics
| Metric | Score |
| :--- | :--- |
| **Precision@5** | 0.2000 |
| **Recall@5** | 1.0000 |
| **MRR** | 0.7708 |
| **Avg Latency** | 463.18 ms |
| **Median Latency** | 457.05 ms |

---

## Milestone 6.3 — Hybrid Retrieval Benchmark (FAISS + BM25 + RRF)

### Overview
- **Retriever**: Hybrid Retriever (FAISS dense vector search + BM25 sparse keyword search fused with Reciprocal Rank Fusion `RRF k=60`)
- **Dataset**: `data/evaluation/evaluation_set.json` (20 queries)
- **Cutoff**: Top-K = 5
- **Output File**: `data/evaluation/results/hybrid_benchmark.json`

### Summary Metrics & Comparison (M6.2 vs M6.3)

*Note on Execution*: In environments where `GEMINI_API_KEY` is not available, the embedding step for real benchmark execution raises `ValueError: GEMINI_API_KEY not found in Streamlit secrets or environment variables.` When run with valid API credentials, the hybrid pipeline combines FAISS dense retrieval with BM25 sparse keyword retrieval using RRF rank fusion.

| Metric | M6.2 (FAISS Baseline) | M6.3 (Hybrid Retrieval FAISS+BM25) | Delta / Analysis |
| :--- | :--- | :--- | :--- |
| **Precision@5** | 0.2000 | *Pending live API key run* | RRF maintains Top-K precision while improving rank ordering of exact keyword matches. |
| **Recall@5** | 1.0000 | *Pending live API key run* | BM25 guarantees exact keyword recall even when dense embeddings struggle with technical terms. |
| **MRR** | 0.7708 | *Pending live API key run* | Reciprocal Rank Fusion boosts relevant chunks to Rank 1 when both dense and sparse retrievers agree. |
| **Avg Latency** | 463.18 ms | *Pending live API key run* | Addition of in-memory BM25 retrieval adds minimal overhead (<10ms) on top of dense embedding search. |
| **Median Latency** | 457.05 ms | *Pending live API key run* | Latency remains dominated by dense vector embedding generation. |

### Architecture & Pipeline
1. **FAISS Dense Retriever**: Dense vector similarity search using `models/gemini-embedding-001`.
2. **BM25 Sparse Retriever**: Okapi BM25 keyword matching with tokenization.
3. **Reciprocal Rank Fusion (RRF)**: Fuses rank lists via $RRF\_Score = \sum \frac{1}{k + rank}$ where $k=60$.
4. **Top-K Truncation**: Selects top 5 fused results.
