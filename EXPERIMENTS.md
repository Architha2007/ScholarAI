# ScholarAI Retrieval Benchmark Experiments

This document records the evaluation benchmark experiments conducted for the ScholarAI Hybrid Retrieval System across standard research paper PDFs (`attention_all_you_need`, `dense_passage_retrieval`, `retrieval_augmented_generation`) evaluated against the 20-query ground-truth dataset (`data/evaluation/evaluation_set.json`).

---

## 📊 Benchmark Experiments Summary Table

| Milestone | Strategy / Retriever | Precision@5 | Recall@5 | MRR | Avg Latency (ms) | Median Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **M6.2** | Dense Vector Only (FAISS) | `0.2000` | `1.0000` | `0.7708` | 454.27 | 455.42 |
| **M6.3** | Hybrid Retrieval (FAISS + BM25 RRF) | **`0.2000`** | **`1.0000`** | **`0.9350`** | 464.67 | 458.36 |
| **M6.4** | Cross-Encoder Reranked (RRF Top-20 → Cross-Encoder Top-5) | `0.1600` | `0.8000` | `0.6492` | 3227.91 | 2519.69 |

---

## 🔬 Detailed Experiment Findings & Analysis

### 1. Milestone 6.2 — FAISS Baseline Benchmark
- **Retriever:** FAISS dense vector store using Gemini embeddings (`models/gemini-embedding-001`).
- **Precision@5:** `0.2000`
- **Recall@5:** `1.0000`
- **MRR:** `0.7708`
- **Avg Latency:** `454.27 ms`
- **Key Observation:** FAISS dense embeddings achieved 100% recall@5, successfully retrieving the ground-truth chunk in the top-5 results for all queries. However, for queries requiring exact term matching (e.g. mathematical formulas, precise paper titles, specific abbreviations), FAISS sometimes ranked the relevant chunk at rank 2–4 rather than rank 1, resulting in an MRR of 0.7708.

---

### 2. Milestone 6.3 — Hybrid Retrieval Benchmark (FAISS + BM25 RRF)
- **Retriever:** Reciprocal Rank Fusion (RRF with $k=60$) combining FAISS dense retrieval and BM25 sparse keyword retrieval.
- **Precision@5:** `0.2000`
- **Recall@5:** `1.0000`
- **MRR:** **`0.9350`** *(+21.3% relative improvement over FAISS baseline)*
- **Avg Latency:** `464.67 ms`
- **Key Observation:** Fusing BM25 sparse keyword scores with FAISS dense vector scores via Reciprocal Rank Fusion significantly improved ranking quality. Exact keyword matches (such as "Scaled Dot-Product Attention", "ORQA", "NIPS 2017") received strong BM25 rank boosts that pushed the true relevant chunk directly to **Rank 1** for almost all queries, increasing MRR from `0.7708` to `0.9350` with minimal latency overhead (+10.4 ms).

---

### 3. Milestone 6.4 — Cross-Encoder Reranking Benchmark
- **Retriever:** Hybrid RRF candidate retrieval ($k=20$) followed by SentenceTransformers `cross-encoder/ms-marco-MiniLM-L-6-v2` reranking ($k=5$).
- **Precision@5:** `0.1600`
- **Recall@5:** `0.8000`
- **MRR:** `0.6492`
- **Avg Latency:** `3227.91 ms` *(Median: 2519.69 ms)*
- **Key Observation:** While generic Cross-Encoder rerankers excel at general domain open-web QA, on highly specialized scientific research paper chunks (containing complex latex formulas, paper citations, and specialized terminology), the general-purpose Cross-Encoder model occasionally ranked noisy narrative chunks higher than concise formula chunks. Furthermore, neural pair scoring increased per-query latency by ~7x (from ~464 ms to ~3227 ms). Consequently, **Hybrid Retrieval (M6.3)** offers the highest accuracy and best latency profile for ScholarAI.

---

## 📁 Result Files Artifacts
- **M6.2 FAISS Baseline Results:** [data/evaluation/results/faiss_baseline.json](file:///c:/Users/archi/OneDrive/Desktop/ScholarAI/data/evaluation/results/faiss_baseline.json)
- **M6.3 Hybrid Benchmark Results:** [data/evaluation/results/hybrid_benchmark.json](file:///c:/Users/archi/OneDrive/Desktop/ScholarAI/data/evaluation/results/hybrid_benchmark.json)
- **M6.4 Reranking Benchmark Results:** [data/evaluation/results/reranking_benchmark.json](file:///c:/Users/archi/OneDrive/Desktop/ScholarAI/data/evaluation/results/reranking_benchmark.json)
