# ScholarAI Retrieval Benchmark Experiments

This document records the evaluation benchmark experiments and controlled ablation studies conducted for the ScholarAI Retrieval System across standard research paper PDFs (`attention_all_you_need`, `dense_passage_retrieval`, `retrieval_augmented_generation`) evaluated against the 20-query ground-truth dataset (`data/evaluation/evaluation_set.json`).

---

## 📊 Consolidated M6.2–M6.5 Retrieval Benchmark & Ablation Study

| Strategy / Retriever Pipeline | Precision@5 | Recall@5 | MRR | Avg Latency (ms) | Median Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. FAISS-only (Dense Vector)** | `0.2000` | **`1.0000`** | `0.7708` | `517.15` | `479.84` |
| **2. BM25-only (Sparse Keyword)** | `0.1900` | `0.9500` | **`0.9500`** | **`0.43`** | **`0.38`** |
| **3. Hybrid RRF (FAISS + BM25)** | **`0.2000`** | **`1.0000`** | **`0.9350`** | `492.99` | `478.10` |
| **4. Hybrid + Cross-Encoder Reranked** | `0.1600` | `0.8000` | `0.6492` | `2968.43` | `2449.30` |

---

## 🔬 Component Contribution Analysis (Retrieval Quality vs Latency)

### 1. BM25 Sparse Keyword Retrieval — Highest Impact on MRR & Speed
- **MRR Impact:** Boosts Mean Reciprocal Rank from `0.7708` to `0.9500` when evaluated standalone. Exact lexical matching excels at placing exact formula tokens (e.g. `Attention(Q, K, V)`), author lists, and acronyms (`ORQA`, `NIPS 2017`) directly at **Rank 1**.
- **Latency Impact:** Sub-millisecond execution (`0.43 ms` avg). BM25 provides maximum ranking speed with zero API overhead.
- **Limitation:** Fails on semantic paraphrasing where exact query keywords are absent from the chunk text, dropping Recall@5 to `0.9500`.

### 2. FAISS Dense Vector Retrieval — Highest Impact on Recall
- **Recall Impact:** Achieves **1.0000 Recall@5**, ensuring that 100% of ground-truth relevant chunks are captured within the top-5 retrieved results even when queries use alternative phrasing.
- **MRR Impact:** `0.7708`. Dense embeddings capture overall semantic context but occasionally rank surrounding narrative chunks slightly above exact formula snippets.
- **Latency Impact:** Requires embedding generation (~`450–500 ms`).

### 3. Hybrid RRF (FAISS + BM25) — Optimal Balanced Strategy for ScholarAI
- **Performance:** Combines the **1.0000 Recall@5** of FAISS with the high **0.9350 MRR** of BM25.
- **Tradeoff Analysis:** By fusing sparse and dense rankings via Reciprocal Rank Fusion ($k=60$), Hybrid RRF ensures zero semantic recall loss while pushing exact formula/citation matches to Rank 1.
- **Latency:** ~`493 ms` total per query.

### 4. Cross-Encoder Reranking — Heavy Latency Overhead with Lower Formula Precision
- **Performance:** Precision@5 drops to `0.1600`, Recall@5 to `0.8000`, and MRR to `0.6492`.
- **Latency Impact:** Increases per-query latency by **~6x** (up to `2968 ms` avg).
- **Tradeoff Analysis:** Pretrained open-web Cross-Encoder models (`ms-marco-MiniLM-L-6-v2`) prioritize generic conversational relevance over specialized scientific latex syntax, making them unsuited for raw technical paper reranking compared to Hybrid RRF.

---

## 📁 Result File Artifacts
- **M6.2 FAISS Baseline Results:** [data/evaluation/results/faiss_baseline.json](file:///c:/Users/archi/OneDrive/Desktop/ScholarAI/data/evaluation/results/faiss_baseline.json)
- **M6.3 Hybrid Benchmark Results:** [data/evaluation/results/hybrid_benchmark.json](file:///c:/Users/archi/OneDrive/Desktop/ScholarAI/data/evaluation/results/hybrid_benchmark.json)
- **M6.4 Reranking Benchmark Results:** [data/evaluation/results/reranking_benchmark.json](file:///c:/Users/archi/OneDrive/Desktop/ScholarAI/data/evaluation/results/reranking_benchmark.json)
- **M6.5 Ablation Study Results:** [data/evaluation/results/ablation_study.json](file:///c:/Users/archi/OneDrive/Desktop/ScholarAI/data/evaluation/results/ablation_study.json)
