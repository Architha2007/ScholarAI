# ScholarAI Evaluation Corpus

This directory contains the ground-truth evaluation corpus used to assess and rank the retrieval quality of the ScholarAI Hybrid Retrieval system.

## 📄 Selected Papers

### 1. Attention Is All You Need
*   **Filename:** `attention_all_you_need.pdf`
*   **Source:** [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
*   **Selection Rationale:** This paper introduces the Transformer architecture, self-attention mechanisms, and multi-head attention. It is highly technical, dense in specific mathematical concepts, and features distinct keywords (e.g., "Transformer", "Scaled Dot-Product Attention", "Multi-Head Attention"). It is excellent for testing precise keyword matching (BM25) as well as complex conceptual understanding (FAISS dense retrieval).

### 2. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
*   **Filename:** `retrieval_augmented_generation.pdf`
*   **Source:** [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
*   **Selection Rationale:** This is the seminal paper that introduced RAG. It combines parametric memory (pre-trained seq2seq models) and non-parametric memory (dense vector retrieval). It is perfect for evaluating information retrieval terminology, dense retriever architectures, and comparing keyword-heavy descriptions of generative tasks against semantic search queries.

### 3. Dense Passage Retrieval for Open-Domain Question Answering
*   **Filename:** `dense_passage_retrieval.pdf`
*   **Source:** [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)
*   **Selection Rationale:** This paper describes Dense Passage Retrieval (DPR), highlighting how dense embeddings outperform traditional TF-IDF or BM25 retrieval. It contains rich terminology comparing lexical models (BM25) and dense models (DPR), mapping perfectly to evaluating scenarios where FAISS dense retrieval outperforms BM25, and vice versa.
