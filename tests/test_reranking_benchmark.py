"""Unit tests for Cross-Encoder Reranking Benchmark module.

Tests candidate retrieval, reranking pipeline, and benchmark runner
using fake embeddings and mocked reranking to avoid network/API calls.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, ".")

from langchain_core.embeddings import FakeEmbeddings  # noqa: E402

from src.evaluation.reranking_benchmark import (  # noqa: E402
    build_reranked_retriever,
    run_reranking_benchmark,
)
from src.retrieval.hybrid_retriever import hybrid_search  # noqa: E402


class TestRerankingBenchmark(unittest.TestCase):
    """Test Cross-Encoder reranking benchmark pipeline and runner."""

    def setUp(self):
        self.fake_embeddings = FakeEmbeddings(size=64)
        self.dummy_chunks = [
            {
                "chunk_id": "doc1:chunk_1",
                "document": "doc1",
                "local_chunk_id": 1,
                "text": "Transformer self-attention mechanism.",
            },
            {
                "chunk_id": "doc1:chunk_2",
                "document": "doc1",
                "local_chunk_id": 2,
                "text": "Multi-head attention projections.",
            },
            {
                "chunk_id": "doc2:chunk_1",
                "document": "doc2",
                "local_chunk_id": 1,
                "text": "Dense passage retrieval dual encoder.",
            },
        ]

    def test_build_reranked_retriever(self):
        vs, bm25 = build_reranked_retriever(
            self.dummy_chunks, embeddings=self.fake_embeddings
        )
        candidates = hybrid_search(vs, bm25, "self-attention", top_k=2)
        self.assertLessEqual(len(candidates), 2)

    def test_run_reranking_benchmark(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = os.path.join(temp_dir, "evaluation_set.json")
            dummy_eval_data = {
                "queries": [
                    {
                        "query_id": "q001",
                        "query": "What is self-attention?",
                        "expected_relevant_chunks": [
                            {"doc_id": "doc1.pdf", "chunk_id": 1}
                        ],
                    },
                    {
                        "query_id": "q002",
                        "query": "What architecture does DPR use?",
                        "expected_relevant_chunks": [
                            {"doc_id": "doc2.pdf", "chunk_id": 1}
                        ],
                    },
                ]
            }
            with open(dataset_path, "w", encoding="utf-8") as f:
                json.dump(dummy_eval_data, f)

            mock_reranked = [
                {
                    "chunk_id": "doc1:chunk_1",
                    "chunk_text": "Transformer self-attention.",
                    "reranker_score": 0.95,
                    "metadata": {},
                }
            ]

            target_mod = "src.evaluation.reranking_benchmark"

            with patch(
                f"{target_mod}.load_all_evaluation_chunks",
                return_value=self.dummy_chunks,
            ), patch(
                f"{target_mod}.rerank_results",
                return_value=mock_reranked,
            ), patch(
                f"{target_mod}.evaluate_retriever",
                wraps=sys.modules[target_mod].evaluate_retriever,
            ) as mock_evaluator:
                res = run_reranking_benchmark(
                    data_dir=temp_dir,
                    output_dir=temp_dir,
                    k=1,
                    candidate_k=2,
                    embeddings=self.fake_embeddings,
                )

                self.assertTrue(mock_evaluator.called)

            self.assertEqual(res["num_queries"], 2)
            self.assertEqual(res["k"], 1)
            self.assertIn("precision@1", res["metrics"])
            self.assertIn("recall@1", res["metrics"])
            self.assertIn("mrr", res["metrics"])

            output_file = os.path.join(temp_dir, "reranking_benchmark.json")
            self.assertTrue(os.path.exists(output_file))

            with open(output_file, "r", encoding="utf-8") as f:
                saved_res = json.load(f)

            self.assertEqual(saved_res["num_queries"], 2)


if __name__ == "__main__":
    unittest.main()
