"""Unit tests for Hybrid Retrieval Benchmark module.

Tests hybrid retriever construction, RRF fusion, and benchmark runner
using fake embeddings to avoid requiring Gemini API calls.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, ".")

from langchain_core.embeddings import FakeEmbeddings  # noqa: E402

from src.evaluation.hybrid_benchmark import (  # noqa: E402
    build_hybrid_retriever,
    run_hybrid_benchmark,
)
from src.retrieval.hybrid_retriever import hybrid_search  # noqa: E402


class TestHybridBenchmark(unittest.TestCase):
    """Test Hybrid benchmark construction, RRF fusion, and benchmark runner."""

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

    def test_build_hybrid_retriever(self):
        vs, bm25 = build_hybrid_retriever(
            self.dummy_chunks, embeddings=self.fake_embeddings
        )
        results = hybrid_search(vs, bm25, "self-attention mechanism", top_k=2)

        self.assertLessEqual(len(results), 2)
        self.assertIn("chunk_id", results[0])
        self.assertIn("chunk_text", results[0])
        self.assertIn("retrieval_score", results[0])

    def test_run_hybrid_benchmark(self):
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

            with patch(
                "src.evaluation.hybrid_benchmark.load_all_evaluation_chunks",
                return_value=self.dummy_chunks,
            ), patch(
                "src.evaluation.hybrid_benchmark.evaluate_retriever",
                wraps=sys.modules[
                    "src.evaluation.hybrid_benchmark"
                ].evaluate_retriever,
            ) as mock_evaluator:
                res = run_hybrid_benchmark(
                    data_dir=temp_dir,
                    output_dir=temp_dir,
                    k=2,
                    embeddings=self.fake_embeddings,
                )

                self.assertTrue(mock_evaluator.called)

            self.assertEqual(res["num_queries"], 2)
            self.assertEqual(res["k"], 2)
            self.assertIn("precision@2", res["metrics"])
            self.assertIn("recall@2", res["metrics"])
            self.assertIn("mrr", res["metrics"])

            output_file = os.path.join(temp_dir, "hybrid_benchmark.json")
            self.assertTrue(os.path.exists(output_file))

            with open(output_file, "r", encoding="utf-8") as f:
                saved_res = json.load(f)

            self.assertEqual(saved_res["num_queries"], 2)


if __name__ == "__main__":
    unittest.main()
