"""Unit tests for Hybrid Retrieval Benchmark module (M6.3).

Tests vector store + BM25 setup, hybrid retrieval via RRF, top-k, evaluator compatibility,
and JSON output writing using offline mock embeddings.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, ".")

from langchain_core.embeddings import FakeEmbeddings  # noqa: E402

from src.evaluation.hybrid_benchmark import run_hybrid_benchmark  # noqa: E402


class TestHybridBenchmark(unittest.TestCase):
    """Test hybrid benchmark execution and output format."""

    def setUp(self):
        self.fake_embeddings = FakeEmbeddings(size=64)
        self.dummy_chunks = [
            {
                "chunk_id": "attention_all_you_need:chunk_1",
                "document": "attention_all_you_need",
                "local_chunk_id": 1,
                "text": "Attention Is All You Need presented at NIPS 2017 conference.",
            },
            {
                "chunk_id": "attention_all_you_need:chunk_5",
                "document": "attention_all_you_need",
                "local_chunk_id": 5,
                "text": "Scaled Dot-Product Attention formula Softmax(QK^T / sqrt(d_k))V.",
            },
            {
                "chunk_id": "dense_passage_retrieval:chunk_5",
                "document": "dense_passage_retrieval",
                "local_chunk_id": 5,
                "text": "Dense passage retrieval uses dual-encoder dot product similarity function.",
            },
        ]

    def test_run_hybrid_benchmark(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = os.path.join(temp_dir, "evaluation_set.json")
            dummy_eval_data = {
                "queries": [
                    {
                        "query_id": "q1",
                        "query": "What is the formula for Scaled Dot-Product Attention?",
                        "expected_relevant_chunks": [
                            {"doc_id": "attention_all_you_need.pdf", "chunk_id": 5}
                        ],
                    },
                    {
                        "query_id": "q2",
                        "query": "Where was Attention Is All You Need presented?",
                        "expected_relevant_chunks": [
                            {"doc_id": "attention_all_you_need.pdf", "chunk_id": 1}
                        ],
                    },
                ]
            }
            with open(dataset_path, "w", encoding="utf-8") as f:
                json.dump(dummy_eval_data, f)

            with patch(
                "src.evaluation.hybrid_benchmark.load_all_evaluation_chunks",
                return_value=self.dummy_chunks,
            ):
                res = run_hybrid_benchmark(
                    data_dir=temp_dir,
                    output_dir=temp_dir,
                    k=2,
                    rrf_k=60,
                    embeddings=self.fake_embeddings,
                )

            # 1. Check structure
            self.assertEqual(res["num_queries"], 2)
            self.assertEqual(res["k"], 2)
            self.assertIn("precision@2", res["metrics"])
            self.assertIn("recall@2", res["metrics"])
            self.assertIn("mrr", res["metrics"])
            self.assertIn("avg_latency_ms", res["metrics"])
            self.assertIn("median_latency_ms", res["metrics"])
            self.assertEqual(len(res["per_query"]), 2)

            # 2. Check document-qualified chunk IDs in retrieved results
            for q_res in res["per_query"]:
                self.assertIn("id", q_res)
                self.assertIn("question", q_res)
                self.assertIn("retrieved_chunk_ids", q_res)
                self.assertIn("relevant_chunk_ids", q_res)
                self.assertEqual(len(q_res["retrieved_chunk_ids"]), 2)
                for cid in q_res["retrieved_chunk_ids"]:
                    self.assertIn(":chunk_", cid)

            # 3. Check JSON file output
            output_file = os.path.join(temp_dir, "hybrid_benchmark.json")
            self.assertTrue(os.path.exists(output_file))

            with open(output_file, "r", encoding="utf-8") as f:
                saved_res = json.load(f)

            self.assertEqual(saved_res["num_queries"], 2)
            self.assertEqual(saved_res["k"], 2)


if __name__ == "__main__":
    unittest.main()
