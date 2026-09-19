"""Unit tests for FAISS baseline benchmark module.

Tests vector store construction and benchmark runner using fake embeddings
to avoid requiring Gemini API calls.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, ".")

from langchain_core.embeddings import FakeEmbeddings  # noqa: E402

from src.evaluation.baseline_faiss import (  # noqa: E402
    build_eval_vector_store,
    load_eval_set,
    run_faiss_baseline,
)


class TestFAISSBaseline(unittest.TestCase):
    """Test FAISS baseline vector store creation and evaluation runner."""

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

    def test_build_eval_vector_store(self):
        vs = build_eval_vector_store(
            self.dummy_chunks, embeddings=self.fake_embeddings
        )
        results = vs.similarity_search("self-attention", k=2)

        self.assertEqual(len(results), 2)
        self.assertIn("chunk_id", results[0].metadata)
        self.assertIn("document", results[0].metadata)

    def test_load_eval_set_format_handling(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = os.path.join(temp_dir, "evaluation_set.json")
            data_content = {
                "dataset_metadata": {"version": "1.0.0"},
                "queries": [
                    {
                        "query_id": "q1",
                        "query": "What is self-attention?",
                        "expected_relevant_chunks": [
                            {"doc_id": "doc1.pdf", "chunk_id": 1}
                        ],
                    }
                ],
            }
            with open(dataset_path, "w", encoding="utf-8") as f:
                json.dump(data_content, f)

            eval_set = load_eval_set(temp_dir)
            self.assertEqual(len(eval_set), 1)
            self.assertEqual(eval_set[0]["id"], "q1")
            self.assertEqual(
                eval_set[0]["relevant_chunk_ids"], ["doc1:chunk_1"]
            )

    def test_run_faiss_baseline(self):
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
                "src.evaluation.baseline_faiss.load_all_evaluation_chunks",
                return_value=self.dummy_chunks,
            ), patch(
                "src.evaluation.baseline_faiss.evaluate_retriever",
                wraps=sys.modules[
                    "src.evaluation.baseline_faiss"
                ].evaluate_retriever,
            ) as mock_evaluator:
                res = run_faiss_baseline(
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

            output_file = os.path.join(temp_dir, "faiss_baseline.json")
            self.assertTrue(os.path.exists(output_file))

            with open(output_file, "r", encoding="utf-8") as f:
                saved_res = json.load(f)

            self.assertEqual(saved_res["num_queries"], 2)


if __name__ == "__main__":
    unittest.main()
