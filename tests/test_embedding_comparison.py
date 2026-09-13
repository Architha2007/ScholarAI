"""Unit test suite for Milestone 8 Embedding Model Comparison.

Tests model configuration, loading, embedding dimension extraction,
result structure, JSON file generation, and metric aggregation, using
mocked calls to avoid model downloads during automated testing.
"""

import json
import os
import sys
import tempfile
import unittest
from typing import List
from unittest.mock import MagicMock, patch

# Add project root directory to sys.path to allow imports from src/
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

from langchain_core.embeddings import Embeddings  # noqa: E402

from src.evaluation.embedding_comparison import (  # noqa: E402
    MODELS_TO_COMPARE,
    SentenceTransformerEmbeddings,
    evaluate_single_embedding_model,
    instantiate_embedding_model,
    run_embedding_comparison,
)


class MockDeterministicEmbeddings(Embeddings):
    """Deterministic fake embeddings model for fast offline testing."""

    def __init__(self, dimension: int = 64):
        """Initializes with specified vector dimension.

        Args:
            dimension: Float vector length to return.
        """
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds document texts deterministically."""
        return [[0.1] * self.dimension for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embeds query text deterministically."""
        return [0.1] * self.dimension


class TestEmbeddingComparison(unittest.TestCase):
    """Test suite for embedding model comparison module."""

    def setUp(self) -> None:
        """Sets up mock evaluation corpus and dataset."""
        self.fake_chunks = [
            {
                "chunk_id": "doc1:chunk_1",
                "document": "doc1",
                "local_chunk_id": 1,
                "text": "Transformer self-attention mechanism paper.",
            },
            {
                "chunk_id": "doc1:chunk_2",
                "document": "doc1",
                "local_chunk_id": 2,
                "text": "Dense passage retrieval for open domain QA.",
            },
        ]
        self.fake_eval_set = [
            {
                "id": "q001",
                "question": "What is self-attention?",
                "relevant_chunk_ids": ["doc1:chunk_1"],
            },
            {
                "id": "q002",
                "question": "What is dense passage retrieval?",
                "relevant_chunk_ids": ["doc1:chunk_2"],
            },
        ]
        self.fake_embeddings = MockDeterministicEmbeddings(dimension=128)

    @patch("src.evaluation.embedding_comparison.SentenceTransformer")
    def test_sentence_transformer_embeddings_wrapper(
        self, mock_st_class: MagicMock
    ) -> None:
        """Verifies SentenceTransformerEmbeddings wrapper methods."""
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(
            tolist=lambda: [[0.2, 0.4, 0.6]]
        )
        mock_st_class.return_value = mock_model

        wrapper = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
        self.assertEqual(wrapper.model_name, "all-MiniLM-L6-v2")

        doc_vecs = wrapper.embed_documents(["sample document"])
        self.assertEqual(doc_vecs, [[0.2, 0.4, 0.6]])

        mock_model.encode.return_value = [MagicMock(tolist=lambda: [0.2, 0.4])]
        q_vec = wrapper.embed_query("sample query")
        self.assertEqual(q_vec, [0.2, 0.4])

    @patch("src.evaluation.embedding_comparison.get_eval_embeddings")
    @patch(
        "src.evaluation.embedding_comparison.SentenceTransformerEmbeddings"
    )
    def test_instantiate_embedding_model(
        self, mock_st_wrapper: MagicMock, mock_get_gemini: MagicMock
    ) -> None:
        """Verifies model instantiation based on config type."""
        gemini_cfg = {
            "model_id": "models/gemini-embedding-001",
            "type": "gemini",
        }
        instantiate_embedding_model(gemini_cfg)
        self.assertTrue(mock_get_gemini.called)

        st_cfg = {
            "model_id": "sentence-transformers/all-MiniLM-L6-v2",
            "type": "sentence_transformer",
        }
        instantiate_embedding_model(st_cfg)
        mock_st_wrapper.assert_called_with(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def test_evaluate_single_embedding_model_structure(self) -> None:
        """Verifies result structure and metrics for a single model run."""
        cfg = {
            "model_id": "mock-model-v1",
            "display_name": "Mock Model V1",
            "type": "sentence_transformer",
        }
        res = evaluate_single_embedding_model(
            model_config=cfg,
            eval_chunks=self.fake_chunks,
            eval_set=self.fake_eval_set,
            k=2,
            embeddings_instance=self.fake_embeddings,
        )

        self.assertEqual(res["model_id"], "mock-model-v1")
        self.assertEqual(res["display_name"], "Mock Model V1")
        self.assertEqual(res["embedding_dimension"], 128)
        self.assertIn("embedding_generation_latency_ms", res)
        self.assertIn("metrics", res)

        metrics = res["metrics"]
        self.assertIn("precision@2", metrics)
        self.assertIn("recall@2", metrics)
        self.assertIn("mrr", metrics)
        self.assertIn("avg_latency_ms", metrics)
        self.assertEqual(len(res["per_query"]), 2)

    @patch("src.evaluation.embedding_comparison.load_all_evaluation_chunks")
    @patch("src.evaluation.embedding_comparison.load_eval_set")
    def test_run_embedding_comparison_json_and_aggregation(
        self, mock_load_set: MagicMock, mock_load_chunks: MagicMock
    ) -> None:
        """Verifies comparison run, matrix, and JSON output."""
        mock_load_chunks.return_value = self.fake_chunks
        mock_load_set.return_value = self.fake_eval_set

        custom_cfgs = [
            {
                "model_id": "test-model-1",
                "display_name": "Test Model 1",
                "type": "test",
            },
            {
                "model_id": "test-model-2",
                "display_name": "Test Model 2",
                "type": "test",
            },
        ]
        custom_embeddings_map = {
            "test-model-1": MockDeterministicEmbeddings(dimension=64),
            "test-model-2": MockDeterministicEmbeddings(dimension=256),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            res = run_embedding_comparison(
                data_dir=temp_dir,
                output_dir=temp_dir,
                k=2,
                models_config=custom_cfgs,
                custom_embeddings_map=custom_embeddings_map,
            )

            self.assertEqual(res["num_queries"], 2)
            self.assertEqual(res["num_chunks"], 2)
            self.assertEqual(res["k"], 2)
            self.assertIn("timestamp", res)
            self.assertIn("models", res)
            self.assertIn("summary_matrix", res)

            # Check JSON file written
            json_file = os.path.join(temp_dir, "embedding_comparison.json")
            self.assertTrue(os.path.exists(json_file))

            with open(json_file, "r", encoding="utf-8") as f:
                saved_data = json.load(f)

            self.assertEqual(len(saved_data["summary_matrix"]), 2)
            first_row = saved_data["summary_matrix"][0]
            self.assertEqual(first_row["model_id"], "test-model-1")
            self.assertEqual(first_row["embedding_dimension"], 64)
            self.assertIn("precision@2", first_row)
            self.assertIn("recall@2", first_row)
            self.assertIn("mrr", first_row)

    def test_default_models_to_compare_list(self) -> None:
        """Verifies default MODELS_TO_COMPARE contains candidate models."""
        model_ids = [m["model_id"] for m in MODELS_TO_COMPARE]
        self.assertIn("models/gemini-embedding-001", model_ids)
        self.assertIn("sentence-transformers/all-MiniLM-L6-v2", model_ids)
        self.assertIn("sentence-transformers/all-mpnet-base-v2", model_ids)
        self.assertIn("BAAI/bge-base-en-v1.5", model_ids)


if __name__ == "__main__":
    unittest.main()
