"""Unit tests for ScholarAI evaluation framework and metrics.

Tests precision_at_k, recall_at_k, reciprocal_rank, mean_reciprocal_rank,
normalize_chunk_id, evaluate_retriever, and eval_corpus module without
requiring Gemini API calls.
"""

import json
import os
import unittest
from typing import Any, Dict, List

from src.evaluation.eval_corpus import (
    load_all_evaluation_chunks,
    load_evaluation_corpus,
)
from src.evaluation.evaluator import (
    evaluate_retriever,
    mean_reciprocal_rank,
    normalize_chunk_id,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class DummyDocument:
    """Mock LangChain Document for testing ID normalization."""

    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata


class TestIDNormalization(unittest.TestCase):
    """Test chunk ID normalization across various object representations."""

    def test_primitive_string_and_int(self):
        self.assertEqual(normalize_chunk_id("doc1:chunk_1"), "doc1:chunk_1")
        self.assertEqual(normalize_chunk_id(42), "42")

    def test_dict_chunk(self):
        self.assertEqual(normalize_chunk_id({"chunk_id": "c10"}), "c10")
        self.assertEqual(normalize_chunk_id({"id": "c20"}), "c20")
        doc_dict = {"metadata": {"chunk_id": "c30"}}
        self.assertEqual(normalize_chunk_id(doc_dict), "c30")

    def test_langchain_document(self):
        doc = DummyDocument("text content", {"chunk_id": "doc_chunk_5"})
        self.assertEqual(normalize_chunk_id(doc), "doc_chunk_5")


class TestIRMetrics(unittest.TestCase):
    """Test Information Retrieval metric calculations."""

    def test_precision_at_k_standard(self):
        retrieved = ["c1", "c2", "c3", "c4", "c5"]
        relevant = ["c2", "c5"]
        # Top 5: c2 and c5 are relevant -> 2/5 = 0.4
        self.assertAlmostEqual(precision_at_k(retrieved, relevant, k=5), 0.4)

    def test_precision_at_k_cutoff(self):
        retrieved = ["c1", "c2", "c3", "c4", "c5"]
        relevant = ["c1", "c2"]
        # k=2 -> c1, c2 -> 2/2 = 1.0
        self.assertAlmostEqual(precision_at_k(retrieved, relevant, k=2), 1.0)

    def test_recall_at_k_standard(self):
        retrieved = ["c1", "c2", "c3", "c4", "c5"]
        relevant = ["c2", "c5", "c6"]
        # Top 5 retrieved c2 and c5 -> 2 out of 3 relevant = 2/3
        self.assertAlmostEqual(
            recall_at_k(retrieved, relevant, k=5), 2.0 / 3.0
        )

    def test_reciprocal_rank(self):
        # Relevant item c3 is at 3rd retrieved position -> 1/3
        retrieved = ["c1", "c2", "c3", "c4"]
        relevant = ["c3", "c5"]
        self.assertAlmostEqual(reciprocal_rank(retrieved, relevant), 1.0 / 3.0)

    def test_mean_reciprocal_rank(self):
        scores = [1.0, 0.5, 0.0, 0.25]
        self.assertAlmostEqual(mean_reciprocal_rank(scores), 1.75 / 4.0)

    def test_empty_retrieved_results(self):
        self.assertEqual(precision_at_k([], ["c1"], k=5), 0.0)
        self.assertEqual(recall_at_k([], ["c1"], k=5), 0.0)
        self.assertEqual(reciprocal_rank([], ["c1"]), 0.0)

    def test_empty_relevant_set(self):
        self.assertEqual(precision_at_k(["c1", "c2"], [], k=5), 0.0)
        self.assertEqual(recall_at_k(["c1", "c2"], [], k=5), 0.0)
        self.assertEqual(reciprocal_rank(["c1", "c2"], []), 0.0)

    def test_fewer_than_k_retrieved_results(self):
        retrieved = ["c1", "c2"]
        relevant = ["c2"]
        # Top-k=5 with 2 items: 1 relevant / 5 = 0.2
        self.assertAlmostEqual(precision_at_k(retrieved, relevant, k=5), 0.2)
        # Recall: 1/1 = 1.0
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, k=5), 1.0)


class TestGenericRetrieverEvaluator(unittest.TestCase):
    """Test generic evaluate_retriever function and output shape."""

    def setUp(self):
        self.eval_set = [
            {
                "id": "q001",
                "question": "What is self-attention?",
                "relevant_chunk_ids": ["doc1:chunk_1", "doc1:chunk_2"],
            },
            {
                "id": "q002",
                "question": "What is DPR?",
                "relevant_chunk_ids": ["doc2:chunk_5"],
            },
        ]

    def test_evaluate_retriever(self):
        def dummy_retriever(
            query: str, top_k: int = 5
        ) -> List[Dict[str, Any]]:
            if "self-attention" in query:
                return [
                    {"chunk_id": "doc1:chunk_1"},
                    {"chunk_id": "doc1:chunk_99"},
                ]
            return [{"chunk_id": "doc2:chunk_5"}]

        res = evaluate_retriever(dummy_retriever, self.eval_set, k=5)

        self.assertEqual(res["num_queries"], 2)
        self.assertEqual(res["k"], 5)
        self.assertIn("precision@5", res["metrics"])
        self.assertIn("recall@5", res["metrics"])
        self.assertIn("mrr", res["metrics"])
        self.assertIn("avg_latency_ms", res["metrics"])
        self.assertIn("median_latency_ms", res["metrics"])

        self.assertEqual(len(res["per_query"]), 2)
        q1_res = res["per_query"][0]
        self.assertIn("latency_ms", q1_res)
        self.assertGreaterEqual(q1_res["latency_ms"], 0.0)
        self.assertEqual(
            q1_res["retrieved_chunk_ids"], ["doc1:chunk_1", "doc1:chunk_99"]
        )


class TestEvaluationCorpusAndDatasetIntegrity(unittest.TestCase):
    """Test creation and schema integrity of evaluation corpus and dataset."""

    def test_eval_corpus_loader(self):
        data_dir = "data/evaluation"
        if not os.path.exists(data_dir):
            self.skipTest(f"{data_dir} does not exist")

        corpus = load_evaluation_corpus(data_dir=data_dir)
        self.assertIn("attention_all_you_need", corpus)
        self.assertIn("dense_passage_retrieval", corpus)
        self.assertIn("retrieval_augmented_generation", corpus)

        all_chunks = load_all_evaluation_chunks(data_dir=data_dir)
        self.assertEqual(
            len(all_chunks),
            len(corpus["attention_all_you_need"])
            + len(corpus["dense_passage_retrieval"])
            + len(corpus["retrieval_augmented_generation"]),
        )

    def test_eval_dataset_json_grounding(self):
        dataset_path = "data/evaluation/eval_dataset.json"
        if not os.path.exists(dataset_path):
            self.skipTest(f"{dataset_path} does not exist")

        with open(dataset_path, "r", encoding="utf-8") as f:
            eval_set = json.load(f)

        self.assertGreaterEqual(len(eval_set), 20)
        self.assertLessEqual(len(eval_set), 30)

        corpus_chunks = load_all_evaluation_chunks("data/evaluation")
        valid_chunk_ids = {c["chunk_id"] for c in corpus_chunks}

        for q in eval_set:
            self.assertIn("id", q)
            self.assertIn("question", q)
            self.assertIn("source_document", q)
            self.assertIn("relevant_chunk_ids", q)
            self.assertGreater(len(q["relevant_chunk_ids"]), 0)

            for cid in q["relevant_chunk_ids"]:
                self.assertIn(
                    cid,
                    valid_chunk_ids,
                    f"Question {q['id']} references invalid chunk ID: {cid}",
                )


if __name__ == "__main__":
    unittest.main()
