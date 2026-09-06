"""Unit tests for Retrieval Metrics Dashboard data handling functions.

Tests loading JSON result files live from disk and formatting stage comparison metrics.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, ".")

from src.dashboard.metrics_dashboard import (  # noqa: E402
    get_stage_comparison_data,
    load_evaluation_results,
    load_single_result,
)


class TestMetricsDashboard(unittest.TestCase):
    """Test metrics dashboard data loading and data processing."""

    def test_load_single_result_valid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_result.json")
            sample_data = {"num_queries": 20, "metrics": {"precision@5": 0.2}}
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(sample_data, f)

            result = load_single_result(file_path)
            self.assertIsNotNone(result)
            self.assertEqual(result["num_queries"], 20)
            self.assertEqual(result["metrics"]["precision@5"], 0.2)

    def test_load_single_result_nonexistent(self):
        result = load_single_result("non_existent_directory/non_existent_file.json")
        self.assertIsNone(result)

    def test_load_single_result_invalid_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "corrupt.json")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("{invalid json content...")

            result = load_single_result(file_path)
            self.assertIsNone(result)

    def test_load_evaluation_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            files_to_create = {
                "faiss_baseline.json": {"metrics": {"precision@5": 0.2, "recall@5": 1.0, "mrr": 0.77, "avg_latency_ms": 500.0, "median_latency_ms": 480.0}},
                "hybrid_benchmark.json": {"metrics": {"precision@5": 0.2, "recall@5": 1.0, "mrr": 0.93, "avg_latency_ms": 490.0, "median_latency_ms": 470.0}},
                "reranking_benchmark.json": {"metrics": {"precision@5": 0.16, "recall@5": 0.8, "mrr": 0.65, "avg_latency_ms": 2900.0, "median_latency_ms": 2800.0}},
                "ablation_study.json": {"comparison_matrix": {}},
            }
            for name, content in files_to_create.items():
                with open(os.path.join(temp_dir, name), "w", encoding="utf-8") as f:
                    json.dump(content, f)

            results = load_evaluation_results(results_dir=temp_dir)
            self.assertIn("faiss_baseline", results)
            self.assertIn("hybrid_benchmark", results)
            self.assertIn("reranking_benchmark", results)
            self.assertIn("ablation_study", results)
            self.assertIsNotNone(results["faiss_baseline"])
            self.assertEqual(results["hybrid_benchmark"]["metrics"]["mrr"], 0.93)

    def test_get_stage_comparison_data(self):
        mock_results = {
            "faiss_baseline": {
                "metrics": {
                    "precision@5": 0.20,
                    "recall@5": 1.00,
                    "mrr": 0.7708,
                    "avg_latency_ms": 517.2,
                    "median_latency_ms": 510.0,
                }
            },
            "hybrid_benchmark": {
                "metrics": {
                    "precision@5": 0.20,
                    "recall@5": 1.00,
                    "mrr": 0.9350,
                    "avg_latency_ms": 493.1,
                    "median_latency_ms": 485.0,
                }
            },
            "reranking_benchmark": {
                "metrics": {
                    "precision@5": 0.16,
                    "recall@5": 0.80,
                    "mrr": 0.6492,
                    "avg_latency_ms": 2968.4,
                    "median_latency_ms": 2950.0,
                }
            },
        }

        stage_rows = get_stage_comparison_data(mock_results)
        self.assertEqual(len(stage_rows), 3)

        faiss_row = stage_rows[0]
        self.assertIn("FAISS Baseline", faiss_row["Stage"])
        self.assertEqual(faiss_row["Precision@5"], 0.20)
        self.assertEqual(faiss_row["Recall@5"], 1.00)
        self.assertEqual(faiss_row["MRR"], 0.7708)

        hybrid_row = stage_rows[1]
        self.assertIn("Hybrid Benchmark", hybrid_row["Stage"])
        self.assertEqual(hybrid_row["MRR"], 0.9350)

        rerank_row = stage_rows[2]
        self.assertIn("Reranking Benchmark", rerank_row["Stage"])
        self.assertEqual(rerank_row["Precision@5"], 0.16)

    def test_get_stage_comparison_data_missing_data(self):
        mock_results = {
            "faiss_baseline": None,
            "hybrid_benchmark": {},
            "reranking_benchmark": None,
        }
        stage_rows = get_stage_comparison_data(mock_results)
        self.assertEqual(len(stage_rows), 3)
        self.assertIsNone(stage_rows[0]["Precision@5"])
        self.assertIsNone(stage_rows[1]["MRR"])


if __name__ == "__main__":
    unittest.main()
