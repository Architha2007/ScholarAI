"""Deterministic validation tests for ScholarAI retrieval evaluation dataset.

Ensures schema correctness, query uniqueness, document existence,
and text snippet alignment against the source corpus PDFs.
"""

import json
import os
import unittest
import pypdf

# Constants for evaluation paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_DIR = os.path.join(PROJECT_ROOT, "data", "evaluation")
DATASET_PATH = os.path.join(EVAL_DIR, "evaluation_set.json")


class TestEvaluationDataset(unittest.TestCase):
    """Integrity checks for dataset JSON and corpus documents."""

    @classmethod
    def setUpClass(cls):
        """Pre-load and cache corpus PDF text to speed up verification."""
        cls.pdf_texts = {}
        dataset_exists = os.path.exists(DATASET_PATH)
        if not dataset_exists:
            return

        try:
            with open(DATASET_PATH, "r", encoding="utf-8") as f:
                cls.dataset = json.load(f)
        except Exception:
            cls.dataset = None
            return

        # Pre-extract text from any corpus files listed in metadata
        corpus_files = (
            cls.dataset.get("dataset_metadata", {}).get("corpus_files", [])
        )
        for doc in corpus_files:
            doc_id = doc.get("doc_id")
            if not doc_id:
                continue

            pdf_path = os.path.join(EVAL_DIR, doc_id)
            if os.path.exists(pdf_path):
                try:
                    reader = pypdf.PdfReader(pdf_path)
                    text_parts = []
                    for page in reader.pages:
                        text_parts.append(page.extract_text() or "")
                    cls.pdf_texts[doc_id] = "\n\n".join(text_parts)
                except Exception as e:
                    print(f"Error loading PDF {doc_id}: {e}")

    def test_dataset_file_exists(self):
        """1. Validate that the dataset JSON file exists."""
        self.assertTrue(
            os.path.exists(DATASET_PATH),
            f"Dataset not found at {DATASET_PATH}"
        )

    def test_json_is_valid(self):
        """2. Validate that the dataset JSON file is well-formed."""
        try:
            with open(DATASET_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIsNotNone(data)
        except Exception as e:
            self.fail(f"Dataset JSON is invalid: {e}")

    def test_required_top_level_fields_exist(self):
        """3. Validate top-level dataset structure."""
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("dataset_metadata", data)
        self.assertIn("queries", data)

        metadata = data["dataset_metadata"]
        self.assertIn("version", metadata)
        self.assertIn("creation_date", metadata)
        self.assertIn("chunk_size", metadata)
        self.assertIn("chunk_overlap", metadata)
        self.assertIn("corpus_files", metadata)

    def test_query_structure_and_non_empty(self):
        """4 & 5. Verify query contains required fields and target chunk."""
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        queries = data.get("queries", [])
        for idx, q in enumerate(queries):
            prefix = f"Query index {idx}"
            self.assertIn("query_id", q, f"{prefix} is missing query_id")
            self.assertIn("query", q, f"{prefix} is missing query text")
            self.assertIn(
                "expected_relevant_chunks",
                q,
                f"{prefix} is missing expected_relevant_chunks"
            )

            relevant_chunks = q.get("expected_relevant_chunks", [])
            qid = q.get('query_id')
            self.assertGreaterEqual(
                len(relevant_chunks),
                1,
                f"Query {qid} must have at least one relevant target."
            )

    def test_relevant_chunk_fields(self):
        """6. Every relevant chunk has doc_id and text_snippet."""
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        queries = data.get("queries", [])
        for q in queries:
            qid = q.get("query_id")
            chunks = q.get("expected_relevant_chunks", [])
            for c_idx, c in enumerate(chunks):
                prefix = f"Query {qid}, chunk {c_idx}"
                self.assertIn("doc_id", c, f"{prefix} is missing doc_id")
                self.assertIn(
                    "text_snippet",
                    c,
                    f"{prefix} is missing text_snippet"
                )

    def test_corpus_files_exist(self):
        """7. Referenced PDF files exist under data/evaluation/."""
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        corpus_files = data.get("dataset_metadata", {}).get("corpus_files", [])
        for doc in corpus_files:
            doc_id = doc.get("doc_id")
            pdf_path = os.path.join(EVAL_DIR, doc_id)
            self.assertTrue(
                os.path.exists(pdf_path),
                f"Referenced PDF file {doc_id} not found at {pdf_path}"
            )

    def test_text_snippets_present_in_pdfs(self):
        """8. Ground-truth text snippets are present in the PDF text."""
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        queries = data.get("queries", [])
        for q in queries:
            qid = q.get("query_id")
            chunks = q.get("expected_relevant_chunks", [])
            for c in chunks:
                doc_id = c.get("doc_id")
                snippet = c.get("text_snippet")

                self.assertIn(
                    doc_id,
                    self.pdf_texts,
                    f"PDF {doc_id} was not loaded or does not exist."
                )

                pdf_text = self.pdf_texts[doc_id]
                self.assertIn(
                    snippet,
                    pdf_text,
                    f"Snippet for query {qid} not found in PDF {doc_id}. "
                    f"Snippet: {repr(snippet)}"
                )

    def test_query_ids_unique(self):
        """9. Query IDs are unique across the dataset."""
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        queries = data.get("queries", [])
        query_ids = [q.get("query_id") for q in queries if q.get("query_id")]
        self.assertEqual(
            len(query_ids),
            len(set(query_ids)),
            f"Duplicate query IDs found: {query_ids}"
        )

    def test_no_duplicate_queries(self):
        """10. No duplicate queries exist in the dataset."""
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        queries = data.get("queries", [])
        query_texts = [
            q.get("query").strip().lower()
            for q in queries
            if q.get("query")
        ]
        self.assertEqual(
            len(query_texts),
            len(set(query_texts)),
            "Duplicate query texts found in the dataset."
        )

    def test_dataset_size_at_least_15(self):
        """11. Dataset contains at least 15 queries."""
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        queries = data.get("queries", [])
        self.assertGreaterEqual(
            len(queries),
            15,
            f"Dataset has {len(queries)} queries, but requires at least 15."
        )

    def test_dataset_covers_all_corpus_documents(self):
        """12. Dataset contains all selected corpus documents."""
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        corpus_files = data.get("dataset_metadata", {}).get("corpus_files", [])
        expected_docs = {
            doc.get("doc_id") for doc in corpus_files if doc.get("doc_id")
        }

        queries = data.get("queries", [])
        referenced_docs = set()
        for q in queries:
            for c in q.get("expected_relevant_chunks", []):
                if c.get("doc_id"):
                    referenced_docs.add(c.get("doc_id"))

        self.assertEqual(
            expected_docs,
            referenced_docs,
            f"Referenced docs {referenced_docs} do not match "
            f"corpus docs {expected_docs}"
        )


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RUNNING SCHOLARAI RETRIEVAL EVALUATION DATASET VALIDATION TESTS")
    print("=" * 80)

    # Run the tests programmatically
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEvaluationDataset)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total Tests Run: {result.testsRun}")
    print(f"Passed: {result.wasSuccessful()}")
    print(f"Errors: {len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print("=" * 80)

    # Propagate exit status
    import sys
    sys.exit(not result.wasSuccessful())
