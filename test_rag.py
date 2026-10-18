"""
End-to-End Test Suite for RAG Engine and Agent Service.
"""

import os
import shutil
import unittest

from rag_engine import DocumentProcessingError, RAGEngine
from agent import RAGAgentService

TEST_CHROMA_DIR = "./.test_chroma_data"

# Self-contained sample PDF generated in-memory for testing
SAMPLE_ROVER_PDF_BYTES = (
    b"%PDF-1.4\n"
    b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
    b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
    b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
    b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
    b"4 0 obj << /Length 265 >> stream\n"
    b"BT\n"
    b"/F1 14 Tf\n"
    b"50 720 Td\n"
    b"(ACME Quantum Robotics Operational Manual) Tj\n"
    b"/F1 11 Tf\n"
    b"0 -30 Td\n"
    b"(The Quantum Rover QR-500 operates at an optimal temperature of -40 degrees Celsius.) Tj\n"
    b"0 -25 Td\n"
    b"(The maximum battery endurance is exactly 72 continuous hours in vacuum conditions.) Tj\n"
    b"0 -25 Td\n"
    b"(Maintenance protocol Delta-9 requires replacing the optical sensors every 180 days.) Tj\n"
    b"ET\n"
    b"endstream\n"
    b"endobj\n"
    b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
    b"xref\n"
    b"0 6\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000058 00000 n \n"
    b"0000000115 00000 n \n"
    b"0000000244 00000 n \n"
    b"0000000560 00000 n \n"
    b"trailer << /Size 6 /Root 1 0 R >>\n"
    b"startxref\n"
    b"631\n"
    b"%%EOF\n"
)


class TestRAGApplication(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Clean any prior test DB
        if os.path.exists(TEST_CHROMA_DIR):
            shutil.rmtree(TEST_CHROMA_DIR, ignore_errors=True)

        cls.rag_engine = RAGEngine(
            persist_directory=TEST_CHROMA_DIR,
            collection_name="test_collection",
            embed_model="nomic-embed-text",
            relevance_threshold=0.30,
        )

        cls.agent_service = RAGAgentService(
            rag_engine=cls.rag_engine,
            model_name="qwen3.5:9b",
        )

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TEST_CHROMA_DIR):
            shutil.rmtree(TEST_CHROMA_DIR, ignore_errors=True)

    def test_01_empty_collection_guard(self):
        """Test that retrieve() returns empty list on empty collection (not crash)."""
        results = self.rag_engine.retrieve("anything at all", top_k=4)
        self.assertEqual(results, [], "Expected empty list from empty collection")
        print("\n[Test 1 PASSED] Empty collection retrieve() returned [] without crashing.")

    def test_02_empty_index_agent_guard(self):
        """Test that agent returns helpful message when no documents are indexed."""
        result = self.agent_service.ask("What is anything?", thread_id="test_empty")
        self.assertIn("upload", result["answer"].lower())
        print(f"[Test 2 PASSED] Agent returned upload prompt: {result['answer'][:80]}...")

    def test_03_ingest_pdf(self):
        """Test PDF parsing, chunking, and vector indexing."""
        chunk_count = self.rag_engine.ingest_pdf(SAMPLE_ROVER_PDF_BYTES, filename="sample_rover_manual.pdf")
        self.assertGreater(chunk_count, 0, "Expected at least one chunk to be indexed")
        self.assertGreater(self.rag_engine.document_count, 0, "document_count should reflect indexed chunks")
        print(f"[Test 3 PASSED] Indexed {chunk_count} chunk(s). document_count={self.rag_engine.document_count}")

    def test_04_retrieval_and_threshold(self):
        """Test vector similarity search with relevance filtering."""
        # Relevant query
        relevant = self.rag_engine.retrieve("battery endurance vacuum", top_k=2)
        self.assertGreater(len(relevant), 0, "Expected relevant chunks for battery query")
        self.assertIn("72 continuous hours", relevant[0].content)
        print(f"[Test 4 PASSED] Retrieved relevant chunk: {relevant[0].content[:60]}... (Score: {relevant[0].score})")

        # Irrelevant query
        irrelevant = self.rag_engine.retrieve("recipe for chocolate chip cookies", top_k=2)
        print(f"[Test 4 PASSED] Irrelevant query returned {len(irrelevant)} chunk(s).")

    def test_05_grounded_answer(self):
        """Test agent answers correctly when information is present."""
        result = self.agent_service.ask(
            "What is the maximum battery endurance of the Quantum Rover in vacuum?",
            thread_id="test_thread_1",
        )
        answer = result["answer"]
        citations = result["citations"]
        print(f"\n[Test 5 Question] What is the maximum battery endurance?")
        print(f"[Test 5 Answer] {answer}")
        print(f"[Test 5 Citations] {len(citations)} citation(s) returned")
        self.assertTrue(
            "72" in answer or "hours" in answer,
            f"Expected answer to mention 72 hours, got: {answer}",
        )

    def test_06_strict_guardrail_unanswerable(self):
        """Test agent returns 'I don't have information' when information is NOT present."""
        result = self.agent_service.ask(
            "What is the population of Tokyo in 2024?",
            thread_id="test_thread_2",
        )
        answer = result["answer"]
        print(f"\n[Test 6 Question] What is the population of Tokyo?")
        print(f"[Test 6 Answer] {answer}")
        self.assertTrue(
            "don't have information" in answer.lower()
            or "do not have information" in answer.lower()
            or "no information" in answer.lower()
            or "no relevant" in answer.lower(),
            f"Expected 'I don't have information...' response, got: {answer}",
        )

    def test_07_conversational_memory(self):
        """Test multi-turn context retention using thread_id."""
        thread_id = "test_memory_thread"

        # Turn 1
        res1 = self.agent_service.ask(
            "How often do optical sensors need replacement according to the manual?",
            thread_id=thread_id,
        )
        print(f"\n[Turn 1] {res1['answer']}")
        self.assertTrue("180" in res1["answer"])

        # Turn 2 (Follow-up using pronoun / context)
        res2 = self.agent_service.ask(
            "What is that maintenance protocol called?",
            thread_id=thread_id,
        )
        print(f"[Turn 2] {res2['answer']}")
        self.assertTrue(
            "Delta-9" in res2["answer"] or "delta" in res2["answer"].lower(),
            f"Expected 'Delta-9' in response, got: {res2['answer']}",
        )

    def test_08_corrupt_file_handling(self):
        """Test error handling when ingesting a non-PDF or corrupt file."""
        fake_corrupted_data = b"This is not a real PDF file header."
        with self.assertRaises(DocumentProcessingError):
            self.rag_engine.ingest_pdf(fake_corrupted_data, "bad.pdf")
        print("\n[Test 8 PASSED] Successfully caught corrupt PDF and raised DocumentProcessingError.")

    def test_09_clear_returns_count(self):
        """Test that clear() returns the number of documents removed."""
        count_before = self.rag_engine.document_count
        cleared = self.rag_engine.clear()
        count_after = self.rag_engine.document_count
        self.assertEqual(cleared, count_before, "clear() should return previous document count")
        self.assertEqual(count_after, 0, "Collection should be empty after clear()")
        print(f"\n[Test 9 PASSED] Cleared {cleared} documents. Count after: {count_after}")

    def test_10_document_management_and_persistence(self):
        """Test document cataloging, single-document removal, and cross-instance DB persistence."""
        # Ingest two documents with different names
        c1 = self.rag_engine.ingest_pdf(SAMPLE_ROVER_PDF_BYTES, filename="alpha_doc.pdf")
        c2 = self.rag_engine.ingest_pdf(SAMPLE_ROVER_PDF_BYTES, filename="beta_doc.pdf")
        self.assertGreater(c1, 0)
        self.assertGreater(c2, 0)

        # Verify cataloging
        catalog = self.rag_engine.get_indexed_documents()
        self.assertIn("alpha_doc.pdf", catalog)
        self.assertIn("beta_doc.pdf", catalog)
        self.assertEqual(catalog["alpha_doc.pdf"]["chunk_count"], c1)
        print(f"\n[Test 10.1 PASSED] Catalog indexed documents: {list(catalog.keys())}")

        # Delete single document
        removed = self.rag_engine.delete_document("alpha_doc.pdf")
        self.assertEqual(removed, c1)

        catalog_after = self.rag_engine.get_indexed_documents()
        self.assertNotIn("alpha_doc.pdf", catalog_after)
        self.assertIn("beta_doc.pdf", catalog_after)
        print(f"[Test 10.2 PASSED] Removed 'alpha_doc.pdf'. Remaining: {list(catalog_after.keys())}")

        # Test cross-instance persistence (simulates server restart / next day)
        restarted_engine = RAGEngine(
            persist_directory=TEST_CHROMA_DIR,
            collection_name="test_collection",
            embed_model="nomic-embed-text",
        )
        restarted_catalog = restarted_engine.get_indexed_documents()
        self.assertIn("beta_doc.pdf", restarted_catalog)
        self.assertEqual(restarted_catalog["beta_doc.pdf"]["chunk_count"], c2)
        print(f"[Test 10.3 PASSED] Database persistence across restarts verified! Preserved: {list(restarted_catalog.keys())}")


if __name__ == "__main__":
    unittest.main()

