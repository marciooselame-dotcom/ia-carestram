import unittest
import os
import sqlite3
from src.core.rag_engine import RAGEngine

class TestRAGEngine(unittest.TestCase):
    """Unit test suite for validating the RAG engine, text parsing, and search fallbacks."""

    def setUp(self) -> None:
        """Initializes the RAGEngine with an in-memory database to isolate data."""
        self.temp_faiss_path = "test_temp_index.index"
        self.rag = RAGEngine(db_path=":memory:", faiss_path=self.temp_faiss_path)

    def tearDown(self) -> None:
        """Closes connections and cleans up temporary files."""
        self.rag.close()
        if os.path.exists(self.temp_faiss_path):
            try:
                os.remove(self.temp_faiss_path)
            except Exception:
                pass

    def test_identify_joint_type(self) -> None:
        """Tests that joint type is accurately parsed from text variations."""
        text_knee = "Exame de RM do joelho esquerdo. Menisco sem lesão."
        text_shoulder = "Laudo de RM de ombro direito."
        text_unknown = "Exame de RM de coluna lombar."
        
        self.assertEqual(self.rag.identify_joint_type(text_knee), "joelho")
        self.assertEqual(self.rag.identify_joint_type(text_shoulder), "ombro")
        self.assertEqual(self.rag.identify_joint_type(text_unknown), "desconhecido")

    def test_split_into_chunks(self) -> None:
        """Tests text chunk segmentation and filtering of tiny text residues."""
        text = "Este é um laudo longo para testar a função de chunking do sistema de RAG do copilot."
        # Using tiny chunk sizes for testing
        chunks = self.rag.split_into_chunks(text, chunk_size=25, overlap=5)
        self.assertTrue(len(chunks) > 1)
        # Verify overlap and content is kept
        self.assertIn("Este é um", chunks[0])

    def test_split_findings_and_conclusion(self) -> None:
        """Tests dividing findings and conclusions by matching anatomical boundary terms."""
        text = "Achados: Condropatia grau II. Conclusão: 1. Condropatia patelar."
        findings, conclusion = self.rag._split_findings_and_conclusion(text)
        self.assertEqual(findings, "Achados: Condropatia grau II.")
        self.assertEqual(conclusion, "Conclusão: 1. Condropatia patelar.")

    def test_tfidf_fallback_search(self) -> None:
        """Tests keyword matching and retrieval via TF-IDF search fallback."""
        cursor = self.rag.conn.cursor()
        
        # Insert mock documents directly into the database
        reports_data = [
            ("path1.txt", "hash1", "RM de Joelho. Rotura de menisco medial.", "joelho"),
            ("path2.txt", "hash2", "RM de Joelho. Normal sem alterações.", "joelho"),
            ("path3.txt", "hash3", "RM de Ombro. Rotura do supraespinhal.", "ombro")
        ]
        
        for file_path, file_hash, content, joint in reports_data:
            cursor.execute("""
                INSERT INTO reports (file_path, file_hash, raw_content, anonymized_content, joint_type, findings, conclusion)
                VALUES (?, ?, ?, ?, ?, '', '')
            """, (file_path, file_hash, content, content, joint))
        self.rag.conn.commit()

        # Run search query on 'joelho' for 'rotura' keyword
        results = self.rag._tfidf_fallback_search(query="rotura menisco", joint_type="joelho", k=1)
        
        self.assertIn("Rotura de menisco medial", results)
        self.assertNotIn("supraespinhal", results) # Should filter by joint type

if __name__ == "__main__":
    unittest.main()
