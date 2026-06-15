import os
import sqlite3
import hashlib
import logging
import math
import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import requests
from dotenv import load_dotenv

# Try importing FAISS and document parsers.
# If they fail, we log and provide degradation fallbacks.
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from pypdf import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    from striprtf.striprtf import rtf_to_text
    RTF_AVAILABLE = True
except ImportError:
    RTF_AVAILABLE = False

from src.core.anonymizer import Anonymizer

load_dotenv()
logger = logging.getLogger(__name__)

class RAGEngine:
    """A local Retrieval-Augmented Generation (RAG) system for medical reports.
    
    Responsible for scanning a local directory of reports, extracting text from multiple
    formats, de-identifying data, vectorizing using Ollama, indexing using FAISS,
    and retrieving relevant context. If FAISS or Ollama embeddings are unavailable,
    it falls back to a custom pure-Python TF-IDF similarity search.
    """

    def __init__(self, db_path: Optional[str] = None, faiss_path: Optional[str] = None) -> None:
        """Initializes RAGEngine and connects to SQLite database.

        Args:
            db_path (Optional[str]): Path to SQLite database.
            faiss_path (Optional[str]): Path to FAISS index file.
        """
        self.db_path = db_path or os.getenv("DB_PATH", "copilot_data.db")
        self.faiss_path = faiss_path or os.getenv("FAISS_INDEX_PATH", "copilot_index.index")
        self.anonymizer = Anonymizer()
        
        # Ollama connection settings
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        self.default_model = os.getenv("DEFAULT_MODEL", "gemma3:4b")
        self.fallback_model = os.getenv("FALLBACK_MODEL", "qwen2.5:0.5b")

        # Database and Vector Index state
        self.conn: sqlite3.Connection = sqlite3.connect(self.db_path)
        self.faiss_index: Optional[Any] = None
        self.embedding_dimension: int = 0
        
        self._init_database()
        self._load_vector_index()

    def _init_database(self) -> None:
        """Creates database tables if they do not exist."""
        try:
            cursor = self.conn.cursor()
            # Reports table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE,
                    file_hash TEXT,
                    raw_content TEXT,
                    anonymized_content TEXT,
                    joint_type TEXT,
                    findings TEXT,
                    conclusion TEXT,
                    imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Report chunks for vector search
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS report_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id INTEGER,
                    chunk_text TEXT,
                    vector_index INTEGER,
                    FOREIGN KEY(report_id) REFERENCES reports(id) ON DELETE CASCADE
                )
            """)
            self.conn.commit()
            logger.info("Database initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize database: %s", str(e), exc_info=True)

    def _load_vector_index(self) -> None:
        """Loads FAISS index from disk or remains empty for lazy init."""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS is not installed. Running RAG in TF-IDF fallback mode.")
            return

        try:
            if os.path.exists(self.faiss_path):
                self.faiss_index = faiss.read_index(self.faiss_path)
                self.embedding_dimension = self.faiss_index.d
                logger.info("Loaded FAISS index from '%s' with dimension %d.", self.faiss_path, self.embedding_dimension)
        except Exception as e:
            logger.error("Failed to load FAISS index. Creating a new one. Error: %s", str(e), exc_info=True)

    def _save_vector_index(self) -> None:
        """Saves the active FAISS index to disk."""
        if not FAISS_AVAILABLE or self.faiss_index is None:
            return
        try:
            faiss.write_index(self.faiss_index, self.faiss_path)
            logger.info("Saved FAISS index to '%s'.", self.faiss_path)
        except Exception as e:
            logger.error("Failed to save FAISS index: %s", str(e), exc_info=True)

    def compute_file_hash(self, file_path: str) -> str:
        """Computes MD5 hash of a file to detect changes.

        Args:
            file_path (str): Path to the file.

        Returns:
            str: Hexadecimal MD5 hash of the file contents.
        """
        hasher = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                buf = f.read(65536)
                while len(buf) > 0:
                    hasher.update(buf)
                    buf = f.read(65536)
            return hasher.hexdigest()
        except Exception as e:
            logger.error("Failed to calculate hash for %s: %s", file_path, str(e))
            return ""

    def extract_text(self, file_path: str) -> str:
        """Extracts plain text from PDF, DOCX, RTF, or TXT.

        Args:
            file_path (str): Absolute path to the file.

        Returns:
            str: Extracted text content.
        """
        ext = os.path.splitext(file_path)[1].lower()
        text = ""

        try:
            if ext == ".txt":
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                except UnicodeDecodeError:
                    with open(file_path, "r", encoding="latin-1") as f:
                        text = f.read()
            elif ext == ".pdf":
                if PDF_AVAILABLE:
                    reader = PdfReader(file_path)
                    text = "\n".join([page.extract_text() or "" for page in reader.pages])
                else:
                    logger.warning("PDF parser unavailable for: %s", file_path)
            elif ext == ".docx":
                if DOCX_AVAILABLE:
                    doc = docx.Document(file_path)
                    text = "\n".join([para.text for para in doc.paragraphs])
                else:
                    logger.warning("DOCX parser unavailable for: %s", file_path)
            elif ext == ".rtf":
                if RTF_AVAILABLE:
                    with open(file_path, "r", errors="ignore") as f:
                        text = rtf_to_text(f.read())
                else:
                    logger.warning("RTF parser unavailable for: %s", file_path)
            else:
                logger.warning("Unsupported file format: %s", ext)
        except Exception as e:
            logger.error("Error reading file %s: %s", file_path, str(e), exc_info=True)

        return text.strip()

    def identify_joint_type(self, text: str) -> str:
        """Detects the joint type from text.

        Args:
            text (str): Report text.

        Returns:
            str: Joint type (joelho, ombro, quadril, tornozelo, punho, cotovelo,
                 sacroilíacas, coluna, or desconhecido).
        """
        normalized = text.lower()
        # Ordered by specificity — check multi-word before single-word
        joint_patterns = [
            ("sacroilíacas", "sacroilíacas"),
            ("sacroilíaca", "sacroilíacas"),
            ("sacroiliacas", "sacroilíacas"),
            ("sacroiliaca", "sacroilíacas"),
            ("coluna lombar", "coluna_lombar"),
            ("coluna cervical", "coluna_cervical"),
            ("coluna torácica", "coluna_toracica"),
            ("coluna dorsal", "coluna_toracica"),
            ("coluna", "coluna_lombar"),
            ("quadril", "quadril"),
            ("joelho", "joelho"),
            ("tornozelo", "tornozelo"),
            ("ombro", "ombro"),
            ("punho", "punho"),
            ("cotovelo", "cotovelo"),
        ]
        for pattern, joint in joint_patterns:
            if pattern in normalized:
                return joint
        return "desconhecido"

    def split_into_chunks(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """Segments text into overlapping character chunks.

        Args:
            text (str): Full text to segment.
            chunk_size (int): Target length of each chunk in characters.
            overlap (int): Number of characters to overlap between adjacent chunks.

        Returns:
            List[str]: List of text chunks.
        """
        if not text:
            return []
            
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start += chunk_size - overlap
            
        return [c for c in chunks if len(c) > 20] # Filter out tiny noise chunks

    def fetch_embedding(self, text: str) -> Optional[List[float]]:
        """Queries local Ollama to generate embeddings.

        Args:
            text (str): Text to embed.

        Returns:
            Optional[List[float]]: Vector representation or None if failed.
        """
        # Try /api/embed (newer) and fallback to /api/embeddings (older)
        url_embed = f"{self.ollama_url}/api/embed"
        url_embeddings = f"{self.ollama_url}/api/embeddings"
        
        # Detect model to use
        # In a real environment we query get_active_model, let's use a lightweight model like qwen2.5:0.5b
        # since it's already installed on user's machine.
        model = self.fallback_model 
        
        # First try: /api/embed
        try:
            payload = {"model": model, "input": text}
            response = requests.post(url_embed, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                embeddings = data.get("embeddings")
                if embeddings and len(embeddings) > 0:
                    return embeddings[0]
        except Exception:
            pass

        # Second try: /api/embeddings
        try:
            payload = {"model": model, "prompt": text}
            response = requests.post(url_embeddings, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get("embedding")
        except Exception as e:
            logger.warning("Ollama embedding extraction failed: %s", str(e))
            
        return None

    def index_report(self, file_path: str) -> bool:
        """Parses, anonymizes, chunks, vectorizes, and stores a report file.

        Args:
            file_path (str): Path to the report document.

        Returns:
            bool: True if indexed successfully, False if skipped or failed.
        """
        try:
            # 1. Check for duplicate or unmodified files
            file_hash = self.compute_file_hash(file_path)
            cursor = self.conn.cursor()
            cursor.execute("SELECT id, file_hash FROM reports WHERE file_path = ?", (file_path,))
            row = cursor.fetchone()
            
            if row:
                if row[1] == file_hash:
                    logger.info("File %s has not changed. Skipping re-indexing.", file_path)
                    return False
                else:
                    # File has changed, delete old chunks and record before re-indexing
                    cursor.execute("DELETE FROM reports WHERE id = ?", (row[0],))
                    self.conn.commit()

            # 2. Extract and anonymize content
            raw_content = self.extract_text(file_path)
            if not raw_content:
                logger.warning("Skipping empty document: %s", file_path)
                return False

            anonymized = self.anonymizer.anonymize_text(raw_content)
            joint_type = self.identify_joint_type(anonymized)

            # Try to extract a clean separation of findings and conclusion
            findings, conclusion = self._split_findings_and_conclusion(anonymized)

            # 3. Store in SQLite
            cursor.execute("""
                INSERT INTO reports (file_path, file_hash, raw_content, anonymized_content, joint_type, findings, conclusion)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (file_path, file_hash, raw_content, anonymized, joint_type, findings, conclusion))
            report_id = cursor.lastrowid
            
            # 4. Chunking and Vectorization
            chunks = self.split_into_chunks(anonymized)
            for chunk in chunks:
                vector_idx = -1
                
                # Try generating vector embedding
                if FAISS_AVAILABLE:
                    embedding = self.fetch_embedding(chunk)
                    if embedding:
                        arr = np.array([embedding], dtype=np.float32)
                        if self.faiss_index is None:
                            self.embedding_dimension = len(embedding)
                            self.faiss_index = faiss.IndexFlatL2(self.embedding_dimension)
                        
                        # Add vector to FAISS
                        self.faiss_index.add(arr)
                        vector_idx = self.faiss_index.ntotal - 1

                # Write chunk details to SQLite
                cursor.execute("""
                    INSERT INTO report_chunks (report_id, chunk_text, vector_index)
                    VALUES (?, ?, ?)
                """, (report_id, chunk, vector_idx))

            self.conn.commit()
            self._save_vector_index()
            logger.info("Successfully indexed document: %s (ID: %d)", file_path, report_id)
            return True

        except Exception as e:
            logger.error("Failed to index report %s: %s", file_path, str(e), exc_info=True)
            self.conn.rollback()
            return False

    def _split_findings_and_conclusion(self, text: str) -> Tuple[str, str]:
        """Splits findings and conclusion sections from text using regex keywords."""
        findings = text
        conclusion = ""

        # Search for conclusion section markers
        match = re.search(r"\b(conclusão|conclusao|impressão|impressao|diagnóstico|diagnostico)\b", text, re.IGNORECASE)
        if match:
            split_idx = match.start()
            findings = text[:split_idx].strip()
            conclusion = text[split_idx:].strip()

        return findings, conclusion

    def scan_directory(self, folder_path: str) -> int:
        """Scans a directory recursively and indexes all supported files.

        Args:
            folder_path (str): The folder containing reports.

        Returns:
            int: Number of new/updated files successfully indexed.
        """
        # Section 2.1: Resilience
        # If folder doesn't exist, try creating it, fallback to default local directory.
        if not os.path.exists(folder_path):
            try:
                os.makedirs(folder_path)
                logger.info("Created directory: %s", folder_path)
            except Exception as e:
                logger.warning("Could not create directory %s. Using local workspace backup: %s", folder_path, str(e))
                folder_path = os.path.join(os.getcwd(), "Laudos_Local")
                os.makedirs(folder_path, exist_ok=True)

        indexed_count = 0
        supported_exts = {".pdf", ".docx", ".txt", ".rtf"}

        try:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in supported_exts:
                        full_path = os.path.join(root, file)
                        if self.index_report(full_path):
                            indexed_count += 1
        except Exception as e:
            logger.error("Failed scanning directory %s. Error: %s", folder_path, str(e), exc_info=True)

        return indexed_count

    def search_similar(self, query: str, joint_type: str, k: int = 3) -> str:
        """Searches for reports/chunks similar to the query.

        Tries vector search with FAISS, falls back to a custom TF-IDF search.

        Args:
            query (str): Search term or clinical findings.
            joint_type (str): Joint type filter.
            k (int): Number of chunks to retrieve.

        Returns:
            str: Concatenated text of similar reports/chunks to act as LLM context.
        """
        # Ensure joint type is formatted
        joint_type = joint_type.lower()
        
        # 1. Attempt FAISS search if available
        if FAISS_AVAILABLE and self.faiss_index is not None:
            query_vector = self.fetch_embedding(query)
            if query_vector:
                try:
                    q_arr = np.array([query_vector], dtype=np.float32)
                    distances, indices = self.faiss_index.search(q_arr, k * 2) # Grab extra to filter by joint
                    
                    found_chunks: List[str] = []
                    cursor = self.conn.cursor()
                    
                    for idx in indices[0]:
                        if idx == -1:
                            continue
                        
                        # Query sqlite for chunk text and joint type
                        cursor.execute("""
                            SELECT rc.chunk_text, r.joint_type 
                            FROM report_chunks rc 
                            JOIN reports r ON rc.report_id = r.id 
                            WHERE rc.vector_index = ?
                        """, (int(idx),))
                        row = cursor.fetchone()
                        if row:
                            chunk_text, chunk_joint = row
                            if chunk_joint.lower() == joint_type:
                                found_chunks.append(chunk_text)
                                if len(found_chunks) >= k:
                                    break
                                    
                    if found_chunks:
                        return "\n\n---\n\n".join(found_chunks)
                except Exception as e:
                    logger.error("FAISS vector query failed. Falling back to TF-IDF. Error: %s", str(e), exc_info=True)

        # 2. Fallback to TF-IDF search
        return self._tfidf_fallback_search(query, joint_type, k)

    def _tfidf_fallback_search(self, query: str, joint_type: str, k: int) -> str:
        """Pure-Python TF-IDF similarity search as a fallback when vector search is down."""
        logger.info("Executing TF-IDF fallback search for joint_type: %s", joint_type)
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT raw_content FROM reports WHERE joint_type = ?", (joint_type,))
            rows = cursor.fetchall()
            
            if not rows:
                return ""
                
            documents = [row[0] for row in rows]
            
            # Helper to tokenize and stem-ish
            def tokenize(text: str) -> List[str]:
                return re.findall(r"\b\w{3,}\b", text.lower())

            query_tokens = tokenize(query)
            if not query_tokens:
                return "\n\n---\n\n".join(documents[:k])

            # Calculate TF-IDF vectors
            all_tokens_sets = [tokenize(doc) for doc in documents]
            
            # Document frequency
            df: Dict[str, int] = {}
            for token in set(query_tokens):
                df[token] = sum(1 for doc_tokens in all_tokens_sets if token in doc_tokens)

            # Cosine similarity matching
            scores: List[Tuple[float, str]] = []
            for doc, doc_tokens in zip(documents, all_tokens_sets):
                # Simple TF-IDF dot product
                score = 0.0
                doc_tf: Dict[str, int] = {}
                for token in doc_tokens:
                    doc_tf[token] = doc_tf.get(token, 0) + 1

                for token in query_tokens:
                    if token in doc_tf and df.get(token, 0) > 0:
                        # TF * IDF where IDF = log(N / DF)
                        idf = math.log(len(documents) / df[token])
                        score += doc_tf[token] * idf

                scores.append((score, doc))

            # Sort by score descending
            scores.sort(key=lambda x: x[0], reverse=True)
            top_k_docs = [doc for score, doc in scores[:k] if score > 0.0]
            
            # If no keyword matches, just return the first few
            if not top_k_docs:
                top_k_docs = documents[:k]

            return "\n\n---\n\n".join(top_k_docs)

        except Exception as e:
            logger.error("TF-IDF fallback search failed: %s", str(e), exc_info=True)
            return ""

    def add_custom_laudo(self, raw_text: str) -> bool:
        """Manually adds a structured laudo to database and indexes it.

        Args:
            raw_text (str): Complete finalized report text.

        Returns:
            bool: True if successfully added.
        """
        try:
            # Create a mock filename based on hash
            text_hash = hashlib.md5(raw_text.encode("utf-8")).hexdigest()
            mock_path = f"manual_upload_{text_hash}.txt"
            
            # Write to a file in workspace directory
            folder_path = os.path.join(os.getcwd(), "Laudos_Manual")
            os.makedirs(folder_path, exist_ok=True)
            full_path = os.path.join(folder_path, mock_path)
            
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(raw_text)
                
            return self.index_report(full_path)
        except Exception as e:
            logger.error("Failed to add manual laudo. Error: %s", str(e), exc_info=True)
            return False
            
    def close(self) -> None:
        """Closes SQLite database connection safely."""
        try:
            self.conn.close()
        except Exception:
            pass
