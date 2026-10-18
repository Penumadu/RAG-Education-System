"""
RAG Engine: Document ingestion, chunking, embedding, and vector retrieval.
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import EmptyFileError, FileNotDecryptedError, PdfReadError

logger = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./.chroma_data")
DEFAULT_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))


@dataclass
class RetrievedChunk:
    """Represents a retrieved document chunk with relevance score."""

    content: str
    source: str
    page: int
    score: float


class DocumentProcessingError(Exception):
    """Raised when processing a document fails."""


def sanitize_text(text: str) -> str:
    """Sanitize raw extracted PDF text to avoid tokenizer/embedding crashes."""
    if not text:
        return ""
    # Remove null bytes which crash C/C++ llama.cpp tokenizers
    text = text.replace("\x00", "")
    # Normalize unicode (compatibility composition)
    text = unicodedata.normalize("NFKC", text)
    # Remove non-printable control characters except standard whitespace
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)
    # Replace multiple whitespaces/tabs with single space
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def check_ollama_status(ollama_url: str = DEFAULT_OLLAMA_URL, timeout: float = 0.8) -> bool:
    """Probe if Ollama daemon is reachable and responding."""
    import urllib.request
    try:
        req = urllib.request.Request(
            f"{ollama_url.rstrip('/')}/api/tags",
            headers={"User-Agent": "RAG-Engine"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


class RAGEngine:
    """Core RAG engine managing document ingestion and vector retrieval."""

    def __init__(
        self,
        persist_directory: str = DEFAULT_CHROMA_DIR,
        collection_name: str = "pdf_knowledge_base",
        embed_model: str = DEFAULT_EMBED_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        relevance_threshold: float = 0.35,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embed_model = embed_model
        self.ollama_url = ollama_url
        self.relevance_threshold = relevance_threshold
        self.batch_size = batch_size

        gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if gemini_api_key:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            embed_model_name = os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004")
            self.embed_model = embed_model_name
            logger.info("Using Google Generative AI Embeddings (%s)", embed_model_name)
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=embed_model_name,
                google_api_key=gemini_api_key,
            )
            self.provider = "gemini"
        else:
            logger.info("Using Ollama Embeddings (%s) at %s", self.embed_model, self.ollama_url)
            self.embeddings = OllamaEmbeddings(
                model=self.embed_model,
                base_url=self.ollama_url,
            )
            self.provider = "ollama"

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    @property
    def document_count(self) -> int:
        """Return the number of documents (chunks) currently in the vector store."""
        try:
            return self.vector_store._collection.count()
        except Exception:
            return 0

    def extract_text_from_pdf(self, file_path_or_bytes: str | bytes, filename: str) -> List[Document]:
        """Extract pages from a PDF into LangChain Documents with error handling."""
        documents: List[Document] = []
        try:
            if isinstance(file_path_or_bytes, bytes):
                import io

                stream = io.BytesIO(file_path_or_bytes)
                reader = PdfReader(stream)
            else:
                reader = PdfReader(file_path_or_bytes)

            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception as e:
                    raise DocumentProcessingError(
                        f"PDF '{filename}' is password protected and cannot be read."
                    ) from e

            total_pages = len(reader.pages)
            if total_pages == 0:
                raise DocumentProcessingError(f"PDF '{filename}' contains no readable pages.")

            for page_idx, page in enumerate(reader.pages, start=1):
                raw_text = page.extract_text() or ""
                cleaned = sanitize_text(raw_text)
                if cleaned:
                    documents.append(
                        Document(
                            page_content=cleaned,
                            metadata={
                                "source": filename,
                                "page": page_idx,
                                "total_pages": total_pages,
                            },
                        )
                    )

            if not documents:
                raise DocumentProcessingError(
                    f"PDF '{filename}' had {total_pages} page(s) but no extractable text (it may be a scanned image)."
                )

            return documents

        except (EmptyFileError, PdfReadError, FileNotDecryptedError) as e:
            logger.error("Failed to parse PDF '%s': %s", filename, e)
            raise DocumentProcessingError(f"Invalid or corrupted PDF file '{filename}': {e}") from e
        except Exception as e:
            if isinstance(e, DocumentProcessingError):
                raise
            logger.error("Unexpected error reading PDF '%s': %s", filename, e)
            raise DocumentProcessingError(f"Error reading '{filename}': {e}") from e

    def ingest_pdf(
        self,
        file_path_or_bytes: str | bytes,
        filename: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> int:
        """
        Parse, chunk, and index a PDF file into the vector store.
        Embeds chunks in controlled batches to avoid Ollama runner/tokenizer buffer overflows.
        """
        docs = self.extract_text_from_pdf(file_path_or_bytes, filename)
        raw_chunks = self.text_splitter.split_documents(docs)

        # Sanitize chunks and enforce max chunk size safety
        chunks: List[Document] = []
        for idx, chunk in enumerate(raw_chunks):
            content = sanitize_text(chunk.page_content)
            # Hard limit chunk length to 1500 chars to prevent tokenizer context overflow
            if len(content) > 1500:
                content = content[:1500]

            if content:
                chunk.page_content = content
                chunk.metadata["chunk_id"] = f"{filename}_p{chunk.metadata.get('page', 1)}_c{idx}"
                chunks.append(chunk)

        total_chunks = len(chunks)
        if total_chunks == 0:
            raise DocumentProcessingError(f"PDF '{filename}' produced no readable text chunks.")

        indexed_count = 0
        batch_size = max(1, self.batch_size)

        last_error: Optional[Exception] = None
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            try:
                self.vector_store.add_documents(batch)
                indexed_count += len(batch)
            except Exception as batch_error:
                last_error = batch_error
                logger.warning(
                    "Batch insertion failed for chunks %d-%d of '%s': %s. Retrying item-by-item...",
                    i,
                    i + len(batch),
                    filename,
                    batch_error,
                )
                # Resilient fallback: index chunk-by-chunk to bypass any single malformed string
                for single_chunk in batch:
                    try:
                        self.vector_store.add_documents([single_chunk])
                        indexed_count += 1
                    except Exception as single_error:
                        last_error = single_error
                        logger.error(
                            "Skipping corrupt or un-indexable chunk on page %s of '%s': %s",
                            single_chunk.metadata.get("page"),
                            filename,
                            single_error,
                        )

            if progress_callback:
                progress_callback(
                    min(i + len(batch), total_chunks),
                    total_chunks,
                    f"Indexing '{filename}': {min(i + len(batch), total_chunks)}/{total_chunks} chunks...",
                )

        if indexed_count == 0 and total_chunks > 0:
            err_details = f" ({last_error})" if last_error else ""
            raise DocumentProcessingError(
                f"Failed to index '{filename}'. Unable to connect to embedding service at '{self.ollama_url}'{err_details}. "
                "Ensure your LLM/embedding provider is running and reachable from the server."
            )

        logger.info("Successfully indexed %d/%d chunks from '%s'", indexed_count, total_chunks, filename)
        return indexed_count

    def retrieve(self, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        """
        Retrieve chunks with distance/relevance scoring.
        Applies confidence thresholding. If no documents pass threshold, returns empty list.
        Returns empty list immediately if the collection is empty.
        """
        # Guard: don't query an empty collection (Chroma raises when k > count)
        collection_size = self.document_count
        if collection_size == 0:
            return []

        # Clamp k to collection size to prevent Chroma errors
        effective_k = min(top_k, collection_size)

        try:
            results_with_score = self.vector_store.similarity_search_with_score(
                query, k=effective_k
            )
        except Exception as e:
            logger.error("Retrieval failed for query '%s': %s", query, e)
            raise

        retrieved: List[RetrievedChunk] = []
        for doc, score in results_with_score:
            relevance = max(0.0, 1.0 - float(score) / 2.0) if score >= 0 else 0.0

            if relevance >= self.relevance_threshold:
                retrieved.append(
                    RetrievedChunk(
                        content=doc.page_content,
                        source=doc.metadata.get("source", "Unknown Document"),
                        page=int(doc.metadata.get("page", 1)),
                        score=round(relevance, 3),
                    )
                )

        return retrieved

    def get_indexed_documents(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a summary of all documents currently persisted in the vector store.
        Returns a mapping of filename -> {"chunk_count": int, "page_count": int, "min_page": int, "max_page": int}.
        """
        if self.document_count == 0:
            return {}

        try:
            collection_data = self.vector_store._collection.get(include=["metadatas"])
            metadatas = collection_data.get("metadatas", [])
            docs: Dict[str, Dict[str, Any]] = {}
            for meta in metadatas:
                if not meta:
                    continue
                source = str(meta.get("source", "Unknown Document"))
                if source not in docs:
                    docs[source] = {
                        "chunk_count": 0,
                        "pages": set(),
                        "total_pages": int(meta.get("total_pages", 1)),
                    }
                docs[source]["chunk_count"] += 1
                if "page" in meta:
                    docs[source]["pages"].add(int(meta["page"]))

            summary: Dict[str, Dict[str, Any]] = {}
            for source, data in docs.items():
                pages_list = sorted(list(data["pages"]))
                summary[source] = {
                    "chunk_count": data["chunk_count"],
                    "page_count": len(pages_list) if pages_list else data["total_pages"],
                    "min_page": pages_list[0] if pages_list else 1,
                    "max_page": pages_list[-1] if pages_list else data["total_pages"],
                }
            return summary
        except Exception as e:
            logger.error("Failed to query indexed documents from vector store: %s", e)
            return {}

    def get_document_names(self) -> List[str]:
        """Return a sorted list of unique document filenames present in the vector store."""
        return sorted(list(self.get_indexed_documents().keys()))

    def delete_document(self, filename: str) -> int:
        """
        Remove all chunks associated with a specific PDF filename from the vector store.
        Returns the number of chunks removed.
        """
        try:
            data = self.vector_store._collection.get(
                where={"source": filename},
                include=["metadatas"],
            )
            ids = data.get("ids", [])
            chunk_count = len(ids)
            if chunk_count > 0:
                self.vector_store._collection.delete(where={"source": filename})
                logger.info("Deleted %d chunks for document '%s'", chunk_count, filename)
            return chunk_count
        except Exception as e:
            logger.error("Failed to delete document '%s': %s", filename, e)
            raise DocumentProcessingError(f"Failed to delete '{filename}': {e}") from e

    def delete_documents(self, filenames: List[str]) -> int:
        """Remove multiple documents by filename. Returns total chunks removed."""
        total = 0
        for fn in filenames:
            total += self.delete_document(fn)
        return total

    def clear(self) -> int:
        """Reset the vector store collection. Returns count of documents that were cleared."""
        count = self.document_count
        try:
            self.vector_store.reset_collection()
            logger.info("Vector collection '%s' reset successfully (%d documents cleared).", self.collection_name, count)
        except Exception as e:
            logger.warning("Reset collection encountered: %s", e)
        return count

