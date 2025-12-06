# ingestion/vector_store.py
from typing import List, Optional, Dict, Any
import hashlib
import os
import traceback

from utils.logger import get_logger
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config.models import HF_EMBEDDING_MODEL, VECTOR_DB_PATH, COLLECTION_NAME

# Optional BM25; keep but don't enable by default
try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None


class EmbedStore:
    """
    Single vector store wrapper (Chroma) with a small, explicit API:
      - upsert_documents(docs: List[Document], persist: bool = True)
      - similarity_search(query: str, top_k: int = 5, filter: Optional[Dict] = None) -> List[Document]
      - delete_by_filter(filter: Dict[str, Any])
      - get_document_by_id(doc_id: str) -> Optional[Document]
      - persist()
      - build_bm25_from_all_documents()  # optional
    """

    def __init__(
        self,
        persist_directory: str = None,
        collection_name: str = None,
        embedding_model_name: str = None,
        device: str = "cpu",
    ):
        self.logger = get_logger(self.__class__.__name__)
        self.persist_directory = persist_directory or VECTOR_DB_PATH
        self.collection_name = collection_name or COLLECTION_NAME
        
        os.makedirs(self.persist_directory, exist_ok=True)

        # Embedding model wrapper (LangChain HF wrapper)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name or HF_EMBEDDING_MODEL,
            model_kwargs={"device": device},
        )

        # Single Chroma collection for the project
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory,
        )

        # Optional BM25 index (built on demand)
        self._bm25 = None
        self._bm25_texts = []

        self.logger.info(f"EmbedStore initialized: collection={self.collection_name}, dir={self.persist_directory}")

    # --------------------------
    # ID utilities
    # --------------------------
    def _make_doc_id(self, file_hash: str, chunk_index: int, start_char: int, end_char: int) -> str:
        key = f"{file_hash}:{chunk_index}:{start_char}:{end_char}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    # --------------------------
    # Upsert / persist
    # --------------------------
    def upsert_documents(self, documents: List[Document]) -> None:
        """
        Upsert documents into the single vector collection.

        Each Document must have:
          - page_content (str)
          - metadata including: file_hash, chunk_index, start_char (optional), end_char (optional)
        """
        try:
            docs_to_add = []
            for d in documents:
                meta = dict(d.metadata or {})
                file_hash = meta.get("file_hash", "")
                chunk_index = int(meta.get("chunk_index", 0))
                start_char = int(meta.get("start_char", 0))
                end_char = int(meta.get("end_char", len(d.page_content or "")))
                doc_id = meta.get("id") or self._make_doc_id(file_hash, chunk_index, start_char, end_char)

                # ensure id is present in metadata for traceability
                meta.update({"id": doc_id, "file_hash": file_hash, "chunk_index": chunk_index})

                docs_to_add.append(Document(page_content=d.page_content, metadata=meta))

            # Add documents. LangChain's Chroma wrapper will handle dedup/upsert semantics by id where supported.
            self.vectorstore.add_documents(docs_to_add)
            self.logger.info(f"Upserted {len(docs_to_add)} documents into '{self.collection_name}'")
        except Exception as e:
            self.logger.error("EmbedStore.upsert_documents error: " + str(e))
            self.logger.debug(traceback.format_exc())
            raise

    # --------------------------
    # Search
    # --------------------------
    def similarity_search(self, query: str, top_k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Run a semantic similarity search against the vectorstore.
        `filter` is a metadata filter dict compatible with Chroma's filtering.
        Returns a list of langchain_core.documents.Document
        """
        try:
            results = self.vectorstore.similarity_search(query, k=top_k, filter=filter)
            return results
        except Exception as e:
            self.logger.error("EmbedStore.similarity_search error: " + str(e))
            self.logger.debug(traceback.format_exc())
            raise

    # --------------------------
    # Delete
    # --------------------------
    def delete_by_filter(self, filter: Dict[str, Any]) -> None:
        """
        Delete documents matching the given metadata filter.
        Ex: {"file_hash": "abc..."} to remove all docs from a file.
        """
        try:
            self.vectorstore.delete(filter=filter)
            self.vectorstore.persist()
            self.logger.info(f"Deleted documents by filter: {filter}")
        except Exception as e:
            self.logger.error("EmbedStore.delete_by_filter error: " + str(e))
            self.logger.debug(traceback.format_exc())
            raise

    # --------------------------
    # Get / Persist
    # --------------------------
    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a document by its ID. Returns a langchain Document or None.
        """
        try:
            docs = self.vectorstore.get(ids=[doc_id])
            if docs and len(docs) > 0:
                return docs[0]
            return None
        except Exception as e:
            self.logger.error("EmbedStore.get_document_by_id error: " + str(e))
            self.logger.debug(traceback.format_exc())
            raise

    def persist(self) -> None:
        """Force persist to disk."""
        try:
            self.vectorstore.persist()
            self.logger.info("Vectorstore persisted to disk")
        except Exception as e:
            self.logger.error("EmbedStore.persist error: " + str(e))
            self.logger.debug(traceback.format_exc())
            raise

    # --------------------------
    # BM25 helper (optional)
    # --------------------------
    def build_bm25_from_all_documents(self) -> None:
        """
        Build/update a BM25 index from all document texts stored in the collection.
        This is optional and only available if rank_bm25 is installed.
        """
        if BM25Okapi is None:
            self.logger.warning("BM25 not available (rank_bm25 missing)")
            return

        try:
            # Fetch all docs' texts - vectorstore.get() returns a dict with 'documents' key
            all_data = self.vectorstore.get()
            texts = all_data.get('documents', []) if isinstance(all_data, dict) else []
            
            # Filter out empty texts
            texts = [t for t in texts if t and t.strip()]
            
            if not texts:
                self.logger.warning("No documents found for BM25 indexing")
                return
            
            tokenized = [t.split() for t in texts]
            self._bm25 = BM25Okapi(tokenized)
            self._bm25_texts = texts
            self.logger.info(f"BM25 index built from {len(texts)} documents")
        except Exception as e:
            self.logger.error("EmbedStore.build_bm25 error: " + str(e))
            self.logger.debug(traceback.format_exc())

    def bm25_search(self, query: str, top_k: int = 5) -> List[str]:
        """
        Search with BM25; returns top_k texts. Requires BM25 to be built first.
        """
        if self._bm25 is None:
            self.build_bm25_from_all_documents()
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(query.split())
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self._bm25_texts[i] for i in top_n]
    
    # new feature
    def sample_documents(self, k: int = 5) -> List[Document]:
        """
        Retrieve a random sample of documents (or first k) from the store.
        Useful for generating suggested questions.
        """
        try:
            # Chroma get() returns dict with lists. limit=k gets first k.
            # We assume first k is 'random enough' for this purpose or we could fetch more and shuffle.
            data = self.vectorstore.get(limit=k)
            docs = []
            if data and data['documents']:
                for i in range(len(data['documents'])):
                    text = data['documents'][i]
                    meta = data['metadatas'][i] if data['metadatas'] else {}
                    docs.append(Document(page_content=text, metadata=meta))
            return docs
        except Exception as e:
            self.logger.error("EmbedStore.sample_documents error: " + str(e))
            return []
