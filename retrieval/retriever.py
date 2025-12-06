# retrieval/retriever.py
from typing import List, Optional
from utils.logger import get_logger
from langchain_core.documents import Document
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

class Retriever:
    """
    Retriever that performs semantic (vector) search + optional BM25 fallback + optional reranking.
    Returns list[Document] (langchain_core.documents.Document) with page_content and metadata.
    """

    def __init__(self, embed_store):
        self.embed_store = embed_store
        self.logger = get_logger(self.__class__.__name__)

    # Semantic similarity via EmbedStore
    def query_db(self, query: str, top_k: int = 5, filter: Optional[dict] = None) -> List[Document]:
        self.logger.info(f"[Retriever] semantic query (top_k={top_k})")
        try:
            results = self.embed_store.similarity_search(query=query, top_k=top_k, filter=filter)
            # Ensure results are Documents
            results = [r for r in results if hasattr(r, "page_content")]
            self.logger.info(f"[Retriever] semantic search found {len(results)} docs")
            return results
        except Exception as e:
            self.logger.error(f"[Retriever] Error querying vector store: {e}")
            raise

    # BM25 search via EmbedStore helper (returns list[str] by default). We convert to Document if possible.
    def bm25_query(self, query: str, top_k: int = 5) -> List[Document]:
        self.logger.info(f"[Retriever] BM25 query (top_k={top_k})")
        try:
            # Use embed_store.bm25_search if available (returns list[str])
            if hasattr(self.embed_store, "bm25_search"):
                texts = self.embed_store.bm25_search(query, top_k=top_k)
                # Convert to Document objects with minimal metadata (can't reconstruct original metadata here)
                docs = [Document(page_content=t, metadata={}) for t in texts]
                self.logger.info(f"[Retriever] BM25 returned {len(docs)} docs")
                return docs
            else:
                self.logger.warning("[Retriever] BM25 not implemented in EmbedStore")
                return []
        except Exception as e:
            self.logger.error(f"[Retriever] BM25 query error: {e}")
            return []

    # Hybrid: combine semantic + BM25 based on alpha, then optional reranker
    def hybrid_query(self, query: str, top_k: int = 5, alpha: float = 0.5, use_reranker: bool = True, filter: Optional[dict] = None) -> List[Document]:
        """
        Hybrid search:
          - n_dense = int(top_k * alpha)
          - n_sparse = top_k - n_dense
          - Merge results, dedupe by metadata/id/text
          - Optionally rerank using FlashrankRerank (or skip on failure)
        Returns: list[Document]
        """
        self.logger.info(f"[Retriever] hybrid_query query='{query}' top_k={top_k} alpha={alpha}")

        # 1) semantic/dense
        try:
            dense_docs = self.query_db(query=query, top_k=top_k, filter=filter)
        except Exception:
            dense_docs = []

        # 2) sparse/BM25
        sparse_docs = self.bm25_query(query=query, top_k=top_k)

        n_dense = max(0, int(top_k * alpha))
        n_sparse = max(0, top_k - n_dense)

        hybrid_candidates: List[Document] = (dense_docs[:n_dense] if dense_docs else []) + (sparse_docs[:n_sparse] if sparse_docs else [])

        # Deduplicate by id or text
        seen = set()
        unique_docs: List[Document] = []
        for d in hybrid_candidates:
            # Use metadata id if available, else fallback to page_content hash
            doc_id = None
            try:
                doc_id = d.metadata.get("id") if getattr(d, "metadata", None) else None
            except Exception:
                doc_id = None
            key = doc_id or (d.page_content[:200])
            if key not in seen:
                seen.add(key)
                unique_docs.append(d)

        # Optionally rerank
        if use_reranker and unique_docs:
            try:
                self.logger.info(f"[Retriever] reranking {len(unique_docs)} candidates")
                reranker = FlashrankRerank(top_n=top_k)
                # Flashrank expects Document objects
                reranked_docs = reranker.compress_documents(unique_docs, query)
                # Ensure output is list[Document]
                unique_docs = [doc for doc in reranked_docs if hasattr(doc, "page_content")]
            except Exception as e:
                self.logger.warning(f"[Retriever] Reranking failed: {e}")

        # ensure we return at most top_k documents
        final = unique_docs[:top_k]
        self.logger.info(f"[Retriever] hybrid_query returning {len(final)} results")
        return final
