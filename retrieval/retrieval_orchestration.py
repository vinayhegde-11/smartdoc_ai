# retrieval/retrieval_orchestration.py
from typing import Dict, Any, Optional
from utils.logger import get_logger
from ingestion import EmbedStore
from retrieval.retriever import Retriever

class RetrievalPipeline:
    """
    Retrieval orchestrator that holds an EmbedStore instance and Retriever.
    Allows DI of embed_store for testing.
    """

    def __init__(self, embed_store: Optional[EmbedStore] = None):
        self.embed_store = embed_store or EmbedStore()
        self.retriever = Retriever(self.embed_store)
        self.logger = get_logger(self.__class__.__name__)

    def run_retrieval(self, query: str, top_k: int = 5, alpha: float = 0.5, use_reranker: bool = True, filter: Optional[dict] = None) -> Dict[str, Any]:
        self.logger.info(f"[RetrievalPipeline] Starting retrieval for query: {query}")
        results = self.retriever.hybrid_query(query=query, top_k=top_k, alpha=alpha, use_reranker=use_reranker, filter=filter)
        # Return serializable structure (texts + metadata) so callers don't need Document objects
        serial_results = [{"text": r.page_content, "metadata": r.metadata or {}} for r in results]
        self.logger.info(f"[RetrievalPipeline] Retrieval returned {len(serial_results)} results")
        return {"query": query, "results": serial_results, "count": len(serial_results)}
