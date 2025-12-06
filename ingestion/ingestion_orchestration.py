# ingestion/ingestion_orchestration.py
import os
import traceback
from typing import Optional, Dict, Any

from utils.logger import get_logger
from ingestion import DataLoader, Chunker, Summarizer, EmbedStore
from langchain_core.documents import Document


class IngestionPipeline:
    """
    Ingestion orchestrator (dependency-injectable).
    Usage:
        pipeline = IngestionPipeline()
        result = pipeline.run_ingestion("/path/to/file.pdf")
    """

    def __init__(
        self,
        loader: Optional[DataLoader] = None,
        chunker: Optional[Chunker] = None,
        summarizer: Optional[Summarizer] = None,
        embed_store: Optional[EmbedStore] = None,
    ):
        self.loader = loader or DataLoader()
        self.chunker = chunker or Chunker()
        self.summarizer = summarizer or Summarizer()
        self.embed_store = embed_store or EmbedStore()
        self.logger = get_logger(self.__class__.__name__)

    def _quick_already_ingested_check(self, file_hash: str, first_chunk_preview: str) -> bool:
        """
        Cheap idempotence check:
        - Run a short semantic query using first_chunk_preview, filtered by file_hash.
        - If any result exists, assume file already ingested.
        """
        try:
            if not first_chunk_preview:
                return False
            results = self.embed_store.similarity_search(
                query=first_chunk_preview,
                top_k=1,
                filter={"file_hash": file_hash}
            )
            return bool(results)
        except Exception as e:
            # On any error, be conservative and continue ingestion (do not silently skip)
            self.logger.warning(f"Idempotence check failed (will proceed with ingestion): {e}")
            return False

    def run_ingestion(self, file_path: str, force_reingest: bool = False) -> Dict[str, Any]:
        """
        Run full ingestion pipeline:
          1) load & extract
          2) chunk
          3) summarize / AI-enhance
          4) upsert into single vector store

        Returns a summary dict with file_hash, file_name, num_chunks, upserted_count.
        """
        self.logger.info(f"Starting ingestion for: {file_path}")
        file_name = os.path.basename(file_path)

        try:
            # 1) Load & extract (expects loader.load_and_extract -> (elements, file_hash, file_name))
            load_result = self.loader.load_and_extract(file_path)
            # Some loader implementations return tuple (elements, file_hash, file_name)
            if isinstance(load_result, tuple) and len(load_result) >= 2:
                elements, file_hash = load_result[0], load_result[1]
            else:
                # Backwards-compatible: if loader returns only elements, compute fallback hash via loader if available
                elements = load_result
                file_hash = getattr(self.loader, "_compute_file_hash", lambda p=None: None)(file_path)

            # 2) Chunking: chunker.chunk should accept elements + file metadata or we pass them explicitly
            # Prefer chunker.chunk(elements, file_name=file_name, file_hash=file_hash)
            try:
                chunks = self.chunker.chunk(elements, file_name=file_name, file_hash=file_hash)
            except TypeError:
                # fallback for older chunker signature
                chunks = self.chunker.chunk(elements)

            # ensure chunks are a list of langchain Documents
            processed_chunks = []
            for c in chunks:
                if isinstance(c, Document):
                    processed_chunks.append(c)
                else:
                    # if chunker returned plain dict or str -> convert to Document with minimal metadata
                    text = c.get("text") if isinstance(c, dict) else str(c)
                    meta = c.get("metadata", {}) if isinstance(c, dict) else {}
                    meta.setdefault("file_name", file_name)
                    meta.setdefault("file_hash", file_hash)
                    processed_chunks.append(Document(page_content=text, metadata=meta))

            # Quick idempotence check: if file already ingested (and not forced), skip
            if not force_reingest and processed_chunks:
                preview = processed_chunks[0].page_content[:200]
                already = self._quick_already_ingested_check(file_hash=file_hash, first_chunk_preview=preview)
                if already:
                    self.logger.info(f"File {file_name} (hash={file_hash[:8]}) appears already ingested. Skipping upsert.")
                    return {
                        "file_name": file_name,
                        "file_hash": file_hash,
                        "skipped": True,
                        "reason": "already_ingested",
                        "num_chunks": len(processed_chunks),
                        "upserted_count": 0
                    }

            # 3) Summarize (returns embedding-ready Documents)
            try:
                documents = self.summarizer.summary(processed_chunks, file_name=file_name)
            except TypeError:
                # older summarizer may only accept chunks and filename, attempt best-effort
                documents = self.summarizer.summary(processed_chunks, file_name)

            # 4) Upsert into EmbedStore
            self.embed_store.upsert_documents(documents)

            self.logger.info(f"Ingestion completed for: {file_path} (file_hash={file_hash})")
            return {
                "file_name": file_name,
                "file_hash": file_hash,
                "skipped": False,
                "num_chunks": len(documents),
                "upserted_count": len(documents)
            }

        except Exception as e:
            self.logger.error(f"Ingestion failed for {file_path}: {e}")
            self.logger.debug(traceback.format_exc())
            raise
