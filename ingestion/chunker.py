# ingestion/chunker.py (core)
from utils.logger import get_logger
from unstructured.chunking.title import chunk_by_title
from langchain_core.documents import Document
import math

class Chunker:
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    def _estimate_tokens(self, text: str) -> int:
        # crude tokens estimator; replace with tiktoken or similar if available
        return max(1, math.ceil(len(text) / 4))

    def chunk(self, elements: list, file_name: str, file_hash: str, max_chars: int = 3000) -> list:
        """
        Returns list of langchain_core.Document with metadata required by EmbedStore.
        """
        self.logger.info(f"Chunking elements for file {file_name}")
        raw_chunks = chunk_by_title(
                elements=elements,
                max_characters=3000,
                new_after_n_chars=2400,
                combine_text_under_n_chars=500)
        docs = []
        for idx, rc in enumerate(raw_chunks):
            text = rc.get("text") if isinstance(rc, dict) else str(rc)
            start_char = 0  # if source gives offsets, use them; otherwise leave 0
            end_char = len(text)
            tokens = self._estimate_tokens(text)
            metadata = {
                "file_name": file_name,
                "file_hash": file_hash,
                "chunk_index": idx,
                "start_char": start_char,
                "end_char": end_char,
                "token_count": tokens,
                "source": rc.get("source", None) if isinstance(rc, dict) else None
            }
            docs.append(Document(page_content=text, metadata=metadata))
        self.logger.info(f"Created {len(docs)} chunks for {file_name}")
        return docs