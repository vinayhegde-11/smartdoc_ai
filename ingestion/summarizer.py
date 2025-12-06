# ingestion/summarizer.py
from typing import List, Optional
import os
import json
import base64
from pathlib import Path
import traceback

from utils.logger import get_logger
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_google_vertexai import ChatVertexAI
from config.models import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, ARTIFACTS_DIR


ARTIFACTS_ROOT = Path(ARTIFACTS_DIR)


class Summarizer:
    """
    Summarizer converts chunk Documents -> embedding-ready Documents.
    - Input: list[langchain_core.documents.Document] where each doc has .page_content and metadata that includes:
        - file_name (str)
        - file_hash (str)
        - chunk_index (int)
      (chunker should provide these)
    - Output: list[Document] with summary text in .page_content and compact metadata:
        {
          "file_name": ...,
          "file_hash": ...,
          "chunk_index": ...,
          "original_ref": "artifacts/<file_hash>/original_{chunk_index}.txt",
          "artifact_refs": {"images": [...], "tables": [...]},
          "original_preview": "<first 300 chars>",
          "original_length": 1234
        }
    """

    def __init__(self, model_name: str = None, temperature: float = None, max_tokens: int = None):
        self.logger = get_logger(self.__class__.__name__)
        self.llm_model_name = model_name or LLM_MODEL
        self.llm_temperature = temperature if temperature is not None else LLM_TEMPERATURE
        self.llm_max_tokens = max_tokens or LLM_MAX_TOKENS

        # Ensure artifacts root
        ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)

    # ---------------------
    # Utilities for artifacts
    # ---------------------
    def _ensure_file_dir(self, file_hash: str) -> Path:
        p = ARTIFACTS_ROOT / file_hash
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _persist_original_text(self, file_hash: str, chunk_index: int, text: str) -> str:
        d = self._ensure_file_dir(file_hash)
        path = d / f"original_{chunk_index}.txt"
        try:
            path.write_text(text, encoding="utf-8")
        except Exception:
            # best-effort; log but continue
            self.logger.warning(f"Failed to persist original text for {file_hash}:{chunk_index}")
        return str(path)

    def _persist_table_html(self, file_hash: str, chunk_index: int, table_html: str, table_i: int) -> str:
        d = self._ensure_file_dir(file_hash) / "tables"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"chunk_{chunk_index}_table_{table_i}.html"
        try:
            path.write_text(table_html, encoding="utf-8")
        except Exception:
            self.logger.warning(f"Failed to persist table for {file_hash}:{chunk_index}:{table_i}")
        return str(path)

    def _persist_image(self, file_hash: str, chunk_index: int, image_base64: str, image_i: int) -> str:
        d = self._ensure_file_dir(file_hash) / "images"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"chunk_{chunk_index}_image_{image_i}.jpg"
        try:
            # image_base64 is expected to be raw base64 (no data: prefix)
            image_bytes = base64.b64decode(image_base64)
            path.write_bytes(image_bytes)
        except Exception:
            self.logger.warning(f"Failed to persist image for {file_hash}:{chunk_index}:{image_i}")
        return str(path)

    # ---------------------
    # LLM wrapper
    # ---------------------
    def _get_llm(self) -> ChatVertexAI:
        return ChatVertexAI(
            model_name=self.llm_model_name,
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens
        )

    def create_ai_enhanced_summary(self, text: str, tables: List[str], image_refs: List[str]) -> str:
        """
        Create a searchable description using Vertex AI.
        Falls back to original text (or truncated) on failure.
        """
        try:
            model = self._get_llm()

            prompt = (
                "You are creating a searchable description for document content retrieval.\n\n"
                "CONTENT TO ANALYZE:\n\n"
                "TEXT:\n"
                f"{text}\n\n"
            )

            if tables:
                prompt += "TABLES:\n"
                for i, t in enumerate(tables):
                    prompt += f"Table {i+1} (html):\n{t}\n\n"

            prompt += (
                "YOUR TASK:\n"
                "Generate a comprehensive, searchable description that covers:\n"
                "1) Key facts, numbers, and data points from text and tables\n"
                "2) Main topics and concepts discussed\n"
                "3) Questions this content could answer\n"
                "4) Visual content analysis (what images/charts convey)\n"
                "5) Alternative search terms users might use\n\n"
                "Make it detailed and searchable — prioritize findability over extreme brevity.\n\n"
                "SEARCHABLE DESCRIPTION:\n"
            )

            message = HumanMessage(content=[{"type": "text", "text": prompt}])

            # If there are image refs, include short caption about images (we don't embed binary in messages)
            # Some SDKs require different handling for images; here we just append the refs for context
            if image_refs:
                img_block = "\n".join([f"- image: {p}" for p in image_refs])
                # append image refs to prompt in a short form
                message.content.append({"type": "text", "text": f"Images saved at:\n{img_block}\n\n"})

            response = model.invoke([message])
            # response.content can be str or structured; fallback robust parsing
            if hasattr(response, "content"):
                return response.content if isinstance(response.content, str) else str(response.content)
            return str(response)
        except Exception as e:
            self.logger.error("LLM summary generation failed: " + str(e))
            self.logger.debug(traceback.format_exc())
            # fallback: return a truncated preview so embeddings still get useful signal
            return text[:2000]

    # ---------------------
    # Content separation
    # ---------------------
    def _separate_content_type(self, chunk: Document) -> dict:
        """
        Inspects the chunk Document and extracts:
          - text
          - tables (as html strings)
          - images (as base64 strings)
        Assumes chunk.metadata may contain 'orig_elements' as in original code; otherwise treats entire page_content as text.
        """
        content_data = {"text": chunk.page_content or "", "tables": [], "images": []}

        try:
            # Source-specific original elements (optional)
            orig = chunk.metadata.get("orig_elements") if chunk.metadata else None
            if orig:
                for element in orig:
                    element_type = type(element).__name__
                    if element_type == "Table":
                        # prefer html stored in metadata, otherwise fallback to element.text
                        table_html = getattr(element.metadata, "text_as_html", None) or getattr(element, "text", "")
                        content_data["tables"].append(table_html)
                    elif element_type == "Image":
                        # expect base64 under element.metadata.image_base64
                        img_b64 = getattr(getattr(element, "metadata", None), "image_base64", None)
                        if img_b64:
                            content_data["images"].append(img_b64)
        except Exception:
            self.logger.debug("Error while separating content types; treating chunk as plain text", exc_info=True)

        # final normalization
        return content_data

    # ---------------------
    # Main pipeline method
    # ---------------------
    def summary(self, chunk_documents: List[Document], file_name: str, do_summarize: bool = True) -> List[Document]:
        """
        Convert chunk_documents -> list[Document] ready for embedding.
        Each returned Document includes compact metadata (see class docstring).
        """
        self.logger.info(f"Summarizer: processing {len(chunk_documents)} chunks for {file_name}")
        out_docs: List[Document] = []
        try:
            # We expect chunk_documents have metadata with file_hash & chunk_index
            for doc in chunk_documents:
                # --- inside Summarizer.summary loop (for each chunk Document) ---
                text = (doc.page_content or "").strip()
                metadata = dict(doc.metadata or {})
                file_hash = metadata.get("file_hash") or "unknown_hash"
                chunk_index = int(metadata.get("chunk_index", 0))

                # 1) separate tables/images if present
                content_data = self._separate_content_type(doc)

                # 2) persist original text and heavy artifacts (best-effort)
                original_ref = self._persist_original_text(file_hash, chunk_index, text)

                images_paths = []
                tables_paths = []
                for i, tbl in enumerate(content_data.get("tables", [])):
                    tables_paths.append(self._persist_table_html(file_hash, chunk_index, tbl, i))
                for i, img_b64 in enumerate(content_data.get("images", [])):
                    images_paths.append(self._persist_image(file_hash, chunk_index, img_b64, i))

                # 3) create summary_text (AI-enhanced if appropriate, otherwise fallback/truncated)
                if do_summarize and (len(text) > 1200 or content_data.get("tables") or content_data.get("images")):
                    try:
                        summary_text = self.create_ai_enhanced_summary(text, content_data.get("tables", []), images_paths)
                    except Exception as e:
                        self.logger.warning(f"AI summary failed for {file_hash}:{chunk_index}: {e}")
                        # fallback: truncated original text for embedding usefulness
                        summary_text = text[:2000]
                else:
                    # no summarization requested or short text -> use original text (or a trimmed version)
                    summary_text = text if len(text) <= 2000 else text[:2000]

                # 4) build primitive-only metadata
                out_meta = {
                    "file_name": file_name,
                    "file_hash": file_hash,
                    "chunk_index": chunk_index,
                    "original_ref": str(original_ref),                    # string path
                    "images_count": len(images_paths),                    # int
                    "tables_count": len(tables_paths),                    # int
                    "first_image_ref": images_paths[0] if images_paths else None,  # str or None
                    "first_table_ref": tables_paths[0] if tables_paths else None,  # str or None
                    "original_preview": (text[:300] if text else ""),     # short string
                    "original_length": len(text or "")
                }

                # 5) append Document ready for EmbedStore.upsert_documents()
                out_docs.append(Document(page_content=summary_text, metadata=out_meta))


            self.logger.info(f"Summarizer: produced {len(out_docs)} documents for {file_name}")
            return out_docs

        except Exception as e:
            self.logger.error(f"Error while summarizing chunks: {e}")
            self.logger.debug(traceback.format_exc())
            raise
