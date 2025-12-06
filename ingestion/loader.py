# ingestion/loader.py (core implementation)
import hashlib
import os
from unstructured.partition.pdf import partition_pdf
from utils.logger import get_logger
from typing import Tuple, List, Any

class DataLoader:
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    def _compute_file_hash(self, file_path: str) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def load_and_extract(self, file_path: str) -> Tuple[List[Any], str, str]:
        """
        Returns: (elements, file_hash, file_name)
        elements: list of unstructured elements (as returned by partition_pdf or other partitioners)
        """
        self.logger.info(f"Loading and extracting data from {file_path}")
        file_name = os.path.basename(file_path)
        try:
            # compute hash first
            file_hash = self._compute_file_hash(file_path)

            # route by extension (support docx/txt later as needed)
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".pdf":
                elements = partition_pdf(
                    filename=file_path,
                    strategy="fast",#"hi_res"
                    infer_table_structure=True,
                    extract_image_block_types=["Image"],
                    extract_image_block_to_payload=True,)
            else:
                # fallback or raise based on your supported types
                raise ValueError(f"Unsupported file extension: {ext}")

            self.logger.info(f"Extracted {len(elements)} elements from {file_name} (hash={file_hash[:8]})")
            return elements, file_hash, file_name
        except Exception as e:
            self.logger.error(f"Failed to load/extract {file_path}: {e}")
            raise