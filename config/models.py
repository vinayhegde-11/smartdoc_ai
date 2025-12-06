"""
Centralized configuration module for SmartDoc AI.
All model names and configuration parameters are loaded from environment variables.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==========================================
# LLM Configuration
# ==========================================
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1200"))

# Evaluation-specific LLM settings (lower temperature for deterministic results)
EVAL_LLM_TEMPERATURE = float(os.getenv("EVAL_LLM_TEMPERATURE", "0.2"))

# ==========================================
# Embedding Models
# ==========================================
VERTEX_EMBEDDING_MODEL = os.getenv("VERTEX_EMBEDDING_MODEL", "text-embedding-004")
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ==========================================
# Vector Store Configuration
# ==========================================
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "smartdoc")

# ==========================================
# Retrieval Configuration
# ==========================================
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "10"))
EVAL_TOP_K = int(os.getenv("EVAL_TOP_K", "15"))  # Higher for evaluation

# ==========================================
# Storage Paths
# ==========================================
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "./artifacts")
TEMP_UPLOADS_DIR = os.getenv("TEMP_UPLOADS_DIR", "temp_uploads")
TEMP_UPLOADS_ST_DIR = os.getenv("TEMP_UPLOADS_ST_DIR", "temp_uploads_st")
