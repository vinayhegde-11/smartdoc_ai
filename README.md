# SmartDoc AI

## Table of Contents
- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Technical Concepts](#technical-concepts)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Challenges Faced](#challenges-faced)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Future Improvements](#future-improvements)

---

## Introduction

**SmartDoc AI** is a **production-ready document-oriented AI assistant** built with a **hybrid Retrieval-Augmented Generation (RAG) pipeline**. It processes multiple file types—including PDFs, DOCX, and Markdown—and enables **semantic search, keyword-based retrieval, and multi-turn conversational interactions** over your documents. The project emphasizes **modularity, scalability, and enterprise-grade evaluation** with comprehensive RAG metrics.

---

## Project Overview

SmartDoc AI converts raw documents into structured, searchable data for **LLM-based reasoning**. Its workflow spans **document ingestion, multimodal processing, chunking, embedding generation, vector storage, intelligent querying, and comprehensive evaluation**. Using **ChromaDB** for vector storage, **LangChain** for LLM integration, and **Ragas** for RAG evaluation, the system delivers **accurate, context-aware responses** with measurable quality metrics.

**Key Features:**
- 🚀 **Dual Interface**: FastAPI REST API + Streamlit web app
- 📄 **Multi-format Ingestion**: PDF, DOCX, Markdown with multimodal support (tables, images)
- 🧠 **Advanced RAG**: Hybrid semantic + keyword search with reranking
- 💬 **Multi-turn Conversations**: Session-based chat with context retention
- 📊 **RAG Evaluation Dashboard**: Real-time metrics (Context Precision, Recall, Faithfulness, Relevancy)
- ⚙️ **Centralized Configuration**: Environment-based config with .env support
- 🎨 **Modern UI**: Dark-themed responsive interface with unified styling
- 🔍 **Smart Suggestions**: Auto-generated questions based on document content
- 🏗️ **Modular Architecture**: Clean separation of concerns (ingestion, retrieval, evaluation)

---

## Technical Concepts

### 1. Retrieval-Augmented Generation (RAG)
RAG enhances LLM responses by **retrieving relevant information from a knowledge base** before generating answers. SmartDoc AI implements a **Hybrid RAG system** that combines multiple retrieval strategies for superior accuracy.

### 2. Hybrid Retrieval Strategy

**Semantic Search:**
- Uses embeddings (HuggingFace `all-MiniLM-L6-v2`) for context-aware retrieval
- Captures synonyms, related concepts, and semantic meaning
- Powered by ChromaDB vector database

**Keyword Retrieval (BM25):**
- Fast exact-term matching
- Complements semantic search for precise queries
- Available as optional enhancement

**Reranking:**
- FlashRank reranker prioritizes most relevant results
- Improves precision of final context selection

### 3. Multimodal Document Processing
- **AI-Enhanced Summarization**: Gemini 2.0 Flash generates searchable descriptions
- **Table Extraction**: Preserves table structure for accurate retrieval
- **Image Analysis**: Stores and references visual content
- **Artifact Management**: Organized storage of original chunks, tables, and images

### 4. RAG Evaluation with Ragas
SmartDoc AI implements **industry-standard RAG metrics** using the Ragas framework:

- **Context Precision**: Relevance of retrieved contexts to the question
- **Context Recall**: Coverage of ground truth in retrieved contexts
- **Faithfulness**: Accuracy of generated answer to retrieved context
- **Answer Relevancy**: Relevance of generated answer to the question

Real-time visualization through interactive dashboard with Chart.js.

---

## Key Features

### Document Ingestion
- **Supported Formats**: PDF, DOCX, TXT, Markdown
- **Multimodal Processing**: Tables, images, and complex layouts
- **Chunking Strategy**: Configurable chunk size with overlap
- **Deduplication**: Automatic detection of re-ingested documents

### Retrieval & Generation
- **Hybrid Search**: Combines semantic similarity and keyword matching
- **Reranking**: FlashRank post-retrieval optimization
- **Streaming Responses**: Real-time answer generation
- **Citation Support**: Source references in responses
- **Session Management**: Multi-turn conversation with history

### Evaluation & Monitoring
- **Automated Testing**: Batch evaluation on test datasets
- **Visual Dashboard**: Interactive charts and metrics cards
- **Performance Tracking**: Retrieval and generation latency monitoring
- **Trend Analysis**: Historical comparison of metric improvements
- **Export Capabilities**: JSON export of evaluation results

### User Experience
- **Chat Interface**: Clean, modern dark-themed UI
- **Suggested Questions**: AI-generated queries based on document content
- **File Upload**: Drag-and-drop document ingestion
- **Session Controls**: New/end session management
- **Responsive Design**: Mobile-friendly layouts

---

## Architecture

```
SmartDoc AI
│
├── Ingestion Pipeline
│   ├── Document Loader (Unstructured)
│   ├── Chunker (Semantic splitting)
│   ├── Summarizer (Gemini 2.0 Flash)
│   └── Vector Store (ChromaDB + HuggingFace embeddings)
│
├── Retrieval Pipeline
│   ├── Hybrid Retriever (Semantic + BM25)
│   ├── Reranker (FlashRank)
│   └── Context Aggregation
│
├── Generation
│   ├── LLM Integration (Gemini 2.0 Flash via Vertex AI)
│   ├── Prompt Engineering
│   └── Streaming Response
│
├── Evaluation
│   ├── Test Dataset Management
│   ├── Ragas Metrics (Precision, Recall, Faithfulness, Relevancy)
│   └── Performance Monitoring
│
└── Interfaces
    ├── FastAPI (REST API)
    └── Streamlit (Web App)
```

**Technology Stack:**
- **LLM**: Google Gemini 2.0 Flash (Vertex AI)
- **Embeddings**: HuggingFace all-MiniLM-L6-v2 + Vertex AI text-embedding-004
- **Vector DB**: ChromaDB with persistent storage
- **Framework**: LangChain for orchestration
- **Backend**: FastAPI with async support
- **Frontend**: Streamlit + Custom HTML/CSS/JS
- **Evaluation**: Ragas framework
- **Document Processing**: Unstructured library

---

## Challenges Faced

### 1. LLM Platform Selection
- Migrated from **OpenRouter** to **Google Vertex AI**
- Chose **Gemini 2.0 Flash** for optimal speed/quality balance
- Implemented **temperature tuning** (0.2-0.3) for faithfulness

### 2. Multimodal Document Processing
- Handled **complex PDFs** with tables and images
- Implemented **AI-enhanced summarization** for better retrieval
- **Artifact separation**: Original content vs. searchable summaries

### 3. RAG Quality Optimization
- **Context window optimization**: Balanced chunk size vs. context preservation
- **Retrieval tuning**: Increased `top_k` from 5 → 10 → 15 for evaluation
- **Prompt engineering**: Reduced hallucinations through strict instruction prompts

### 4. Evaluation Integration
- **Async event loop conflicts**: ThreadPoolExecutor for Ragas execution
- **Metric parsing**: Handled EvaluationResult object structure
- **Performance optimization**: Batch processing with configurable workers

### 5. Configuration Management
- Externalized all **hardcoded model names** to environment variables
- Created **centralized config module** for maintainability
- Ensured **GitHub-safe** codebase with .env pattern

### 6. UI/UX Consistency
- **Unified styling** across chat and evaluation pages
- **Responsive design** for mobile/desktop
- **Accessibility**: Clear visual hierarchy and feedback

---

## Project Structure

```
smartdoc_ai/
├── config/
│   ├── __init__.py
│   └── models.py                 # Centralized configuration
├── ingestion/
│   ├── __init__.py
│   ├── loader.py                 # Document loading
│   ├── chunker.py                # Text chunking
│   ├── summarizer.py             # AI-enhanced summarization
│   ├── vector_store.py           # ChromaDB integration
│   └── ingestion_orchestration.py
├── retrieval/
│   ├── __init__.py
│   ├── retriever.py              # Hybrid search
│   └── retrieval_orchestration.py
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py              # Ragas metrics
│   └── test_dataset.py           # Test data management
├── static/
│   ├── index.html                # Chat interface
│   ├── eval.html                 # Evaluation dashboard
│   ├── style.css                 # Unified styling
│   ├── script.js                 # Chat logic
│   └── eval.js                   # Dashboard logic
├── utils/
│   └── logger.py                 # Logging utilities
├── api.py                        # FastAPI server
├── streamlit_app.py              # Streamlit web app
├── main.py                       # CLI entry point
├── .env.example                  # Environment template
├── pyproject.toml                # UV dependencies
└── README.md                     # This file
```

---

## Setup & Usage

### Prerequisites
- Python 3.10+
- Google Cloud account with Vertex AI enabled
- Service account credentials JSON

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/smartdoc_ai.git
cd smartdoc_ai
```

2. **Create virtual environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
# Using pip
pip install -r python_requirements.txt

# Or using uv (faster)
uv sync
```

4. **Configure environment:**
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your credentials
nano .env
```

Required environment variables:
```bash
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account.json
LLM_MODEL=gemini-2.0-flash
VERTEX_EMBEDDING_MODEL=text-embedding-004
# ... see .env.example for all options
```

### Running the Application

#### Option 1: FastAPI Server (Recommended)
```bash
uvicorn api:app --reload --loop asyncio
```
Access at `http://localhost:8000/static/index.html`

#### Option 2: Streamlit App
```bash
streamlit run streamlit_app.py
```
Access at `http://localhost:8501`

#### Option 3: CLI
```bash
python main.py
```

### Using the Application

1. **Upload Documents**: Drag and drop PDF/DOCX files
2. **Ask Questions**: Type queries in the chat interface
3. **View Evaluations**: Navigate to `/static/eval.html` for metrics
4. **Run Batch Evaluation**: Click "Run Evaluation" on dashboard

---

## Configuration

SmartDoc AI uses environment variables for all configuration. See `.env.example` for full options.

### Core Settings
```bash
# LLM Configuration
LLM_MODEL=gemini-2.0-flash          # Main LLM model
LLM_TEMPERATURE=0.3                 # Response creativity (0-1)
LLM_MAX_TOKENS=1200                 # Max response length
EVAL_LLM_TEMPERATURE=0.2            # Evaluation temperature

# Embeddings
VERTEX_EMBEDDING_MODEL=text-embedding-004  # Vertex AI embeddings
HF_EMBEDDING_MODEL=all-MiniLM-L6-v2       # HuggingFace embeddings

# Vector Store
VECTOR_DB_PATH=./vector_db          # ChromaDB storage path
COLLECTION_NAME=smartdoc            # Collection identifier

# Retrieval
DEFAULT_TOP_K=10                    # Results for chat
EVAL_TOP_K=15                       # Results for evaluation

# Paths
ARTIFACTS_DIR=./artifacts           # Document artifacts
TEMP_UPLOADS_DIR=temp_uploads       # Temporary storage
```

### Customization

To use different models:
```bash
# Switch to Gemini Pro
export LLM_MODEL="gemini-1.5-pro"

# Use custom embedding model
export HF_EMBEDDING_MODEL="sentence-transformers/all-mpnet-base-v2"
```
---

## Future Improvements

### Short-term
- [ ] Advanced filtering (by date, author, document type)
- [ ] PDF annotation and highlighting
- [ ] Export chat conversations

### Medium-term
- [ ] Offline LLM support (Ollama integration)
- [ ] Multi-language document processing
- [ ] Voice input/output
- [ ] Real-time collaboration features

### Long-term
- [ ] Fine-tuned embedding models for domain-specific use
- [ ] Graph-based knowledge extraction
- [ ] Automated document summarization
- [ ] Integration with cloud storage (Google Drive, Dropbox)

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ using Google Gemini, LangChain, and ChromaDB**