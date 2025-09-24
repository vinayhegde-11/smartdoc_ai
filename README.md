# SmartDoc AI

## Table of Contents
- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Technical Concepts](#technical-concepts)
- [Challenges Faced](#challenges-faced)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [Future Improvements](#future-improvements)

---

## Introduction

**SmartDoc AI** is a **document-oriented AI assistant** built with a **hybrid Retrieval-Augmented Generation (RAG) pipeline**. It processes multiple file types—including PDFs, DOCX, and Markdown—and enables **semantic search, keyword-based retrieval, and multi-turn conversational interactions** over your documents. The project emphasizes **modularity, scalability, and adaptability** to different LLM platforms.

---

## Project Overview

SmartDoc AI converts raw documents into structured, searchable data for **LLM-based reasoning**. Its workflow spans **document ingestion, format conversion, chunking, embedding generation, vector storage, and intelligent querying**. Using **ChromaDB** for vector storage and **LangChain** for LLM integration, the system delivers **accurate, context-aware responses** in multi-turn conversations.

**Key Features:**
- Multi-format document ingestion and conversion (PDF/DOCX → Markdown)  
- Chunking of large documents for efficient semantic retrieval  
- Embedding generation using a **single optimized embedding model**  
- Permanent and temporary vector storage for hybrid RAG approaches  
- Multi-turn conversation memory  
- Configurable via `config.yml`  
- Modular architecture with separate database and file operation modules  
- Streamlit-based frontend for easy user interaction  

---

## Technical Concepts

**1. Retrieval-Augmented Generation (RAG)**  
RAG enhances LLM responses by **retrieving relevant information from a knowledge base** before generating answers. SmartDoc AI evolved into a **Hybrid RAG system**, which provides more accurate and context-rich results by combining multiple retrieval strategies.

**2. Keyword Retrieval vs Semantic Search vs Hybrid RAG**  

- **Keyword Retrieval:**  
  - Matches exact terms in the document.  
  - Fast and straightforward.  
  - Limited understanding of context, may miss semantically relevant content.  

- **Semantic Search:**  
  - Uses embeddings and vector similarity to retrieve content.  
  - Captures context, synonyms, and semantic meaning.  
  - Slightly more computationally intensive.  

- **Hybrid RAG:**  
  - Combines **keyword retrieval** and **semantic search**.  
  - Ensures both precise matches and context-aware results.  
  - Provides more comprehensive, accurate answers than single-method RAG.  

**3. Document Processing Pipeline**  
- **Extraction & Conversion:** Converts PDFs and DOCX to Markdown to ensure clean text.  
- **Chunking:** Splits documents into manageable pieces for embeddings while preserving context.  
- **Embedding Generation:** Single embedding model converts text chunks into vectors.  
- **Vector Database:** ChromaDB stores embeddings for both permanent and temporary usage.  

---

## Challenges Faced

**1. LLM Platform Selection**  
- Evaluated **Ollama, LM Studio, and OpenRouter**.  
- Considered **ease of integration, inference speed, and local vs cloud capabilities**.  
- Chose **OpenRouter via LangChain** for smooth integration and robust performance.  

**2. Document Conversion**  
- Raw PDF and DOCX caused formatting inconsistencies.  
- Markdown conversion provided **cleaner text extraction and better chunking**.  
- Handled **complex layouts and tables** effectively.  

**3. Vector Storage & Retrieval**  
- Initial RAG system lacked flexibility.  
- Hybrid RAG enables **both keyword and semantic retrieval**, improving result relevance.  
- Permanent and temporary storage helps **manage session-based queries efficiently**.  

**4. Chunking & Context Preservation**  
- Balancing chunk size to **retain context without exceeding model limits**.  
- Optimizing overlap between chunks for **better semantic retrieval**.  
- Ensured **multi-turn conversation consistency** across chunks.  

**5. Embedding Model Selection**  
- Single optimized model chosen to balance **speed and semantic richness**.  
- Ensured embeddings work effectively with **hybrid RAG search**.  
- Tested for **accuracy and retrieval performance** on sample documents.  

---

## Project Structure

- ```db_ops/```: Database client and manager scripts  
- ```file_ops/```: File conversion, chunk generation, and embedding scripts  
- ```samples/```: Sample files for testing  
- ```vector_db/```: Vector database files  
- ```config.yml```: Main configuration file  
- ```config_loader.py```: Loads config for all modules  
- ```frontend.py```: Streamlit frontend for interacting with the assistant  
- ```python_requirements.txt```: Python dependencies  
- ```README.md```: Project documentation

---

## Setup & Usage

1. **Create a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```

2. **Install dependencies:**
```bash
pip install -r python_requirements.txt
```

3. **Configure the project:**
- Edit config.yml with your API keys, selected LLM model, and storage paths.

4. **Configure the project:**
- Run the frontend:
```bash
streamlit run frontend.py
```
- The Streamlit frontend will launch in your browser.
- Upload documents, ask queries, and get context-aware answers.

## Future Improvements

- Add more embedding models for comparative evaluation.
- Support offline LLM inference for local-first AI usage.
- Extend multi-language document processing.
- Enhance Streamlit frontend with voice input and richer UX.
- Enable real-time updates to the vector store when documents change.