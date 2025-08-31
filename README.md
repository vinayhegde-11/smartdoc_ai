# smartdoc_ai

## Features & Workflow

- Uses a YAML config file (`config.yml`) for all major settings (API keys, model, storage paths, etc.)
- All configs loaded via `config_loader.py` for easy environment changes
- Modular file structure: `db_ops`, `file_ops`, `samples`, `vector_db`
- RAG (Retrieval Augmented Generation) pipeline: PDF → Markdown → Chunks → Embeddings → Vector DB → LLM
- Uses ChromaDB for vector storage and retrieval
- Integrates OpenRouter API via LangChain (`ChatOpenAI`)
- Conversation memory with `ConversationSummaryMemory` for multi-turn chat
- Schematic search and memory features

## Setup Instructions

1. **Create a Python virtual environment:**
	```bash
	python3 -m venv venv
	source venv/bin/activate
	```

2. **Install required packages:**
	```bash
	pip install -r python_requirements.txt
	```

3. **Configure your environment:**
	- Edit `config.yml` with your API keys, model, and paths.

4. **Run the main app:**
	```bash
	python test.py
	```

5. **Project Structure:**
	- `db_ops/`: Database client and manager scripts
	- `file_ops/`: File conversion, chunk generation, and embedding scripts
	- `samples/`: Sample files for testing
	- `vector_db/`: Vector database files
	- `config.yml`: Main configuration file
	- `config_loader.py`: Loads config for all modules
	- `python_requirements.txt`: Python dependencies
	- `test.py`: Test script
	- `README.md`: Project documentation