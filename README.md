# smartdoc_ai

## PHASE-1: Work Completed

- Project workspace initialized
- Directory structure created: db_ops, file_ops, samples, vector_db
- Python virtual environment set up
- Initial scripts added: db_client.py, db_manager.py, converter.py, generate_chunks.py, generate_embeddings.py
- Sample file testing
- Vector database files created (chroma.sqlite3, data_level0.bin, etc.)
- Python requirements file added

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

3. **Run tests:**
	```bash
	python test.py
	```

4. **Project Structure:**
	- `db_ops/`: Database client and manager scripts
	- `file_ops/`: File conversion, chunk generation, and embedding scripts
	- `samples/`: Sample files for testing
	- `vector_db/`: Vector database files
	- `python_requirements.txt`: Python dependencies
	- `test.py`: Test script
	- `README.md`: Project documentation