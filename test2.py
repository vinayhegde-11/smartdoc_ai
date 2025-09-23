from fastapi import FastAPI, UploadFile, File, Form
from typing import List
from db_ops.db_manager import ChromaManager
from file_ops.converter import convert_file_to_markdown
from file_ops.generate import create_chunks, generate_embeddings
from config_loader import load_config
import shutil, os

dbm = ChromaManager()
app = FastAPI()
config = load_config()
TEMP_STORAGE_PATH = config['TEMP_STORAGE_PATH']
os.makedirs(TEMP_STORAGE_PATH,exist_ok=True)

@app.post("/upload_files")
async def process_input(
    file: UploadFile = File(None),  # single input, multiple files allowed
    is_permanent: bool = Form(False)
):
    temp_file_path = os.path.join(TEMP_STORAGE_PATH,file.filename)
    with open(temp_file_path, "wb") as temp_file:
        shutil.copyfileobj(file.file,temp_file)

    main(file_path=temp_file_path, is_permanent=is_permanent)
    return {
        "Done": "Done",
        "file_name": file.filename,
        "file_content": file.file
    }

def main(file_path:str,is_permanent:bool):
    markdown_output = convert_file_to_markdown(file_path=file_path,is_permanent=is_permanent)
    print(markdown_output)
    # chunks = create_chunks(markdown_output)
    # embeddings = generate_embeddings(chunks)
    # ids = [f"{file_name}_{i+1}"for i in range(len(chunks))]
    # dbm.add_to_db(ids,chunks,embeddings,is_permanent)