from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

PERMANENT_CHUNK_PATH = ""
TEMP_CHUNK_PATH = ""

def create_chunks(text:str,is_permanent:bool=False):
    # ✅ Split into overlapping chunks
    text = text.replace("\n"," ")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # ~1000 characters per chunk
        chunk_overlap=100,     # ensures ~100-word overlap
        length_function=len,   # based on character count
    )
    chunks = text_splitter.split_text(text)
    return chunks