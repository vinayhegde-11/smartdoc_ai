from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')

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

def generate_embeddings(chunks,is_permanent:bool=False):
    embeddings = model.encode(chunks)
    return embeddings