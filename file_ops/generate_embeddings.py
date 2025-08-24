from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')

PERMANENT_VE_PATH = ""
TEMP_VE_PATH = ""

def generate_embeddings(chunks,is_permanent:bool=False):
    embeddings = model.encode(chunks)
    return embeddings

# returns 