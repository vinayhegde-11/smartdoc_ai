import chromadb
import os

class ChromaClient:
    def __init__(self,db_path:str="vector_db"):
        os.makedirs(db_path,exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(db_path)
        self.permanent_collection = self.chroma_client.get_or_create_collection(name="permanent")
        self.temporory_collection = self.chroma_client.get_or_create_collection(name="temporary")

    def get_permanent_collection(self):
        return self.permanent_collection
    
    def get_temporary_collection(self):
        return self.temporory_collection