from .db_client import ChromaClient

class ChromaManager:
    def __init__(self, chroma_client:ChromaClient=None):
        if chroma_client is None:
            chroma_client = ChromaClient()
        self.perm_collection = chroma_client.get_permanent_collection()
        self.temp_collection = chroma_client.get_temporary_collection()

    def add_to_permanent(self, ids:list, chunks:list, embeddings:list[list]):
        self.perm_collection.add(ids=ids, documents=chunks, embeddings=embeddings)

    def add_to_temporary(self, ids:list, chunks:list, embeddings:list[list]):
        self.temp_collection.add(ids=ids, documents=chunks, embeddings=embeddings)

    def query_permanent(self, permanent_query_embeddings:list, top_k:int=5):
        permanent_query_results = self.perm_collection.query(
            query_embeddings=permanent_query_embeddings, 
            n_results=top_k)
        return permanent_query_results

    def query_temporary(self, temporary_query_embeddings:list, top_k:int=5):
        temporary_query_results = self.temp_collection.query(
            query_embeddings=temporary_query_embeddings,
            n_results=top_k)
        return temporary_query_results

        
