from .db_client import ChromaClient
from rank_bm25 import BM25Okapi

class ChromaManager:
    def __init__(self, chroma_client:ChromaClient=None):
        if chroma_client is None:
            chroma_client = ChromaClient()
        self.perm_collection = chroma_client.get_permanent_collection()
        self.temp_collection = chroma_client.get_temporary_collection()
        self.perm_docs = []
        self.temp_docs = []
        self.perm_bm25 = None
        self.temp_bm25 = None


    def add_to_db(self, ids:list, chunks:list, embeddings:list[list], is_permanent:bool=False):
        if is_permanent:
            self.perm_collection.add(ids=ids, documents=chunks, embeddings=embeddings)
            self.perm_docs.extend(chunks)
            tokenized = [doc.split() for doc in self.perm_docs]
            self.perm_bm25 = BM25Okapi(tokenized)
        else:
            self.temp_collection.add(ids=ids, documents=chunks, embeddings=embeddings)
            self.temp_docs.extend(chunks)
            tokenized = [doc.split() for doc in self.temp_docs]
            self.temp_bm25 = BM25Okapi(tokenized)

    def query_db(self, query_embeddings:list, is_permanent:bool=False, top_k:int=5):
        if is_permanent:
            query_results = self.perm_collection.query(
                query_embeddings=query_embeddings, 
                n_results=top_k)
        else:
            query_results = self.temp_collection.query(
                query_embeddings=query_embeddings, 
                n_results=top_k)
        return query_results
        
    def bm25_permanent(self, query: str, top_k: int = 5):
        if not self.perm_bm25:
            return []
        tokenized_query = query.split()
        scores = self.perm_bm25.get_scores(tokenized_query)
        top_idxs = scores.argsort()[-top_k:][::-1]
        return [self.perm_docs[i] for i in top_idxs]

    def bm25_temporary(self, query: str, top_k: int = 5):
        if not self.temp_bm25:
            return []
        tokenized_query = query.split()
        scores = self.temp_bm25.get_scores(tokenized_query)
        top_idxs = scores.argsort()[-top_k:][::-1]
        return [self.temp_docs[i] for i in top_idxs]
    
    def hybrid_query(self, query: str, query_embeddings: list, top_k: int = 5, alpha: float = 0.5, is_permanent: bool = False):
        """
        Automatic hybrid retrieval:
        - Chooses permanent or temporary DB based on `use_temp`
        - Returns a merged list of dense + BM25 results
        """
        dense_results = self.query_db(query_embeddings, is_permanent, top_k=top_k)["documents"]
        if is_permanent:
            sparse_results = self.bm25_permanent(query, top_k=top_k)
        else:
            sparse_results = self.bm25_temporary(query, top_k=top_k)

        n_dense = int(top_k * alpha)
        n_sparse = top_k - n_dense
        return dense_results[:n_dense] + sparse_results[:n_sparse]