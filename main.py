from ingestion.ingestion_orchestration import IngestionPipeline
from retrieval.retrieval_orchestration import RetrievalPipeline
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from config.models import LLM_MODEL, LLM_TEMPERATURE

load_dotenv()

if __name__ == "__main__":
    # Ingest documents
    # pipeline1 = IngestionPipeline()
    # pipeline1.run_ingestion("samples/transformers.pdf")
    
    # Retrieve documents
    pipeline2 = RetrievalPipeline()
    query = "How many attention heads did the original Transformer use?"
    retrieved_docs = pipeline2.run_retrieval(query=query)
    
    # Initialize Vertex AI model
    model = ChatVertexAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE
    )
    
    # Format the prompt with retrieved documents
    prompt = f"""
    Based on the following documents, answer the question. If answer is not found, respond with "Information not available in the documents."
    Question: {query}
    Documents: {retrieved_docs}
    """
    
    # Get response from Vertex AI
    message = HumanMessage(content=prompt)
    # response = model.invoke([message])
    
    # print(response.content)
    stream = model.stream([message])
    for chunk in stream:
        print(chunk.content, end="", flush=True)
    print()