import os
import uuid
import shutil
from typing import List, Dict, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from ingestion.ingestion_orchestration import IngestionPipeline
from retrieval.retrieval_orchestration import RetrievalPipeline
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from config.models import LLM_MODEL, LLM_TEMPERATURE, DEFAULT_TOP_K, TEMP_UPLOADS_DIR

from utils.logger import get_logger

load_dotenv()

logger = get_logger("api")

app = FastAPI(title="SmartDoc AI API")

# Mount static files
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Initialize pipelines
ingestion_pipeline = IngestionPipeline()
retrieval_pipeline = RetrievalPipeline()

# Initialize Vertex AI model
model = ChatVertexAI(
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE  # Lower temperature for higher faithfulness
)

# In-memory session store: session_id -> list of messages (dicts)
sessions: Dict[str, List[Dict[str, str]]] = {}

class ChatRequest(BaseModel):
    query: str
    session_id: str

class SessionResponse(BaseModel):
    session_id: str

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        temp_dir = TEMP_UPLOADS_DIR
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"File saved to {file_path}")
        
        # Run ingestion
        result = ingestion_pipeline.run_ingestion(file_path)
        
        # Clean up
        os.remove(file_path)
        
        return JSONResponse(content={"message": "Ingestion successful", "details": result})
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/session/new", response_model=SessionResponse)
async def new_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = []
    logger.info(f"New session created: {session_id}")
    return {"session_id": session_id}

@app.post("/session/end")
async def end_session(request: SessionResponse):
    session_id = request.session_id
    if session_id in sessions:
        del sessions[session_id]
        logger.info(f"Session ended: {session_id}")
    return {"message": "Session ended"}

@app.get("/history/{session_id}")
async def get_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]

@app.post("/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id
    query = request.query
    
    if session_id not in sessions:
        # Auto-create session if not exists (optional, but good for UX)
        sessions[session_id] = []
    
    # 1. Retrieve documents (increased top_k for better context)
    retrieval_result = retrieval_pipeline.run_retrieval(query=query, top_k=DEFAULT_TOP_K)
    retrieved_docs = retrieval_result.get("results", [])
    
    # 2. Construct Prompt
    # We include history in the prompt context or as messages
    history = sessions[session_id]
    
    # Format retrieved docs with numbering for citations
    context_str = "\n\n".join([f"Source [{i+1}]:\n{doc['text']}" for i, doc in enumerate(retrieved_docs)])
    
    system_prompt = f"""You are a highly precise document Q&A assistant. Your task is to answer questions STRICTLY based on the provided context.

CRITICAL RULES:
1. **Use ONLY the context below** - Do not use any external knowledge
2. **Be direct and specific** - Answer the exact question asked, nothing more
3. **Cite sources** - Use [1], [2], etc. to reference sources
4. **If unsure** - Say "I don't have enough information in the provided context to answer that question."
5. **Stay focused** - Don't add unnecessary information or context

CONTEXT:
{context_str}

Remember: Answer precisely and cite your sources!
"""
    
    messages = [SystemMessage(content=system_prompt)]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
            
    messages.append(HumanMessage(content=query))
    
    # Update history immediately with user query
    sessions[session_id].append({"role": "user", "content": query})
    
    async def generate():
        full_response = ""
        try:
            async for chunk in model.astream(messages):
                content = chunk.content
                if content:
                    full_response += content
                    yield content
            
            # Update history with full response
            sessions[session_id].append({"role": "assistant", "content": full_response})
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"Error generating response: {str(e)}"

    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/suggest_questions")
async def suggest_questions():
    try:
        # 1. Sample documents
        docs = ingestion_pipeline.embed_store.sample_documents(k=5)
        if not docs:
            return {"questions": ["What is this document about?", "Summarize the key points.", "Explain the main concepts."]}
        
        context_str = "\n".join([f"- {d.page_content[:500]}" for d in docs])
        
        # 2. Generate questions using LLM
        prompt = f"""Based on the following document excerpts, generate 3 short, interesting questions a user might ask about this content.
        Output ONLY the questions, one per line. Do not number them.
        
        Excerpts:
        {context_str}
        """
        
        response = model.invoke([HumanMessage(content=prompt)])
        questions = [q.strip() for q in response.content.split('\n') if q.strip()]
        
        # Fallback if model fails to generate valid list
        if not questions:
             questions = ["What is this document about?", "Summarize the key points.", "Explain the main concepts."]
             
        return {"questions": questions[:3]}
    except Exception as e:
        logger.error(f"Error generating suggestions: {e}")
        return {"questions": ["What is this document about?", "Summarize the key points.", "Explain the main concepts."]}

from evaluation.evaluator import RAGEvaluator
from evaluation.test_dataset import load_test_dataset

# Initialize evaluator (after pipelines are initialized)
evaluator = RAGEvaluator(retrieval_pipeline, ingestion_pipeline.embed_store)

@app.post("/evaluate")
async def run_evaluation():
    """Run batch evaluation on the test dataset."""
    try:
        # Load test set from file
        test_set = load_test_dataset()
        if not test_set:
            return JSONResponse(
                content={
                    "message": "No test set found. Please add questions to evaluation/sample_test_set.json",
                    "success": False
                },
                status_code=404
            )
        
        logger.info(f"Running evaluation with {len(test_set)} test cases")
        
        # Run evaluation
        results = evaluator.evaluate_batch(test_set)
        
        return JSONResponse(content={
            "success": True,
            "message": f"Evaluation completed for {len(test_set)} test cases",
            "results": results
        })
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evaluate/metrics")
async def get_evaluation_metrics():
    """Get evaluation metrics summary and history."""
    try:
        summary = evaluator.get_metrics_summary()
        return JSONResponse(content=summary)
    except Exception as e:
        logger.error(f"Error fetching metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate/single")
async def evaluate_single_query(
    question: str = Body(...),
    answer: str = Body(...),
    contexts: List[str] = Body(...),
    ground_truth: Optional[str] = Body(None)
):
    """Evaluate a single question-answer pair."""
    try:
        result = evaluator.evaluate_single(question, answer, contexts, ground_truth)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Single evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
