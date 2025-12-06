import streamlit as st
import os
import shutil
from ingestion.ingestion_orchestration import IngestionPipeline
from retrieval.retrieval_orchestration import RetrievalPipeline
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
from config.models import LLM_MODEL, LLM_TEMPERATURE, DEFAULT_TOP_K, TEMP_UPLOADS_ST_DIR

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Pipelines (Cached)
@st.cache_resource
def get_pipelines():
    return IngestionPipeline(), RetrievalPipeline()

ingestion_pipeline, retrieval_pipeline = get_pipelines()

# Initialize Model (Cached)
@st.cache_resource
def get_model():
    return ChatVertexAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE  # Lower temperature for higher faithfulness
    )

model = get_model()

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = "default"

# Sidebar
with st.sidebar:
    st.title("🤖 SmartDoc AI")
    
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf", "txt", "md"])
    
    if uploaded_file:
        if st.button("Ingest Document"):
            with st.spinner("Ingesting..."):
                try:
                    # Save temp file
                    temp_dir = TEMP_UPLOADS_ST_DIR
                    os.makedirs(temp_dir, exist_ok=True)
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Run ingestion
                    result = ingestion_pipeline.run_ingestion(file_path)
                    st.success(f"Ingested {result['file_name']} ({result['num_chunks']} chunks)")
                    
                    # Cleanup
                    os.remove(file_path)
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()
    
    st.header("Session Management")
    if st.button("New Session"):
        st.session_state.messages = []
        st.rerun()
        
    if st.button("Clear History"):
        st.session_state.messages = []
        st.rerun()

# Main Chat Interface
st.header("Chat with your Documents")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # 1. Retrieve (increased top_k)
            retrieval_result = retrieval_pipeline.run_retrieval(query=prompt, top_k=DEFAULT_TOP_K)
            retrieved_docs = retrieval_result.get("results", [])
            
            context_str = "\n".join([f"[{i+1}] {doc['text']}" for i, doc in enumerate(retrieved_docs)])
            
            # 2. Construct Prompt
            system_prompt = f"""You are a precise document assistant. Follow these rules strictly:
            1. Answer ONLY using information from the context below
            2. Cite sources using [1], [2], etc. when making statements
            3. If the context doesn't contain the answer, respond: "I don't have enough information to answer that."
            4. Do NOT use external knowledge or make assumptions beyond what's in the context
            5. Quote relevant parts of the context when possible
            
            Context:
            {context_str}
            """
            
            messages = [SystemMessage(content=system_prompt)]
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
            
            # Stream response
            for chunk in model.stream(messages):
                full_response += chunk.content
                message_placeholder.markdown(full_response + "▌")
                
            message_placeholder.markdown(full_response)
            
            # Add assistant message to state
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Error: {e}")

