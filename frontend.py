# ## code1 ##
# import streamlit as st
# import os
# from file_ops.converter import convert_file_to_markdown
# from file_ops.generate import create_chunks, generate_embeddings
# from db_ops.db_client import ChromaClient
# from db_ops.db_manager import ChromaManager
# from langchain.memory import ConversationSummaryMemory
# from langchain.chains import ConversationChain
# from langchain_openai import ChatOpenAI
# from config_loader import load_config

# # ----------------- CONFIG & BACKEND SETUP -----------------
# # Load configuration and initialize backend services.
# config = load_config()
# dbc = ChromaClient()
# dbm = ChromaManager()

# OPENAROUTER_API_KEY = config['OPENAROUTER_API_KEY']
# MODEL_NAME = config['MODEL_NAME']
# REQUEST_URL = config['REQUEST_URL']

# llm = ChatOpenAI(
#     api_key=OPENAROUTER_API_KEY,
#     base_url=REQUEST_URL,
#     model=MODEL_NAME
# )
# memory = ConversationSummaryMemory(llm=llm)
# conversation = ConversationChain(
#     llm=llm,
#     memory=memory,
#     verbose=True
# )

# # ----------------- STREAMLIT UI SETUP -----------------
# st.set_page_config(page_title="SmartDoc AI", page_icon="🤖", layout="wide")
# st.title("SmartDoc AI Assistant")

# # Sidebar for file upload
# with st.sidebar:
#     st.header("Upload Document")
#     uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

#     if uploaded_file:
#         is_permanent = st.checkbox("Save permanently?", value=False)
#         filename = os.path.basename(uploaded_file.name).split('.')[0]
        
#         with st.spinner("Processing document..."):
#             # Ensure the document processing functions are correctly defined and return expected types.
#             markdown_output = convert_file_to_markdown(uploaded_file, is_permanent=is_permanent)
#             chunks = create_chunks(markdown_output)
#             embeddings = generate_embeddings(chunks)
#             ids = [f"{filename}_{i+1}" for i in range(len(chunks))]
#             dbm.add_to_db(ids, chunks, embeddings, is_permanent)
        
#         st.success("Document processed and added to DB!")

# # ----------------- CHAT INTERFACE -----------------
# # Initialize chat history in session state
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     # Use Streamlit's built-in chat elements for a clean look
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # User input at the bottom of the page
# if prompt := st.chat_input("Ask a question about your documents:"):
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})
    
#     # Display user message in chat message container
#     with st.chat_message("user"):
#         st.markdown(prompt)
    
#     # Generate and display assistant response
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             # Embed query & search DB
#             user_embedding = generate_embeddings(prompt)
#             query_res = dbm.hybrid_query(prompt, user_embedding, top_k=5, alpha=0.5, is_permanent=False)
            
#             # Formulate prompt with retrieved documents
#             qna_prompt = f"""
#             Answer the following question based on the retrieved documents.
#             Here is the Question -> {prompt} and the documents -> {query_res}
#             """
            
#             # Generate answer using conversation chain
#             response = conversation.predict(input=qna_prompt)
            
#             # Display the generated response
#             st.markdown(response)
            
#             # Add assistant response to chat history
#             st.session_state.messages.append({"role": "assistant", "content": response})

## code 2 ##
import streamlit as st
import os
from streamlit_option_menu import option_menu
from file_ops.converter import convert_file_to_markdown
from file_ops.generate import create_chunks, generate_embeddings
from db_ops.db_client import ChromaClient
from db_ops.db_manager import ChromaManager
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from config_loader import load_config
from typing import List, Dict

# ----------------- CONFIG & BACKEND SETUP -----------------
# Load configuration and initialize backend services.
config = load_config()
dbc = ChromaClient()
dbm = ChromaManager()

OPENAROUTER_API_KEY = config['OPENAROUTER_API_KEY']
MODEL_NAME = config['MODEL_NAME']
REQUEST_URL = config['REQUEST_URL']

llm = ChatOpenAI(
    api_key=OPENAROUTER_API_KEY,
    base_url=REQUEST_URL,
    model=MODEL_NAME
)
memory = ConversationSummaryMemory(llm=llm)
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# ----------------- STREAMLIT UI SETUP -----------------
st.set_page_config(page_title="SmartDoc AI", page_icon="🤖", layout="wide")

# Initialize session state variables at the very top
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "page" not in st.session_state:
    st.session_state.page = "Chat"

# ----------------- HELPER FUNCTIONS -----------------
def display_chat_history(messages: List[Dict]):
    """Displays messages from a given list."""
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def clear_chat_session():
    """Saves the current chat to history and starts a new session."""
    if st.session_state.messages:
        st.session_state.history.append(st.session_state.messages.copy())
    st.session_state.messages = []
    st.rerun()

def process_document(uploaded_file, is_permanent: bool):
    """
    Handles the end-to-end processing of a new document.
    Converts, chunks, embeds, and adds it to the database.
    """
    try:
        filename = os.path.basename(uploaded_file.name).split('.')[0]
        
        with st.spinner("1/4: Converting file to markdown..."):
            markdown_output = convert_file_to_markdown(uploaded_file, is_permanent=is_permanent)
        
        with st.spinner("2/4: Creating text chunks..."):
            chunks = create_chunks(markdown_output)
        
        with st.spinner("3/4: Generating embeddings..."):
            embeddings = generate_embeddings(chunks)
        
        with st.spinner("4/4: Adding data to the database..."):
            ids = [f"{filename}_{i+1}" for i in range(len(chunks))]
            dbm.add_to_db(ids, chunks, embeddings, is_permanent)
        
        st.success("Document processed and added to DB!")
    except Exception as e:
        st.error(f"An error occurred during document processing: {e}")
        st.error("Please try uploading the file again.")

def get_assistant_response(prompt: str) -> str:
    """
    Retrieves and generates a response from the AI assistant.
    Performs hybrid search and uses a refined prompt for the LLM.
    """
    try:
        user_embedding = generate_embeddings(prompt)
        retrieved_documents = dbm.hybrid_query(
            prompt, 
            user_embedding, 
            top_k=5, 
            alpha=0.5, 
            is_permanent=False
        )
        
        qna_prompt = f"""
        You are an expert Q&A assistant. Your task is to provide a concise and accurate answer
        to the user's question based ONLY on the following documents.
        If the answer is not contained in the documents, state that you cannot find the answer.
        Do not add any external information.

        Question: {prompt}

        Documents:
        {retrieved_documents}

        Answer:
        """
        
        response = conversation.predict(input=qna_prompt)
        return response
    except Exception as e:
        st.error(f"An error occurred while generating the response: {e}")
        return "Sorry, I'm having trouble processing your request right now. Please try again later."


# Sidebar UI using streamlit-option-menu
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Chat", "Upload", "History"],
        icons=["chat-fill", "file-earmark-arrow-up-fill", "clock-history"],
        default_index=0,
    )

st.title("SmartDoc AI Assistant")

if selected == "Chat":
    # Main chat display area
    chat_display_area = st.container()

    with chat_display_area:
        display_chat_history(st.session_state.messages)

    # Use a fixed-position container for the input bar
    st.markdown(
        """
        <style>
        .st-emotion-cache-1av54d4 {
            position: fixed;
            bottom: 0;
            width: 80%; /* Adjust as needed */
            z-index: 999;
            background-color: var(--background-color);
            padding: 1rem;
            box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Input section placed at the bottom
    with st.container():
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            if prompt := st.chat_input("Ask a question about your documents:"):
                if prompt.strip().lower() == "exit":
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    with st.chat_message("assistant"):
                        st.markdown("Goodbye! The chat session has ended. To start a new one, type a new message or click 'New Chat'.")
                    clear_chat_session()
                else:
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response = get_assistant_response(prompt)
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
        with col2:
            if st.button("End Chat"):
                clear_chat_session()

elif selected == "Upload":
    st.subheader("Upload a Document to Analyze")
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload a PDF", 
            type="pdf", 
            label_visibility="collapsed"
        )
    with col2:
        is_permanent = st.checkbox("Save permanently?", value=False)
        upload_button = st.button("Upload Document")
        
    if uploaded_file and upload_button:
        process_document(uploaded_file, is_permanent)

elif selected == "History":
    st.subheader("Previous Conversations")
    if not st.session_state.history:
        st.info("No previous conversations found.")
    else:
        for i, conversation_history in enumerate(st.session_state.history):
            with st.expander(f"Conversation {i + 1}"):
                display_chat_history(conversation_history)