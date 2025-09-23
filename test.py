# ADD EXISTING FILE OR UPLOAD NEW_FILE
from file_ops.converter import convert_file_to_markdown
from file_ops.generate import create_chunks, generate_embeddings
from db_ops.db_client import ChromaClient
from db_ops.db_manager import ChromaManager
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from config_loader import load_config
import os

config = load_config()
dbc = ChromaClient()
dbm = ChromaManager()

OPENAROUTER_API_KEY = config['OPENAROUTER_API_KEY']
MODEL_NAME = config['MODEL_NAME']
REQUEST_URL = config['REQUEST_URL']


file_path = "samples/Football_Basics.pdf"
is_permanent=False
filename = os.path.basename(file_path).split('.')[0]
markdown_output = convert_file_to_markdown(file_path,is_permanent=is_permanent)
chunks = create_chunks(markdown_output)
embeddings = generate_embeddings(chunks)
ids = [f"{filename}_{i+1}"for i in range(len(chunks))]
dbm.add_to_db(ids,chunks,embeddings,is_permanent)


llm = ChatOpenAI(
    api_key= OPENAROUTER_API_KEY,
    base_url= REQUEST_URL,
    model= MODEL_NAME
)

memory = ConversationSummaryMemory(llm=llm)
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

while True:
    user_q = input("Ask your question (or 'exit' to quit): ")
    if user_q.lower() == "exit":
        break
    
    # Embed query, search DB
    user_embedding = generate_embeddings(user_q)
    # query_res = dbm.query_permanent(user_embedding)["documents"]
    query_res = dbm.hybrid_query(user_q, user_embedding, top_k=5, alpha=0.5, is_permanent=is_permanent)


    qna_prompt = f"""
    Answer the following question based on the retrieved documents.
    Here is the Question -> {user_q} and the documents -> {query_res}
    """

    # Now memory keeps track of previous turns
    response = conversation.predict(input=qna_prompt)
    print("Assistant:", response)
