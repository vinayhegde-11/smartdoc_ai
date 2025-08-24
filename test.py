import os
from file_ops.converter import convert_file_to_markdown
from file_ops.generate_chunks import create_chunks
from file_ops.generate_embeddings import generate_embeddings
from db_ops.db_client import ChromaClient
from db_ops.db_manager import ChromaManager

dbc = ChromaClient()
dbm = ChromaManager()

# # Example usage:
file_path = "samples/story.pdf"
filename = os.path.basename(file_path).split('.')[0]

markdown_output = convert_file_to_markdown(file_path)
chunks = create_chunks(markdown_output)
embeddings = generate_embeddings(chunks)
ids = [f"{filename}_{i+1}"for i in range(len(chunks))]

dbm.add_to_permanent(ids,chunks,embeddings)

question = "What is the name of the giraffe in the story?"
question_embedding = generate_embeddings(question)

queryres = dbm.query_permanent(question_embedding)
print(queryres)