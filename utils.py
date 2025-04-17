from langchain_chroma import Chroma
import json

def from_config(name):
    result = None
    with open('config.json', 'r') as f:
        config = json.load(f)
    result = config.get(name)
    if not result:
        print(f'Missed {name} in config.json!')
    return result

# Read vector store
def read_vector_store(vector_store_path: str, embedding_model) -> Chroma:
    # Load the vector store
    print(f"Loading vector store from {vector_store_path}...")
    vector_store = Chroma(
        persist_directory=vector_store_path,
        embedding_function=embedding_model
    )
    
    # Check if vector store is loaded successfully
    if vector_store is None:
        raise ValueError(f"Failed to load vector store from {vector_store_path}")    
     
    print('Vectore store loaded successfully!')
    return vector_store