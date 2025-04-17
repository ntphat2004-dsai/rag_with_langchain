from langchain_huggingface import HuggingFaceEmbeddings

# Load embedding model
def load_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    """
    Load the Hugging Face embedding model.

    :param model_name: Name of the embedding model.
    :return: Loaded embedding model.
    """
    embedding_model = None
    print(f'Loading embedding model {model_name}...')
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    # Check if the embedding model is loaded successfully
    if embedding_model is None:
        raise ValueError(f"Failed to load embedding model: {model_name}")
    
    print('Embedding model loaded successfully!')
    return embedding_model