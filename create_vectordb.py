from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma 
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from models.embed_model import load_embedding_model
import os
from utils import from_config

# Create vector store from PDF files in a directory
def create_vector_store(
        file_path: str, 
        vector_store_path: str,
        embedding_model_name: str, 
        chunk_size: int, 
        chunk_overlap: int
    ) -> Chroma:
    """
    Create a vector store from PDF files in a directory.

    :param file_path: Path to the directory containing PDF files.
    :param vector_store_path: Path to save the vector store.
    :param embedding_model_name: Name of the embedding model.
    :param chunk_size: Size of each text chunk.
    :param chunk_overlap: Overlap between chunks.
    :return: Chroma vector store.
    """
    # Check if the directory exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Directory {file_path} not found")

    print(f"Loading PDFs from: {file_path}")
    print(f"Saving vector store to: {vector_store_path}")
    print(f"Using embedding model: {embedding_model_name}")
    print(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")

    # Load documents
    loader = DirectoryLoader(
        path=file_path, 
        glob="*.pdf",           # Load only PDF files
        loader_cls=PyPDFLoader  # Load PDF files using PyPDFLoader
    )
    documents = loader.load()
    if not documents:
        raise ValueError("No PDF files found in the specified directory.")   
    
    print(f"Loaded {len(documents)} documents.")

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,            # Use len function to calculate chunk length
        is_separator_regex= False       # Do not use regex for separators
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    # Log some chunks to tracking off 
    for i, chunk in enumerate(chunks[:3]):
        print(f"Chunk {i+1}: {chunk.page_content[:200]}...")

    # Embeddings
    embedding_model = load_embedding_model(embedding_model_name)

    # Create vector store and auto-save
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=vector_store_path
    )
    print(f"Vector store created with {len(chunks)} documents and saved to: {vector_store_path}")
    return vector_store

if __name__ == "__main__":
    # Get parameters from config
    data_path = from_config("data_path")
    vector_store_path = from_config("vector_store_path")
    embedding_model_name = from_config("embedding_model_name")
    chunk_size = from_config("chunk_size")
    chunk_overlap = from_config("chunk_overlap")
    
    # Create output directory if not exists
    if not os.path.exists(vector_store_path):
        os.makedirs(vector_store_path)
    
    # Create vector store
    create_vector_store(data_path, vector_store_path, embedding_model_name, chunk_size, chunk_overlap)
