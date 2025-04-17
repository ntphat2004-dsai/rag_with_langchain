from models.embed_model import load_embedding_model
from utils import read_vector_store, from_config
from models.api_model import load_api_model

PROMPT = """
Use the context to answer the question.

Context: {context}

Question: {question}
"""

def main():
    # Get paths from json file
    embedding_model_name = from_config('embedding_model_name')
    vector_store_path = from_config('vector_store_path')
    llm_name = from_config('llm_name')

    # Load embedding model
    embedding_model = load_embedding_model(embedding_model_name)

    # Load vector store
    vector_store = read_vector_store(vector_store_path, embedding_model)

    # Load LLM
    llm = load_api_model(llm_name)

    # QA
    question = input('Enter your question: ')
    retrieved_docs = vector_store.similarity_search(
        query=question, 
        k=1
    )
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    if not context:
        print("No relevant context found! Cannot query model.")
    else:
        response = llm.generate_content(PROMPT.format(context=context, question=question))
        print(response.candidates[0].content.parts[0].text.strip())

    
if __name__ == "__main__":
    main()


