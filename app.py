import streamlit as st
from datetime import datetime
from models.embed_model import load_embedding_model
from utils import read_vector_store, from_config
from models.api_model import load_api_model

PROMPT = """
You are a helpful assistant. Use ONLY the provided context to answer the user's question.
If the answer cannot be found within the context, reply with "I don't know based on the provided information."

Context:
{context}

Question:
{question}

Answer:
"""

@st.cache_resource
def load_components():
    embedding_model_name = from_config('embedding_model_name')
    vector_store_path = from_config('vector_store_path')
    llm_name = from_config('llm_name')

    if not (embedding_model_name and vector_store_path and llm_name):
        raise ValueError("Missing configuration. Please check config.json.")

    embedding_model = load_embedding_model(embedding_model_name)
    vector_store = read_vector_store(vector_store_path, embedding_model)
    llm = load_api_model(llm_name)

    return vector_store, llm

def main():
    st.set_page_config(page_title="QA Bot (API Version)")
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()

    st.title("QA Bot with Streamlit + API Model")
    st.caption("Powered by LangChain + Streamlit + API Model")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Load components
    vector_store, llm = load_components()

    # Display chat history
    for msg in st.session_state.messages:
        timestamp = f" *(at {msg['timestamp']})*" if "timestamp" in msg else ""
        st.chat_message(msg["role"]).markdown(msg["content"] + timestamp)

    if not st.session_state.messages:
        st.info("Ask me anything!")
        st.warning("This is a demo. Please do not share sensitive information.")

    user_input = st.chat_input("Enter your question here:")

    if user_input:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        st.chat_message("user").markdown(user_input)

        with st.spinner("Bot is thinking..."):
            try:
                retrieved_docs = vector_store.similarity_search(query=user_input, k=1)

                if not retrieved_docs:
                    answer = "No relevant context found! Cannot query model."
                else:
                    context = "\n".join(doc.page_content for doc in retrieved_docs)
                    response = llm.generate_content(PROMPT.format(context=context, question=user_input))
                    answer = response.candidates[0].content.parts[0].text.strip()

            except Exception as e:
                answer = f"An error occurred: {e}"

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        st.chat_message("assistant").markdown(answer)

    # Limit chat history size
    MAX_HISTORY = 50
    if len(st.session_state.messages) > MAX_HISTORY:
        st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]

if __name__ == "__main__":
    main()