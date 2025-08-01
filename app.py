import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Init
st.set_page_config(page_title="Chatbot Aivancity", layout="wide")
st.title("🤖 Chatbot Aivancity (RAG + Groq)")

# Chargement du vectorstore (Chroma)
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
vectorstore = Chroma(persist_directory="chroma_store", embedding_function=embedding_model)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Chargement du LLM
llm = ChatGroq(model="llama3-8b-8192", api_key=st.secrets["GROQ_API_KEY"])

# Chaîne RAG
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Interface utilisateur
question = st.text_input("Pose ta question 👇")
if question:
    with st.spinner("Réflexion en cours..."):
        result = rag_chain.invoke({"query": question})
        st.success(result["result"])
