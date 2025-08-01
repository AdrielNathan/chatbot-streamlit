
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Configuration de l'API (remplace par st.secrets ou os.environ en production)
import os
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Prompt personnalisé
CUSTOM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=""" 
    Vous êtes un assistant IA pour les étudiants d'Aivancity. Répondez de manière claire, concise et naturelle, 
    comme si vous parliez directement à l'étudiant. Voici la question :
    ---------------------
    Contexte : {context}
    Question : {question}
    """
)

@st.cache_resource
def load_qa_chain():
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatGroq(api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={"prompt": CUSTOM_PROMPT})
    return qa_chain

st.title("🤖 Assistant IA d’Aivancity")

question = st.text_input("Posez votre question à l’assistant :")
qa_chain = load_qa_chain()

if question:
    with st.spinner("Réflexion en cours..."):
        result = qa_chain.run(question)
        st.success(result)
