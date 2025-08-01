from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = TextLoader("corpus_ecole.txt", encoding="utf-8")
docs = loader.load()


# Split des documents
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# Création du vectorstore Chroma
Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory="chroma_store")

print("✅ Index Chroma généré dans le dossier 'chroma_store'")
