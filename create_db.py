import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
import requests

session = requests.Session()
session.verify = False 

pdf_paths = [
    r'C:\Users\XH656HB\Downloads\Aug 23.pdf',
    r'C:\Users\XH656HB\Downloads\Aug 22.pdf'  # Add second PDF here
]

# Load documents from both PDFs
all_documents = []
for pdf_path in pdf_paths:
    loader = PyPDFLoader(file_path=pdf_path)
    all_documents.extend(loader.load())

# Split text into chunks
text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=160, separator="\n")
docs = text_splitter.split_documents(documents=all_documents)

# Create embeddings and store in FAISS
embeddings = OllamaEmbeddings(model="nomic-embed-text")
print('hi')
vectorstore = FAISS.from_documents(docs, embeddings)

# Save vectorstore
vectorstore.save_local("faiss_gst_vectordb800-160")