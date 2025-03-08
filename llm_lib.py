from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Body, status
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
import uuid
import shutil
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import uuid
import re
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaLLM,OllamaEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
import pandas as pd


# Define Pydantic models
class Message(BaseModel):
    role: str = Field(..., description="Role of the message sender (user or assistant)")
    content: str = Field(..., description="Content of the message")
    timestamp: datetime = Field(default_factory=datetime.now)

class ChatRequest(BaseModel):
    query: str = Field(..., description="User's query to the chatbot")
    chat_id: Optional[str] = Field(None, description="Chat ID for continuing conversations")
    model: Optional[str] = Field(None, description="Model used for conversations")
    context: Optional[List[Message]] = Field(None, description="Previous conversation context")

class ChatResponse(BaseModel):
    chat_id: str = Field(..., description="Unique identifier for the chat session")
    response: str = Field(..., description="AI assistant's response")
    sources: Optional[List[str]] = Field(None, description="Sources used for generating the response")
    
class DocumentUploadRequest(BaseModel):
    file_path: str = Field(..., description="Path to the PDF file to be processed")
    
class DocumentUploadResponse(BaseModel):
    status: str = Field(..., description="Status of the document processing")
    vector_store_id: str = Field(..., description="ID of the created vector store")

# In-memory storage
chat_history = {}
vector_stores = {}

# Dependency for retrieving embeddings
async def get_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")

# Dependency for retrieving LLM
import subprocess
import json
from typing import List

def get_available_models() -> List[str]:
    try:
        models = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True).stdout.split()[4::7]
        return models
    except subprocess.CalledProcessError as e:
        print(f"Error listing models: {e}")
        return []

def get_llm(request: ChatRequest) -> OllamaLLM:
    default_model = "deepseek-r1:1.5b"
    available_models = get_available_models()
    model_name = request.model if request.model in available_models else default_model

    try:
        return OllamaLLM(model=model_name)
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        return OllamaLLM(model=default_model)

# Dependency to get vector store
async def get_vector_store(vector_store_id: str = None):
    if not vector_store_id and "default" in vector_stores:
        return vector_stores["default"]
    
    if vector_store_id and vector_store_id in vector_stores:
        return vector_stores[vector_store_id]
    
    # Check if default vector store exists on disk
    try:
        embeddings = await get_embeddings()
        default_store = FAISS.load_local("faiss_gst_vectordb800-160", embeddings, allow_dangerous_deserialization=True)
        vector_stores["default"] = default_store
        return default_store
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Vector store not found: {str(e)}")


# Background task functions
async def process_document(file_path: str, vector_store_id: str, embedding):
    """
    Process a document (PDF, TXT, CSV, XLSX) and create a vector store.
    """
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        documents = []

        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
            documents = loader.load()

        elif file_extension == ".txt":
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                documents = [text]

        elif file_extension == ".csv":
            df = pd.read_csv(file_path)
            text = df.to_string(index=False)  # Convert CSV to a readable string
            documents = [text]

        elif file_extension in [".xls", ".xlsx"]:
            df = pd.read_excel(file_path, sheet_name=None)  # Read all sheets
            text = "\n\n".join([df[sheet].to_string(index=False) for sheet in df])
            documents = [text]

        else:
            raise ValueError("Unsupported file format. Only PDF, TXT, CSV, and XLSX are allowed.")

        # Ensure document is not empty
        if not documents:
            raise ValueError("Document is empty or unreadable.")

        # Split text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_text("\n".join(documents))

        # Create and store the vector store
        vector_store = FAISS.from_texts(splits, embedding)
        vector_stores[vector_store_id] = vector_store

        # Optionally save to disk
        vector_store.save_local(f"faiss_index_{vector_store_id}")

    except Exception as e:
        print(f"❌ Error processing document: {str(e)}")

async def cleanup_old_chats():
    """
    Remove old chat sessions to manage memory.
    """
    current_time = datetime.now()
    chats_to_remove = []
    
    for chat_id, messages in chat_history.items():
        if messages:
            last_message_time = messages[-1].timestamp
            # Remove chats older than 24 hours
            if (current_time - last_message_time).total_seconds() > 86400:
                chats_to_remove.append(chat_id)
    
    for chat_id in chats_to_remove:
        if chat_id in chat_history:
            del chat_history[chat_id]
