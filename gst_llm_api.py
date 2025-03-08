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
from llm_lib import *
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaLLM,OllamaEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub

# Initialize FastAPI app
app = FastAPI(title="LLM Chatbot API", description="API for interacting with an LLM-powered chatbot")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    background_tasks: BackgroundTasks,
    request: ChatRequest,
    embeddings: OllamaEmbeddings = Depends(get_embeddings),
    vector_store: FAISS = Depends(get_vector_store)
):
    """
    Process a chat request and generate a response using the LLM and vector store.
    """
    llm=get_llm(request)
    print(request)
    try:
        # Generate or retrieve chat ID
        chat_id = request.chat_id if request.chat_id else str(uuid.uuid4())
        
        # Retrieve conversation history or create new
        if chat_id not in chat_history:
            chat_history[chat_id] = []
        
        # Add user message to history
        chat_history[chat_id].append(Message(role="user", content=request.query))
        
        # Create retrieval chain
        # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        prompt = f"""
        Classify the following query into one of the two categories:

        1. **Knowledge-Based**: Queries related to **GST, consumers, laws, taxes, expenditure, or any other factual or informational topics**.  
        2. **Conversation-Based**: Casual interactions like greetings (e.g., "hi", "hello"), general inquiries (e.g., "how are you?"), or vague/open-ended requests (e.g., "can you help me?").  

        **Query:** "{request.query}"  

        **Output the category as a single word:**
        - Respond with **"knowledge-based"** if the query requires factual retrieval.
        - Respond with **"conversation-based"** if it is a general or social interaction.
        """
        classification = re.sub(r"<think>.*?</think>", "", llm.invoke(prompt), flags=re.DOTALL)
        print(classification)
        if 'conversation-based' in classification:
            response=re.sub(r"<think>.*?</think>", "", llm.invoke(request.query), flags=re.DOTALL)
            result={}

        else:
            retrieval_qa_chat_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""
                Use the following context to answer the question.
                Ignore the Context if the query is conversational.
                Question:
                {question}

                Context:
                {context}

                
                Question:
                {question}
                If the answer is not in the context, say "Information not found".
                """
            )
            try:retrieved_docs = vector_store.as_retriever().invoke(request.query)
            except:print('error')

            # Print retrieved documents for debugging
            print("Retrieved Documents:")
            for i, doc in enumerate(retrieved_docs):
                print(f"Doc {i+1}:\n{doc.page_content}\n{'-'*40}")
            combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
            retrieval_chain = create_retrieval_chain(vector_store.as_retriever(), combine_docs_chain)
            
            # Generate response
            print('generating')
            result = retrieval_chain.invoke({"question": request.query,"input": request.query})
            response = re.sub(r"<think>.*?</think>", "", result.get("answer", "I couldn't generate a response."), flags=re.DOTALL)
        print(response)
        # Extract sources if available
        sources = []
        if "source_documents" in result:
            for doc in result["source_documents"]:
                if hasattr(doc, "metadata") and "source" in doc.metadata:
                    sources.append(doc.metadata["source"])
        
        # Add assistant message to history
        chat_history[chat_id].append(Message(role="assistant", content=response))
        
        # Schedule cleanup of old chats in background
        background_tasks.add_task(cleanup_old_chats)
        
        # return ChatResponse(chat_id=chat_id, response=response, sources=sources)
        return ChatResponse(chat_id=chat_id, response=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@app.post("/api/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    embeddings = Depends(get_embeddings)
):
    """
    Process a PDF document and create a vector store from its contents.
    """
    try:
        vector_store_id = str(uuid.uuid4())
        
        # Save the uploaded file to a temporary location
        temp_dir = Path("uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        file_path = temp_dir / file.filename
        
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Add task to process document in the background
        background_tasks.add_task(
            process_document, 
            str(file_path), 
            vector_store_id, 
            embeddings
        )
        
        return DocumentUploadResponse(
            status="processing",
            vector_store_id=vector_store_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@app.get("/api/documents/status/{vector_store_id}")
async def get_document_status(vector_store_id: str):
    """
    Check the status of a document processing task.
    """
    if vector_store_id in vector_stores:
        return {"status": "completed", "vector_store_id": vector_store_id}
    
    return {"status": "processing", "vector_store_id": vector_store_id}
@app.get("/models", response_model=List[str])
async def list_models():
    models = get_available_models()
    if not models:
        raise HTTPException(status_code=500, detail="Failed to retrieve models")
    return models
@app.get("/api/chats/{chat_id}", response_model=List[Message])
async def get_chat_history(chat_id: str):
    """
    Retrieve the history of a specific chat session.
    """
    if chat_id not in chat_history:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    return chat_history[chat_id]

@app.delete("/api/chats/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(chat_id: str):
    """
    Delete a specific chat session.
    """
    if chat_id in chat_history:
        del chat_history[chat_id]
    
    return None

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)