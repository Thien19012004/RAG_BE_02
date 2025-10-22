# -*- coding: utf-8 -*-
"""
FastAPI application for RAG system
- Upload PDF endpoint
- Query endpoint with RAG pipeline
"""

import os
import shutil
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import PDF_PATH, CACHE_DIR, CHROMA_PATH, HASH_FILE
from langchain_multimodal import _build_pipeline


app = FastAPI(title="RAG PDF API", version="1.0.0")

# Global variables to store the RAG pipeline
rag_chain = None
rag_chain_with_ctx = None


class QueryRequest(BaseModel):
    question: str
    include_context: bool = False


class QueryResponse(BaseModel):
    answer: str
    context: Optional[dict] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the RAG pipeline on startup if PDF exists"""
    global rag_chain, rag_chain_with_ctx
    try:
        if os.path.exists(PDF_PATH):
            print("🚀 Initializing RAG pipeline...")
            rag_chain, rag_chain_with_ctx = _build_pipeline()
            print("✅ RAG pipeline initialized successfully")
        else:
            print("⚠️ No PDF found. Please upload a PDF first.")
    except Exception as e:
        print(f"❌ Failed to initialize RAG pipeline: {e}")
        # Don't raise - let the API start without pipeline


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "RAG PDF API is running", "status": "healthy"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file and rebuild the RAG pipeline
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Save uploaded file
        with open(PDF_PATH, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Clear existing cache and vectorstore to force rebuild
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
        
        # Recreate directories
        os.makedirs(CACHE_DIR, exist_ok=True)
        os.makedirs(CHROMA_PATH, exist_ok=True)
        
        # Rebuild pipeline with new PDF
        global rag_chain, rag_chain_with_ctx
        rag_chain, rag_chain_with_ctx = _build_pipeline()
        
        return {
            "message": "PDF uploaded and processed successfully",
            "filename": file.filename,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_pdf(request: QueryRequest):
    """
    Query the uploaded PDF using RAG pipeline
    """
    if rag_chain is None or rag_chain_with_ctx is None:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized. Please upload a PDF first.")
    
    try:
        if request.include_context:
            # Use chain that returns context
            result = rag_chain_with_ctx.invoke(request.question)
            return QueryResponse(
                answer=result["response"],
                context={
                    "texts": [
                        {
                            "text": getattr(t, "text", "")[:400] if hasattr(t, "text") else "",
                            "type": type(t).__name__
                        } 
                        for t in result["context"]["texts"][:3]
                    ],
                    "images": [
                        {"length": len(img), "preview": img[:24] + "..."} 
                        for img in result["context"]["images"][:3]
                    ]
                }
            )
        else:
            # Use simple chain
            answer = rag_chain.invoke(request.question)
            return QueryResponse(answer=answer)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/status")
async def get_status():
    """Get current status of the system"""
    pdf_exists = os.path.exists(PDF_PATH)
    cache_exists = os.path.exists(CACHE_DIR)
    vectorstore_exists = os.path.exists(CHROMA_PATH)
    
    return {
        "pdf_uploaded": pdf_exists,
        "cache_exists": cache_exists,
        "vectorstore_exists": vectorstore_exists,
        "pipeline_ready": rag_chain is not None and rag_chain_with_ctx is not None,
        "pdf_path": PDF_PATH if pdf_exists else None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

