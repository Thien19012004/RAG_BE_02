# -*- coding: utf-8 -*-
"""
FastAPI application for RAG system
- Upload PDF endpoint
- Query endpoint with RAG pipeline
"""

import shutil
from typing import Optional, Dict, Tuple
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from pydantic import BaseModel

from config import FileConfig
from langchain_multimodal import build_pipeline


app = FastAPI(title="RAG PDF API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],   # hoặc ['http://localhost:3000']
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store: file_id -> (rag_chain, rag_chain_with_ctx)
pipelines: Dict[str, Tuple[object, object]] = {}

# Processing status tracking
processing_status: Dict[str, str] = {}  # file_id -> status

# Thread pool for background processing
executor = ThreadPoolExecutor(max_workers=2)


class QueryRequest(BaseModel):
    file_id: str
    question: str
    include_context: bool = False


class QueryResponse(BaseModel):
    answer: str
    context: Optional[dict] = None


class UploadResponse(BaseModel):
    message: str
    filename: str
    file_id: str
    status: str
    processing_time: Optional[float] = None


@app.on_event("startup")
async def startup_event():
    """No-op startup. Pipelines are built per upload."""
    print("ℹ️ API started. Upload a PDF to build a pipeline.")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "RAG PDF API is running", "status": "healthy"}


def build_pipeline_sync(file_config):
    """Synchronous pipeline building for background processing"""
    try:
        processing_status[file_config.file_id] = "processing"
        rag_chain, rag_chain_with_ctx = build_pipeline(file_config)
        pipelines[file_config.file_id] = (rag_chain, rag_chain_with_ctx)
        processing_status[file_config.file_id] = "completed"
        return True
    except Exception as e:
        processing_status[file_config.file_id] = f"error: {str(e)}"
        return False


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    file_id: Optional[str] = Form(default=None)
):
    """
    Upload a PDF file and process it asynchronously for better performance
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        import time
        start_time = time.time()
        
        # Resolve or generate file_id
        fid = file_id or str(uuid.uuid4())
        
        # Create file config
        file_config = FileConfig(fid)
        
        # Save uploaded file
        with open(file_config.pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Start background processing
        processing_status[fid] = "queued"
        background_tasks.add_task(build_pipeline_sync, file_config)
        
        processing_time = time.time() - start_time
        
        return UploadResponse(
            message="PDF uploaded successfully. Processing in background.",
            filename=file.filename,
            file_id=fid,
            status="processing",
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_pdf(request: QueryRequest):
    """
    Query a specific uploaded PDF (by file_id) using its pipeline
    """
    # Check if file is still processing
    if request.file_id in processing_status:
        status = processing_status[request.file_id]
        if status.startswith("processing") or status.startswith("queued"):
            raise HTTPException(
                status_code=202, 
                detail=f"File is still processing. Status: {status}"
            )
        elif status.startswith("error"):
            raise HTTPException(status_code=500, detail=f"Processing failed: {status}")
    
    if request.file_id not in pipelines:
        raise HTTPException(status_code=404, detail="Unknown file_id. Please upload and use the returned file_id.")
    
    try:
        rag_chain, rag_chain_with_ctx = pipelines[request.file_id]
        
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
    known = []
    for fid in pipelines.keys():
        file_config = FileConfig(fid)
        status = processing_status.get(fid, "completed")
        known.append({
            "file_id": fid,
            "pdf_path": str(file_config.pdf_path),
            "cache_dir": str(file_config.cache_dir),
            "chroma_path": str(file_config.chroma_path),
            "status": status,
        })

    return {
        "pipelines": known,
        "count": len(known),
        "processing_status": processing_status,
    }


@app.get("/status/{file_id}")
async def get_file_status(file_id: str):
    """Get processing status for a specific file"""
    if file_id not in processing_status and file_id not in pipelines:
        raise HTTPException(status_code=404, detail="File not found")
    
    status = processing_status.get(file_id, "completed")
    is_ready = file_id in pipelines
    
    return {
        "file_id": file_id,
        "status": status,
        "ready": is_ready,
        "can_query": is_ready and not status.startswith("error")
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

