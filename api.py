# -*- coding: utf-8 -*-
"""
FastAPI application for RAG system
- Upload PDF endpoint
- Query endpoint with RAG pipeline
"""

import os
import shutil
from typing import Optional, Dict, Tuple
from pathlib import Path
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import PDF_PATH, CACHE_DIR, CHROMA_PATH, HASH_FILE
from langchain_multimodal import _build_pipeline


app = FastAPI(title="RAG PDF API", version="1.0.0")

# In-memory store: file_id -> (rag_chain, rag_chain_with_ctx)
pipelines: Dict[str, Tuple[object, object]] = {}


def _paths_for(file_id: str) -> Dict[str, str]:
    base_content = os.path.join("./content", f"{file_id}.pdf")
    base_cache = os.path.join("./cache", file_id)
    base_chroma = os.path.join("./chroma_store", file_id)
    base_hash = os.path.join(base_chroma, "last_hash.txt")
    return {
        "pdf_path": base_content,
        "cache_dir": base_cache,
        "chroma_path": base_chroma,
        "hash_file": base_hash,
    }


def _set_config_paths_for(file_id: str) -> Dict[str, str]:
    """Temporarily point config paths to per-file directories for building."""
    from importlib import reload
    import config as cfg

    paths = _paths_for(file_id)

    # Ensure directories exist
    os.makedirs(os.path.dirname(paths["pdf_path"]), exist_ok=True)
    os.makedirs(paths["cache_dir"], exist_ok=True)
    os.makedirs(paths["chroma_path"], exist_ok=True)

    # Monkey-patch paths in config for this build
    cfg.PDF_PATH = paths["pdf_path"]
    cfg.CACHE_DIR = paths["cache_dir"]
    cfg.CHROMA_PATH = paths["chroma_path"]
    cfg.HASH_FILE = paths["hash_file"]

    return paths


class QueryRequest(BaseModel):
    file_id: str
    question: str
    include_context: bool = False


class QueryResponse(BaseModel):
    answer: str
    context: Optional[dict] = None


@app.on_event("startup")
async def startup_event():
    """No-op startup. Pipelines are built per upload."""
    print("ℹ️ API started. Upload a PDF to build a pipeline.")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "RAG PDF API is running", "status": "healthy"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), file_id: Optional[str] = Form(default=None)):
    """
    Upload a PDF file, build its pipeline, and attach to a file_id
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Resolve or generate file_id
        fid = file_id or str(uuid.uuid4())

        # Point config paths to this file_id
        paths = _set_config_paths_for(fid)

        # Save uploaded file to per-file path
        with open(paths["pdf_path"], "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Rebuild pipeline for this file_id
        rag_chain, rag_chain_with_ctx = _build_pipeline()

        # Attach to memory store
        pipelines[fid] = (rag_chain, rag_chain_with_ctx)
        
        return {
            "message": "PDF uploaded and processed successfully",
            "filename": file.filename,
            "file_id": fid,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_pdf(request: QueryRequest):
    """
    Query a specific uploaded PDF (by file_id) using its pipeline
    """
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
    # Report known pipelines and their paths
    known = []
    for fid in pipelines.keys():
        p = _paths_for(fid)
        known.append({
            "file_id": fid,
            "pdf_path": p["pdf_path"],
            "cache_dir": p["cache_dir"],
            "chroma_path": p["chroma_path"],
        })

    return {
        "pipelines": known,
        "count": len(known),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

