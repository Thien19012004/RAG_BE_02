# -*- coding: utf-8 -*-
"""
FastAPI application for RAG system
- Upload PDF endpoint
- Mode-aware query endpoint (document vs corpus)
"""

import shutil
from typing import Literal, Optional, Dict, Tuple
import uuid
from concurrent.futures import ThreadPoolExecutor
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from pydantic import BaseModel

from config import FileConfig, CHROMA_DIR
from langchain_multimodal import ingest_document
from summarization import build_region_explainer, build_region_explainer_hybrid
from vectorstore_setup import build_local_chroma_backend
from rag_pipeline import build_document_rag, build_corpus_rag, PromptConfig


app = FastAPI(title="RAG PDF API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Vector backend shared by document + corpus modes
VECTOR_BACKEND = build_local_chroma_backend(str(CHROMA_DIR / "global_store"))

# In-memory structures ----------------------------------------------------
pipelines: Dict[str, Dict[str, object]] = {}
corpus_pipeline: Optional[Tuple[object, object]] = None
processing_status: Dict[str, str] = {}
executor = ThreadPoolExecutor(max_workers=2)


class QueryRequest(BaseModel):
    question: str
    file_id: Optional[str] = None
    mode: Optional[Literal["document", "corpus"]] = None
    include_context: bool = False


class QueryResponse(BaseModel):
    answer: str
    context: Optional[dict] = None
    mode: Literal["document", "corpus"]


class UploadResponse(BaseModel):
    message: str
    filename: str
    file_id: str
    status: str
    processing_time: Optional[float] = None


@app.on_event("startup")
async def startup_event():
    print("ℹ️ API started. Upload PDFs to build document pipelines or use corpus mode.")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "RAG PDF API is running", "status": "healthy"}


def build_pipeline_sync(file_config: FileConfig):
    """Synchronous ingestion + document-mode pipeline build used in background tasks."""
    try:
        print(f"🟠 [UPLOAD] Begin processing for file_id={file_config.file_id}")
        processing_status[file_config.file_id] = "processing"
        ingestion_result = ingest_document(file_config, VECTOR_BACKEND)
        prompt_cfg = PromptConfig(
            mode="document",
            paper_id=file_config.file_id,
            paper_title=ingestion_result.title,
        )
        rag_chain, rag_chain_with_ctx = build_document_rag(
            paper_id=file_config.file_id,
            backend=VECTOR_BACKEND,
            prompt_cfg=prompt_cfg,
        )
        pipelines[file_config.file_id] = {
            "chains": (rag_chain, rag_chain_with_ctx),
            "metadata": ingestion_result,
        }
        processing_status[file_config.file_id] = "completed"
        print(f"🟢 [UPLOAD] Completed processing for file_id={file_config.file_id}")
        return True
    except Exception as e:
        print(f"🔴 [UPLOAD] Error while processing file_id={file_config.file_id}: {str(e)}")
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
        print(f"⬆️  [UPLOAD] Receiving file '{file.filename}' -> file_id={fid}")
        
        # Create file config
        file_config = FileConfig(fid)
        
        # Save uploaded file
        with open(file_config.pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"💾 [UPLOAD] Saved PDF to {file_config.pdf_path}")
        
        # Start background processing
        processing_status[fid] = "queued"
        print(f"🕓 [UPLOAD] Queued processing for file_id={fid}")
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
        print(f"🔴 [UPLOAD] Error during upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")


class ExplainRequest(BaseModel):
    image_b64: str
    file_id: Optional[str] = None
    page_number: Optional[int] = None
    # Optional crop rect (normalized or px from FE if needed later)
    rect: Optional[dict] = None


class ExplainResponse(BaseModel):
    explanation: str


@app.post("/explain-region", response_model=ExplainResponse)
async def explain_region(req: ExplainRequest):
    """
    Explain a cropped region (math formula, table, figure, or text) from base64 image.
    FE should send base64 data URL payload (only the base64 part is needed).
    """
    try:
        if not req.image_b64 or len(req.image_b64) < 50:
            raise HTTPException(status_code=400, detail="Invalid image_b64")

        print(f"🟠 [EXPLAIN] Received region for analysis (size={len(req.image_b64)} chars)")

        context_text = ""
        used_hybrid = False

        # If file_id provided, try to load cached summaries to serve as textual context
        if req.file_id:
            try:
                file_config = FileConfig(req.file_id)
                cache_text = file_config.cache_dir / "text_summaries.json"
                cache_table = file_config.cache_dir / "table_summaries.json"
                cache_image = file_config.cache_dir / "image_summaries.json"

                snippets = []
                import json
                if cache_text.exists():
                    txts = json.load(open(cache_text, encoding="utf-8"))
                    snippets.extend([t for t in txts if isinstance(t, str) and t.strip()][:8])
                if cache_table.exists():
                    tabs = json.load(open(cache_table, encoding="utf-8"))
                    snippets.extend([t for t in tabs if isinstance(t, str) and t.strip()][:4])
                if cache_image.exists():
                    imgs = json.load(open(cache_image, encoding="utf-8"))
                    snippets.extend([t for t in imgs if isinstance(t, str) and t.strip()][:4])

                # Join and cap length
                context_text = "\n---\n".join(snippets)
                if len(context_text) > 4000:
                    context_text = context_text[:4000] + " ..."

                if context_text.strip():
                    print(f"📚 [EXPLAIN] Using hybrid context from cache (len={len(context_text)}) for file_id={req.file_id}")
                    chain = build_region_explainer_hybrid()
                    explanation = chain.invoke({"image_b64": req.image_b64, "context_text": context_text})
                    used_hybrid = True
                else:
                    print("ℹ️ [EXPLAIN] No non-empty context found in cache; falling back to vision-only")
            except Exception as ctx_e:
                print(f"⚠️ [EXPLAIN] Failed to load context for file_id={req.file_id}: {str(ctx_e)}")

        if not used_hybrid:
            chain = build_region_explainer()
            explanation = chain.invoke({"image_b64": req.image_b64})

        print("🟢 [EXPLAIN] Explanation generated")
        return ExplainResponse(explanation=explanation.strip() if explanation else "")
    except HTTPException:
        raise
    except Exception as e:
        print(f"🔴 [EXPLAIN] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error explaining region: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_pdf(request: QueryRequest):
    """
    Mode-aware querying. Defaults to document mode when file_id is provided, otherwise corpus mode.
    """
    mode = request.mode or ("document" if request.file_id else "corpus")

    if mode == "document":
        if not request.file_id:
            raise HTTPException(status_code=400, detail="file_id required for document mode")
        status = processing_status.get(request.file_id)
        if status in {"queued", "processing"}:
            raise HTTPException(status_code=202, detail=f"File is still processing. Status: {status}")
        if status and status.startswith("error"):
            raise HTTPException(status_code=500, detail=f"Processing failed: {status}")
        if request.file_id not in pipelines:
            raise HTTPException(status_code=404, detail="Unknown file_id. Please upload first.")

        rag_chain, rag_chain_with_ctx = pipelines[request.file_id]["chains"]
        try:
            if request.include_context:
                result = rag_chain_with_ctx.invoke(request.question)
                return QueryResponse(
                    answer=result["response"],
                    context=result.get("context"),
                    mode="document",
                )
            answer = rag_chain.invoke(request.question)
            return QueryResponse(answer=answer, mode="document")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing document query: {str(e)}")

    # Corpus mode ---------------------------------------------------------
    global corpus_pipeline
    if corpus_pipeline is None:
        print("ℹ️ [QUERY] Initializing corpus pipeline lazily")
        prompt_cfg = PromptConfig(mode="corpus")
        corpus_pipeline = build_corpus_rag(VECTOR_BACKEND, prompt_cfg=prompt_cfg)

    rag_chain, rag_chain_with_ctx = corpus_pipeline
    try:
        if request.include_context:
            result = rag_chain_with_ctx.invoke(request.question)
            return QueryResponse(
                answer=result["response"],
                context=result.get("context"),
                mode="corpus",
            )
        answer = rag_chain.invoke(request.question)
        return QueryResponse(answer=answer, mode="corpus")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing corpus query: {str(e)}")


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
            "vector_backend": "local-chroma",
            "status": status,
            "title": getattr(pipelines[fid].get("metadata"), "title", None),
            "mode": "document",
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
        "can_query": is_ready and not status.startswith("error"),
        "title": getattr(pipelines.get(file_id, {}).get("metadata"), "title", None),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

