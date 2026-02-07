# -*- coding: utf-8 -*-
"""FastAPI application for RAG system"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import FileConfig, CHROMA_DIR, settings
from langchain_multimodal import ingest_document, IngestionResult
from vectorstore_setup import build_local_chroma_backend
from rag_pipeline import (
    build_document_rag_chain,
    build_generative_chain,
    brainstorm_questions_chain,
    PromptConfig,
    REGION_EXPLAIN_INSTRUCTIONS,
    split_docs,
    multi_document_retrieve,
)
from api_utils import (
    retrieve_context_for_explain,
    format_explain_context_for_chain,
)
from arxiv_related import suggest_related_papers

# Database module - reads from Backend's tables, manages RAG cache
from database import (
    get_status,
    load_metadata,
    get_database,
)


# -----------------------------------------------------------------------------
# FastAPI app & CORS
# -----------------------------------------------------------------------------
app = FastAPI(title="RAG PDF API", version="2.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev: mở hết; prod thì siết lại
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Global backends (in-memory)
# -----------------------------------------------------------------------------
VECTOR_BACKEND = build_local_chroma_backend(str(CHROMA_DIR / "global_store"))

# Pipelines kept in-memory (can be rebuilt from vector store after restart)
pipelines: Dict[str, Any] = {}

# In-memory cache for metadata (for performance, backed by database)
file_metadata: Dict[str, IngestionResult] = {}


# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str
    file_id: str


class QueryResponse(BaseModel):
    answer: str
    context: Optional[dict] = None


class ExplainRequest(BaseModel):
    image_b64: str
    file_id: str
    page_number: Optional[int] = None
    question: Optional[str] = "Please analyze and explain this cropped region."


class UploadResponse(BaseModel):
    message: str
    file_id: str
    status: str
    processing_time: Optional[float] = None
    title: Optional[str] = None
    abstract: Optional[str] = None
    authors: Optional[List[str]] = None  # List of author names
    num_pages: Optional[int] = None  # Total page count
    node_count: Optional[int] = None
    table_count: Optional[int] = None
    image_count: Optional[int] = None


class IngestFromUrlRequest(BaseModel):
    """Request to ingest a PDF from a cloud URL (S3/MinIO)."""
    file_url: str
    file_id: Optional[str] = None


class RelatedPapersRequest(BaseModel):
    file_id: str
    top_k: int = 5
    max_results: int = 30


class RelatedPaper(BaseModel):
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    categories: List[str]
    url: str
    score: float
    reason: str


class RelatedPapersResponse(BaseModel):
    file_id: str
    base_title: Optional[str]
    base_abstract: Optional[str]
    results: List[RelatedPaper]


class BrainstormRequest(BaseModel):
    file_id: str

class BrainstormResponse(BaseModel):
    questions: List[str]


class MultiQueryRequest(BaseModel):
    """Request to query across multiple PDFs."""
    question: str
    file_ids: List[str]  # List of file IDs to search across


class MultiQueryResponse(BaseModel):
    """Response from multi-PDF query."""
    answer: str
    context: Optional[dict] = None
    sources: Optional[List[dict]] = None  # Which papers contributed to the answer

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def validate_file_ready(file_id: str) -> None:
    """Validate that a file exists and is ready for querying."""
    status = get_status(file_id)
    if status is None:
        raise HTTPException(404, "File not found")
    if status != "completed":
        # Currently processing or failed
        if status == "failed":
            raise HTTPException(500, "File processing failed")
        raise HTTPException(202, "File processing not finished")


async def build_pipeline_sync(file_config: FileConfig) -> IngestionResult:
    """
    Run ingest + build RAG chain for a PDF synchronously.
    Called by /upload and /ingest-from-url - only returns when complete.

    Note: Backend updates paper status in `papers` table.
    RAG only stores file hash in `rag_paper_cache` for rebuild detection.

    Returns:
        IngestionResult with metadata from ingestion
    """
    try:
        # Ensure file is downloaded if from cloud (to temp location)
        file_config.ensure_local_file()

        res = await ingest_document(file_config, VECTOR_BACKEND)

        # Build query chain
        prompt_cfg = PromptConfig(
            paper_id=file_config.file_id,
            paper_title=res.title,
        )
        chain = build_document_rag_chain(file_config.file_id, VECTOR_BACKEND, prompt_cfg)

        pipelines[file_config.file_id] = chain

        # Save file hash for rebuild detection (RAG's cache)
        db = get_database()
        db.save_file_hash(file_config.file_id, file_config.get_file_hash())

        # Cleanup temporary local file after successful processing
        file_config.cleanup_local_file()

        return res
    except Exception as e:
        print(f"Error while building pipeline for {file_config.file_id}: {e}")
        # Cleanup on error too
        file_config.cleanup_local_file()
        raise


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    file_id: Optional[str] = Form(None),
):
    """
    Upload PDF and build pipeline synchronously.

    NOTE: This endpoint is primarily for development/testing.
    In production, use /ingest-from-url with cloud storage.

    - Request: multipart/form-data with file + optional file_id
    - Response: UploadResponse with metadata
    """
    fid = file_id or str(uuid.uuid4())

    # Create FileConfig - will use temp directory for processing
    file_config = FileConfig(fid)

    # Save uploaded file to temp location for processing
    temp_path = file_config.get_temp_pdf_path()
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    start = time.time()
    try:
        res = await build_pipeline_sync(file_config)
        elapsed = time.time() - start

        return UploadResponse(
            message="Uploaded and processed successfully",
            file_id=fid,
            status="completed",
            processing_time=elapsed,
            title=res.title,
            abstract=res.abstract,
            authors=res.authors,
            num_pages=res.num_pages,
            node_count=res.node_count,
            table_count=res.table_count,
            image_count=res.image_count,
        )
    except Exception as e:
        elapsed = time.time() - start
        status = get_status(fid) or "error"
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest-from-url", response_model=UploadResponse)
async def ingest_from_url(req: IngestFromUrlRequest):
    """
    Ingest a PDF from a cloud URL (S3/MinIO).

    This is the RECOMMENDED endpoint for production:
    1. Backend uploads PDF to S3
    2. Backend calls this endpoint with the S3 URL
    3. RAG service downloads, processes, and cleans up the local file

    - Request: JSON with file_url and optional file_id
    - Response: UploadResponse with metadata
    """
    fid = req.file_id or str(uuid.uuid4())

    # Create FileConfig with cloud URL
    file_config = FileConfig(fid, cloud_url=req.file_url)

    start = time.time()
    try:
        res = await build_pipeline_sync(file_config)
        elapsed = time.time() - start

        return UploadResponse(
            message="Ingested from URL successfully",
            file_id=fid,
            status="completed",
            processing_time=elapsed,
            title=res.title,
            abstract=res.abstract,
            authors=res.authors,
            num_pages=res.num_pages,
            node_count=res.node_count,
            table_count=res.table_count,
            image_count=res.image_count,
        )
    except Exception as e:
        elapsed = time.time() - start
        status = get_status(fid) or "error"
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_pdf(req: QueryRequest):
    """Text query chuẩn trên 1 PDF đã ingest."""
    validate_file_ready(req.file_id)
    chain = pipelines.get(req.file_id)

    # Nếu server restart mất pipeline in-memory thì build lại từ vector store
    if chain is None:
        res = load_metadata(req.file_id)
        if res is None:
            raise HTTPException(404, "Metadata not found for this file")
        prompt_cfg = PromptConfig(
            paper_id=req.file_id,
            paper_title=res.title,
        )
        chain = build_document_rag_chain(req.file_id, VECTOR_BACKEND, prompt_cfg)
        pipelines[req.file_id] = chain

    out = chain.invoke({"question": req.question})
    # rag_pipeline trả {"response": answer, "context": ctx}
    return QueryResponse(answer=out["response"], context=out.get("context"))


@app.post("/query-multi", response_model=MultiQueryResponse)
async def query_multi_pdf(req: MultiQueryRequest):
    """
    Query across multiple PDFs at once.

    This endpoint allows users to ask questions that span multiple papers,
    enabling cross-paper analysis and comparison.
    """
    if not req.file_ids:
        raise HTTPException(400, "At least one file_id is required")

    if len(req.file_ids) > 10:
        raise HTTPException(400, "Maximum 10 papers can be queried at once")

    # Validate all files are ready
    paper_titles = {}
    for file_id in req.file_ids:
        validate_file_ready(file_id)
        meta = load_metadata(file_id)
        if meta:
            paper_titles[file_id] = meta.title or f"Paper {file_id[:8]}"
        else:
            paper_titles[file_id] = f"Paper {file_id[:8]}"

    # Retrieve documents from all papers
    docs = multi_document_retrieve(
        query=req.question,
        paper_ids=req.file_ids,
        backend=VECTOR_BACKEND,
        k_per_paper=6,
        total_k=15,
    )

    # Split docs into modality groups
    context = split_docs(docs)

    # Build prompt config for multi-paper query
    paper_list = ", ".join([f'"{title}"' for title in paper_titles.values()])
    multi_paper_instructions = (
        f"You are analyzing multiple research papers: {paper_list}. "
        "Use the provided context from ALL papers to answer the question. "
        "When citing, indicate which paper the information comes from using [S1], [S2], etc. "
        "If comparing papers, clearly distinguish findings from each source. "
        "Synthesize information across papers when relevant.\n\n"
        "CRITICAL LaTeX Formatting Rules:\n"
        "- For inline math, use single dollar signs: $E = mc^2$\n"
        "- For block/display math, use double dollar signs on separate lines:\n"
        "$$\n"
        "\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}\n"
        "$$\n"
        "- Always use proper LaTeX commands with backslash: \\frac, \\sum, \\int\n"
        "- Ensure all math delimiters are properly balanced (every $ has a matching $)\n"
        "- Use \\text{} for text inside math mode"
    )

    prompt_cfg = PromptConfig(
        paper_id=None,  # Multiple papers
        system_instructions=multi_paper_instructions,
    )

    # Generate answer
    gen_chain = build_generative_chain()
    answer = gen_chain.invoke({
        "context": context,
        "question": req.question,
        "prompt_cfg": prompt_cfg,
        "focus_image_b64": None,
    })

    # Build sources info for response
    sources = []
    seen_papers = set()
    for item in context.get("texts", []) + context.get("tables", []) + context.get("images", []):
        paper_id = item.get("metadata", {}).get("source_paper_id") or item.get("metadata", {}).get("paper_id")
        if paper_id and paper_id not in seen_papers:
            seen_papers.add(paper_id)
            sources.append({
                "paper_id": paper_id,
                "title": paper_titles.get(paper_id, f"Paper {paper_id[:8]}"),
            })

    return MultiQueryResponse(
        answer=answer,
        context=context,
        sources=sources,
    )


@app.post("/explain-region", response_model=QueryResponse)
async def explain_region(req: ExplainRequest):
    """
    Giải thích 1 vùng crop từ PDF (image + local context xung quanh).
    """
    validate_file_ready(req.file_id)

    # 1) Lấy context hybrid (vector + cache)
    context_data = retrieve_context_for_explain(
        file_id=req.file_id,
        backend=VECTOR_BACKEND,
        page_number=req.page_number,
    )

    # 2) Prompt config riêng cho region explain
    prompt_cfg = PromptConfig(
        paper_id=req.file_id,
        system_instructions=REGION_EXPLAIN_INSTRUCTIONS,
        is_visual_explanation=True,
    )

    # 3) Gọi shared multimodal chain
    gen_chain = build_generative_chain()
    answer = gen_chain.invoke(
        {
            "context": context_data,
            "question": req.question or "Explain this region.",
            "prompt_cfg": prompt_cfg,
            "focus_image_b64": req.image_b64,
        }
    )

    # 4) Thêm crop image vào context trả về cho FE (debug / hiển thị)
    final_context = format_explain_context_for_chain(
        context_data, req.image_b64, req.page_number
    )

    return QueryResponse(answer=answer, context=final_context)


@app.post("/related-papers", response_model=RelatedPapersResponse)
async def related_papers(req: RelatedPapersRequest):
    """
    Gợi ý các paper liên quan trên arXiv cho file hiện tại.

    - Dùng title + abstract đã ingest làm "base paper"
    - Arxiv search + LLM re-rank (trong arxiv_related.suggest_related_papers)
    """
    validate_file_ready(req.file_id)

    meta = load_metadata(req.file_id)
    if meta is None:
        raise HTTPException(
            400,
            "No metadata found for this file; cannot suggest related papers.",
        )

    base_title = meta.title or ""
    base_abstract = meta.abstract or ""

    related = suggest_related_papers(
        base_title=base_title,
        base_abstract=base_abstract,
        categories=None,  # sau này có arxiv category thì truyền vào
        max_results=req.max_results,
        top_k=req.top_k,
    )

    results_models = [
        RelatedPaper(
            arxiv_id=item.get("arxiv_id", ""),
            title=item.get("title", ""),
            abstract=item.get("abstract", ""),
            authors=item.get("authors") or [],
            categories=item.get("categories") or [],
            url=item.get("url", ""),
            score=float(item.get("score", 0.0) or 0.0),
            reason=item.get("reason", ""),
        )
        for item in related
    ]

    return RelatedPapersResponse(
        file_id=req.file_id,
        base_title=base_title,
        base_abstract=base_abstract,
        results=results_models,
    )


@app.post("/brainstorm-questions", response_model=BrainstormResponse)
async def brainstorm_questions(req: BrainstormRequest):
    """
    Gợi ý các câu hỏi thông minh dựa trên nội dung Abstract của bài báo.
    """
    # 1. Kiểm tra file đã sẵn sàng chưa
    validate_file_ready(req.file_id)

    # 2. Lấy metadata (Title & Abstract)
    meta = load_metadata(req.file_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Metadata not found for brainstorming")

    # 3. Gọi LLM sinh câu hỏi
    # Lưu ý: Bạn có thể cache kết quả này vào METADATA_REGISTRY nếu không muốn gọi LLM nhiều lần cho cùng 1 file
    questions = brainstorm_questions_chain(
        title=meta.title or "Unknown Title",
        abstract=meta.abstract or "No abstract available."
    )

    return BrainstormResponse(questions=questions)


@app.get("/status/{file_id}")
async def get_file_status(file_id: str):
    status = get_status(file_id) or "unknown"
    return {
        "status": status,
        "ready": file_id in pipelines and status.startswith("completed"),
    }


@app.delete("/cleanup/{file_id}")
async def cleanup_file(file_id: str):
    """
    Cleanup all RAG data for a file.
    Called by Backend when a paper is deleted.

    Cleans up:
    - In-memory pipeline cache
    - rag_paper_cache table entry
    - paper_content_summaries table entries
    - ChromaDB vector store (optional, for disk space)
    """
    import shutil

    cleaned = {
        "pipeline_cache": False,
        "rag_paper_cache": False,
        "content_summaries": False,
        "vector_store": False,
    }

    # 1. Remove from in-memory pipeline cache
    if file_id in pipelines:
        del pipelines[file_id]
        cleaned["pipeline_cache"] = True

    # 2. Remove from in-memory metadata cache
    if file_id in file_metadata:
        del file_metadata[file_id]

    # 3. Cleanup database tables
    try:
        db = get_database()

        # Delete from paper_content_summaries
        db.delete_summaries(file_id)
        cleaned["content_summaries"] = True

        # Delete from rag_paper_cache
        db.delete_paper_cache(file_id)
        cleaned["rag_paper_cache"] = True
    except Exception as e:
        print(f"❌ Database cleanup failed for {file_id}: {e}")

    # 4. Delete ChromaDB vector store directory (optional, saves disk space)
    try:
        chroma_path = CHROMA_DIR / file_id
        if chroma_path.exists():
            shutil.rmtree(chroma_path)
            cleaned["vector_store"] = True
    except Exception as e:
        print(f"❌ ChromaDB cleanup failed for {file_id}: {e}")

    print(f"🧹 Cleanup completed for {file_id}: {cleaned}")
    return {"file_id": file_id, "cleaned": cleaned}


@app.get("/cleanup/orphaned-guests")
async def get_orphaned_guest_files(max_age_hours: int = 24):
    """
    Get list of orphaned guest files (exist in rag_paper_cache but not in papers).
    Used by Backend cleanup cron job.
    """
    try:
        db = get_database()
        orphaned_ids = db.get_orphaned_guest_files(max_age_hours)

        # Note: We don't have file URLs in rag_paper_cache
        # Backend will need to handle S3 cleanup separately or we need to enhance this
        files = [{"rag_paper_id": fid} for fid in orphaned_ids]

        return {
            "count": len(files),
            "max_age_hours": max_age_hours,
            "files": files,
        }
    except Exception as e:
        print(f"❌ Failed to get orphaned guest files: {e}")
        raise HTTPException(500, f"Failed to get orphaned files: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
