# -*- coding: utf-8 -*-
"""FastAPI application for RAG system"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import FileConfig, CHROMA_DIR, CACHE_DIR, CONTENT_DIR, settings
from langchain_multimodal import ingest_document, IngestionResult
from vectorstore_setup import build_local_chroma_backend
from rag_pipeline import (
    build_document_rag_chain,
    build_generative_chain,
    brainstorm_questions_chain,
    PromptConfig,
    REGION_EXPLAIN_INSTRUCTIONS,
)
from api_utils import (
    retrieve_context_for_explain,
    format_explain_context_for_chain,
)
from arxiv_related import suggest_related_papers


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
# Global backends (in-memory) + JSON persistence
# -----------------------------------------------------------------------------
VECTOR_BACKEND = build_local_chroma_backend(str(CHROMA_DIR / "global_store"))

# Pipelines giữ in-memory (có thể build lại từ vector store)
pipelines: Dict[str, Any] = {}

# 2 dict này sẽ được "backup" xuống file JSON để sau restart vẫn còn
file_metadata: Dict[str, IngestionResult] = {}
processing_status: Dict[str, str] = {}

STATUS_REGISTRY = CACHE_DIR / "status_registry.json"
METADATA_REGISTRY = CACHE_DIR / "metadata_registry.json"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def set_status(file_id: str, status: str) -> None:
    """Update status in memory + persist ra STATUS_REGISTRY."""
    processing_status[file_id] = status
    data = _load_json(STATUS_REGISTRY)
    data[file_id] = status
    _save_json(STATUS_REGISTRY, data)


def get_status(file_id: str) -> Optional[str]:
    """Lấy status từ cache hoặc từ file JSON."""
    if file_id in processing_status:
        return processing_status[file_id]
    data = _load_json(STATUS_REGISTRY)
    status = data.get(file_id)
    if status is not None:
        processing_status[file_id] = status
    return status


def save_metadata(file_id: str, res: IngestionResult) -> None:
    """Lưu nhẹ metadata của IngestionResult ra JSON (fake DB)."""
    file_metadata[file_id] = res
    data = _load_json(METADATA_REGISTRY)
    data[file_id] = {
        "paper_id": res.paper_id,
        "title": res.title,
        "abstract": res.abstract,
        "node_count": res.node_count,
        "table_count": res.table_count,
        "image_count": res.image_count,
        "metadata_path": res.metadata_path,
    }
    _save_json(METADATA_REGISTRY, data)


def load_metadata(file_id: str) -> Optional[IngestionResult]:
    """
    Lấy IngestionResult từ cache hoặc từ file.
    Đây đóng vai trò stand-in cho database metadata sau này.
    """
    if file_id in file_metadata:
        return file_metadata[file_id]

    # Thử lấy từ registry JSON
    data = _load_json(METADATA_REGISTRY)
    payload = data.get(file_id)
    if payload is None:
        # Fallback: đọc paper_metadata.json gốc trong cache/<file_id>/
        cfg = FileConfig(file_id)
        meta_path = cfg.cache_dir / "paper_metadata.json"
        if not meta_path.exists():
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            return None
        res = IngestionResult(
            paper_id=meta.get("paper_id", file_id),
            title=meta.get("title", ""),
            abstract=meta.get("abstract", ""),
            node_count=meta.get("node_count", 0),
            table_count=meta.get("table_count", 0),
            image_count=meta.get("image_count", 0),
            metadata_path=str(meta_path),
        )
        file_metadata[file_id] = res
        return res

    res = IngestionResult(
        paper_id=payload.get("paper_id", file_id),
        title=payload.get("title", ""),
        abstract=payload.get("abstract", ""),
        node_count=payload.get("node_count", 0),
        table_count=payload.get("table_count", 0),
        image_count=payload.get("image_count", 0),
        metadata_path=payload.get("metadata_path", ""),
    )
    file_metadata[file_id] = res
    return res


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

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def validate_file_ready(file_id: str) -> None:
    status = get_status(file_id)
    if status is None or status == "unknown":
        raise HTTPException(404, "File not found")
    if not status.startswith("completed"):
        # đang xử lý hoặc lỗi
        if status.startswith("error"):
            raise HTTPException(500, f"File processing failed: {status}")
        raise HTTPException(202, "File processing not finished")


async def build_pipeline_sync(file_config: FileConfig) -> IngestionResult:
    """
    Chạy ingest + build RAG chain cho 1 PDF, theo kiểu *đồng bộ*.
    Được gọi trong /upload – chỉ trả response khi xong.

    Returns:
        IngestionResult with metadata from ingestion
    """
    try:
        set_status(file_config.file_id, "processing")

        # Ensure file is downloaded if from cloud
        file_config.ensure_local_file()

        res = await ingest_document(file_config, VECTOR_BACKEND)

        # Build query chain chuẩn
        prompt_cfg = PromptConfig(
            paper_id=file_config.file_id,
            paper_title=res.title,
        )
        chain = build_document_rag_chain(file_config.file_id, VECTOR_BACKEND, prompt_cfg)

        pipelines[file_config.file_id] = chain
        save_metadata(file_config.file_id, res)
        set_status(file_config.file_id, "completed")

        return res
    except Exception as e:
        print(f"Error while building pipeline for {file_config.file_id}: {e}")
        set_status(file_config.file_id, f"error: {str(e)}")
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
    Upload PDF và build pipeline *đồng bộ*.

    - Request: multipart/form-data với file + optional file_id
    - Response: UploadResponse(message, file_id, status, processing_time, metadata)
    """
    fid = file_id or str(uuid.uuid4())
    file_config = FileConfig(fid)

    # Lưu file vào local (original behavior)
    with open(file_config._local_pdf_path, "wb") as buffer:
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

    This endpoint is designed for Backend integration:
    1. Backend uploads PDF to S3
    2. Backend calls this endpoint with the S3 URL
    3. RAG service downloads and processes the PDF

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

        # Optionally cleanup local file to save space
        # file_config.cleanup_local_file()

        return UploadResponse(
            message="Ingested from URL successfully",
            file_id=fid,
            status="completed",
            processing_time=elapsed,
            title=res.title,
            abstract=res.abstract,
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
