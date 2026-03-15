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
from vectorstore_setup import build_local_chroma_backend, CONTENT_COLLECTION
from rag_pipeline import (
    build_document_rag_chain,
    build_generative_chain,
    brainstorm_questions_chain,
    PromptConfig,
    REGION_EXPLAIN_INSTRUCTIONS,
    split_docs,
    multi_document_retrieve,
    document_retrieve,
    check_grounding,
    UNGROUNDED_INSTRUCTIONS,
    DEFAULT_RAG_INSTRUCTIONS,
    rrf_merge,
    condense_question,
    summarize_conversation,
    decompose_multi_query,
)
from api_utils import (
    retrieve_context_for_explain,
    format_explain_context_for_chain,
)
from arxiv_related import suggest_related_papers
from kb_classifier import classify_paper as classify_paper_fn
from arxiv_lookup import classify_with_lookup

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
    chat_history: Optional[List[Dict[str, str]]] = None
    summary: Optional[str] = None
    custom_prompts: Optional[Dict[str, str]] = None  # {rag_instructions, condense_question}


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
    collection: Optional[str] = None  # target collection, e.g. "system_knowledge_base"


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
    text_input: Optional[str] = None  # User's text hint for question generation

class BrainstormResponse(BaseModel):
    questions: List[str]


class SummaryRequest(BaseModel):
    file_id: str


class SummaryResponse(BaseModel):
    file_id: str
    summary: str


# ---- Freeform AI generation (used by notebook ask-AI tool) ----
class GenerateRequest(BaseModel):
    prompt: str


class GenerateResponse(BaseModel):
    answer: str



class MultiQueryRequest(BaseModel):
    """Request to query across multiple PDFs."""
    question: str
    file_ids: List[str]  # List of file IDs to search across
    paper_summaries: Optional[Dict[str, str]] = None  # paper_id -> summary text


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


async def build_pipeline_sync(file_config: FileConfig, collection: str = CONTENT_COLLECTION) -> IngestionResult:
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

        res = await ingest_document(file_config, VECTOR_BACKEND, collection=collection)

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
        res = await build_pipeline_sync(file_config, collection=req.collection or CONTENT_COLLECTION)
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

    # If custom rag_instructions provided, rebuild chain with custom system prompt
    custom_prompts = req.custom_prompts or {}
    if custom_prompts.get("rag_instructions"):
        res = load_metadata(req.file_id)
        custom_prompt_cfg = PromptConfig(
            paper_id=req.file_id,
            paper_title=res.title if res else "",
            system_instructions=custom_prompts["rag_instructions"],
        )
        chain = build_document_rag_chain(req.file_id, VECTOR_BACKEND, custom_prompt_cfg)
        # Don't cache custom chains — rebuild each time

    # Condense follow-up question using chat history + rolling summary
    effective_question = req.question
    if req.chat_history or req.summary:
        effective_question = condense_question(
            chat_history=req.chat_history,
            question=req.question,
            summary=req.summary or "",
            custom_condense_prompt=custom_prompts.get("condense_question", ""),
        )

    out = chain.invoke({"question": effective_question})
    # rag_pipeline trả {"response": answer, "context": ctx}
    return QueryResponse(answer=out["response"], context=out.get("context"))


# --- Summarize Memory (Rolling Summary) ---
class SummarizeMemoryRequest(BaseModel):
    old_summary: str = ""
    messages: List[Dict[str, str]] = []

class SummarizeMemoryResponse(BaseModel):
    summary: str

@app.post("/summarize-memory", response_model=SummarizeMemoryResponse)
async def summarize_memory(req: SummarizeMemoryRequest):
    """Summarize overflow messages for rolling conversation memory."""
    result = summarize_conversation(req.old_summary, req.messages)
    return SummarizeMemoryResponse(summary=result)


# new endpoint for general AI text generation without paper context
@app.post("/generate", response_model=GenerateResponse)
async def generate_text(req: GenerateRequest):
    """Return freeform text generated by the LLM. This is used by the front-end "Ask AI" tool.

    The model will mirror the language of the user's prompt when producing output.
    """
    # build a generative chain with language-aware instructions
    chain = build_generative_chain()
    prompt_cfg = PromptConfig(
        system_instructions=(
            DEFAULT_RAG_INSTRUCTIONS +
            "\n\nWhen generating text outside of any paper context, respond in the same language as the user's prompt."
        )
    )
    answer = chain.invoke(
        {
            "context": {},
            "question": req.prompt,
            "prompt_cfg": prompt_cfg,
            "focus_image_b64": None,
        }
    )
    # the generative chain returns plain string
    if isinstance(answer, dict) and "response" in answer:
        # sometimes the chain might wrap response
        answer_text = answer.get("response", "")
    else:
        answer_text = str(answer)
    return GenerateResponse(answer=answer_text)


@app.post("/query-multi", response_model=MultiQueryResponse)
async def query_multi_pdf(req: MultiQueryRequest):
    """
    Query across multiple PDFs with smart query decomposition.

    For comparative/meta questions, decomposes into sub-queries
    and injects paper summaries for better cross-paper analysis.
    """
    if not req.file_ids:
        raise HTTPException(400, "At least one file_id is required")

    if len(req.file_ids) > 10:
        raise HTTPException(400, "Maximum 10 papers can be queried at once")

    # Validate all files are ready and load metadata
    paper_titles = {}
    for file_id in req.file_ids:
        validate_file_ready(file_id)
        meta = load_metadata(file_id)
        if meta:
            paper_titles[file_id] = meta.title or f"Paper {file_id[:8]}"
        else:
            paper_titles[file_id] = f"Paper {file_id[:8]}"

    # ── Step 1: Decompose the question ──
    decomposition = decompose_multi_query(req.question, paper_titles)
    query_type = decomposition["type"]
    sub_queries = decomposition["sub_queries"]
    needs_summaries = decomposition["needs_summaries"]

    print(
        f"\n{'='*60}\n"
        f"[QueryMulti] ===== DECOMPOSITION RESULT =====\n"
        f"  Original question: {req.question}\n"
        f"  Detected type: {query_type}\n"
        f"  Sub-queries ({len(sub_queries)}): {sub_queries}\n"
        f"  Needs summaries: {needs_summaries}\n"
        f"  Paper summaries provided: {bool(req.paper_summaries)}\n"
        f"{'='*60}"
    )

    # ── Step 2: Smart retrieval using sub-queries ──
    all_docs = []
    for sub_query in sub_queries:
        docs = multi_document_retrieve(
            query=sub_query,
            paper_ids=req.file_ids,
            backend=VECTOR_BACKEND,
            k_per_paper=6,
            total_k=15,
        )
        all_docs.extend(docs)

    # Deduplicate by content across sub-query results
    seen_content: set = set()
    unique_docs = []
    for doc in all_docs:
        content_key = doc.page_content[:150].strip()
        if content_key not in seen_content:
            unique_docs.append(doc)
            seen_content.add(content_key)
    all_docs = unique_docs[:20]  # Cap at 20 to avoid context overflow

    # ── Step 3: Grounding check (skip for COMPARATIVE/SUMMARY with summaries) ──
    has_grounding = check_grounding(all_docs)
    has_summaries = bool(req.paper_summaries and any(req.paper_summaries.values()))

    print(
        f"[QueryMulti] Step 3: has_grounding={has_grounding}, "
        f"has_summaries={has_summaries}, needs_summaries={needs_summaries}, "
        f"docs_count={len(all_docs)}, query_type={query_type}"
    )

    if not has_grounding and not (needs_summaries and has_summaries):
        print(f"[QueryMulti] >>> FALLBACK to general knowledge (no grounding AND no summaries)")
        # No chunks AND no summaries → general knowledge fallback
        ungrounded_cfg = PromptConfig(
            paper_id=None,
            system_instructions=UNGROUNDED_INSTRUCTIONS,
        )
        empty_context = {"texts": [], "tables": [], "images": []}
        gen_chain = build_generative_chain()
        answer = gen_chain.invoke({
            "context": empty_context,
            "question": req.question,
            "prompt_cfg": ungrounded_cfg,
            "focus_image_b64": None,
        })
        return MultiQueryResponse(
            answer=answer,
            context=empty_context,
            sources=[],
        )

    # ── Step 4: Build context ──
    context = split_docs(all_docs) if all_docs else {"texts": [], "tables": [], "images": []}

    # Inject paper summaries into context for COMPARATIVE/SUMMARY queries
    if needs_summaries and req.paper_summaries:
        summary_texts = []
        for idx, (file_id, summary) in enumerate(req.paper_summaries.items()):
            if summary and summary.strip():
                title = paper_titles.get(file_id, f"Paper {file_id[:8]}")
                summary_texts.append({
                    "text": f"[Paper Overview: {title}]\n{summary}",
                    "source_id": f"S{idx + 1}",
                    "page": "overview",
                    "metadata": {
                        "paper_id": file_id,
                        "source_paper_id": file_id,
                        "section_title": f"Paper Overview: {title}",
                        "type": "summary",
                    },
                })
        # Prepend summaries so LLM sees overviews first
        context["texts"] = summary_texts + context.get("texts", [])

    # ── Step 5: Build prompt ──
    paper_list = ", ".join([f'"{title}"' for title in paper_titles.values()])

    if query_type == "COMPARATIVE":
        multi_paper_instructions = (
            f"You are analyzing and COMPARING multiple research papers: {paper_list}. "
            "The user is asking a comparative/relational question. "
            "You have been provided with:\n"
            "1. Paper overviews/summaries (marked as [Paper Overview: ...])\n"
            "2. Specific text chunks from each paper\n\n"
            "Use BOTH to provide a thorough comparison. "
            "Clearly distinguish findings from each paper. "
            "When citing specific claims, use [S1], [S2], etc. "
            "Structure your answer to highlight similarities, differences, and relationships.\n\n"
        )
    elif query_type == "SUMMARY":
        multi_paper_instructions = (
            f"You are summarizing multiple research papers: {paper_list}. "
            "You have paper overviews and detailed chunks. "
            "Provide a comprehensive overview covering each paper's main contributions, "
            "methodology, and key findings. Use [S1], [S2] for specific citations.\n\n"
        )
    else:
        multi_paper_instructions = (
            f"You are analyzing multiple research papers: {paper_list}. "
            "Use the provided context from ALL papers to answer the question. "
            "When citing, indicate which paper the information comes from using [S1], [S2], etc. "
            "If comparing papers, clearly distinguish findings from each source. "
            "Synthesize information across papers when relevant.\n\n"
        )

    multi_paper_instructions += (
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
        paper_id=None,
        system_instructions=multi_paper_instructions,
    )

    # ── Step 6: Generate answer ──
    gen_chain = build_generative_chain()
    answer = gen_chain.invoke({
        "context": context,
        "question": req.question,
        "prompt_cfg": prompt_cfg,
        "focus_image_b64": None,
    })

    # Build sources info for response
    sources = []
    seen_papers: set = set()
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
    Nếu user cung cấp text_input, câu hỏi sẽ được sinh ra phù hợp với
    ý định của user và nội dung bài báo.
    """
    # 1. Kiểm tra file đã sẵn sàng chưa
    validate_file_ready(req.file_id)

    # 2. Lấy metadata (Title & Abstract)
    meta = load_metadata(req.file_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Metadata not found for brainstorming")

    # 3. Retrieve relevant context if text_input provided
    relevant_context = None
    if req.text_input and req.text_input.strip():
        try:
            docs = document_retrieve(
                query=req.text_input,
                paper_id=req.file_id,
                backend=VECTOR_BACKEND,
                k=5,
            )
            context_parts = split_docs(docs)
            context_texts = []
            for item in context_parts.get("texts", []) + context_parts.get("tables", []):
                context_texts.append(item.get("text", ""))
            relevant_context = "\n\n".join(context_texts[:5])
        except Exception as e:
            print(f"Warning: Failed to retrieve context for brainstorm: {e}")

    # 4. Gọi LLM sinh câu hỏi
    questions = brainstorm_questions_chain(
        title=meta.title or "Unknown Title",
        abstract=meta.abstract or "No abstract available.",
        text_input=req.text_input,
        relevant_context=relevant_context,
    )

    return BrainstormResponse(questions=questions)


@app.post("/summarize-paper", response_model=SummaryResponse)
async def summarize_paper(req: SummaryRequest):
    """
    Generate a comprehensive summary of the paper using its full content.
    Retrieves key sections (abstract, introduction, conclusion, methodology)
    and uses LLM to produce a structured summary.
    """
    validate_file_ready(req.file_id)

    meta = load_metadata(req.file_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Metadata not found for summarization")

    # Retrieve broad context from the paper for summarization
    summary_queries = [
        "abstract introduction background",
        "methodology approach method proposed",
        "results experiments evaluation performance",
        "conclusion future work contributions",
    ]

    all_context_texts = []
    seen_content = set()

    for query in summary_queries:
        try:
            docs = document_retrieve(
                query=query,
                paper_id=req.file_id,
                backend=VECTOR_BACKEND,
                k=6,
            )
            context_parts = split_docs(docs)
            for item in context_parts.get("texts", []) + context_parts.get("tables", []):
                text = item.get("text", "").strip()
                content_key = text[:150]
                if content_key not in seen_content and len(text) > 30:
                    seen_content.add(content_key)
                    section = item.get("metadata", {}).get("section_title", "")
                    all_context_texts.append(f"[{section}] {text}" if section else text)
        except Exception as e:
            print(f"Warning: Failed to retrieve context for query '{query}': {e}")

    if not all_context_texts:
        raise HTTPException(status_code=500, detail="Could not retrieve enough content for summarization")

    # Build summary using LLM
    from rag_pipeline import summarize_paper_chain
    summary = summarize_paper_chain(
        title=meta.title or "Unknown Title",
        abstract=meta.abstract or "",
        context="\n\n".join(all_context_texts[:20]),  # Limit context size
    )

    return SummaryResponse(file_id=req.file_id, summary=summary)


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


# ───────── F1: Paper Classification ─────────────────────────────

class ClassifyRequest(BaseModel):
    title: str
    abstract: str = ""
    doi: Optional[str] = None

@app.post("/classify-paper")
async def classify_paper_endpoint(req: ClassifyRequest):
    """
    Classify a paper with 3-tier strategy:
    1. DOI → CrossRef (2s)
    2. Title → ArXiv API (5s)
    3. LLM fallback (remaining time within 10s total)
    """
    try:
        # Tier 1 & 2: DOI/ArXiv lookup
        lookup_result = await classify_with_lookup(
            title=req.title,
            abstract=req.abstract,
            doi=req.doi,
            total_timeout=10.0,
        )
        if lookup_result is not None:
            return lookup_result

        # Tier 3: LLM fallback
        result = classify_paper_fn(req.title, req.abstract)
        result["source"] = "llm"
        return result
    except Exception as e:
        raise HTTPException(500, f"Classification failed: {str(e)}")


# ───────── F5: KB Explorer / Inspection ─────────────────────────

@app.get("/inspect/collection-stats")
async def get_collection_stats():
    """Get statistics about the vector store collections."""
    try:
        collections_info = {}
        for col_name in ["content_store", "system_knowledge_base"]:
            try:
                store = VECTOR_BACKEND._get_store(col_name)
                raw_col = store._collection
                data = raw_col.get(include=["metadatas"])
                total = len(data["ids"])

                by_modality: dict = {}
                by_category: dict = {}
                paper_ids = set()
                for meta in data.get("metadatas", []):
                    if meta:
                        mod = meta.get("modality", "unknown")
                        by_modality[mod] = by_modality.get(mod, 0) + 1
                        cat = meta.get("category", "uncategorized")
                        by_category[cat] = by_category.get(cat, 0) + 1
                        pid = meta.get("paper_id")
                        if pid:
                            paper_ids.add(pid)

                collections_info[col_name] = {
                    "total_chunks": total,
                    "total_papers": len(paper_ids),
                    "by_modality": by_modality,
                    "by_category": by_category,
                }
            except Exception:
                collections_info[col_name] = {
                    "total_chunks": 0,
                    "total_papers": 0,
                    "by_modality": {},
                    "by_category": {},
                }

        return collections_info
    except Exception as e:
        raise HTTPException(500, f"Failed to get stats: {str(e)}")


@app.get("/inspect/chunks")
async def get_chunks(
    paper_id: Optional[str] = None,
    category: Optional[str] = None,
    collection: str = "content_store",
    page: int = 1,
    limit: int = 20,
):
    """Get paginated chunks from vector store."""
    try:
        store = VECTOR_BACKEND._get_store(collection)
        raw_col = store._collection

        # Build where filter
        where_filter = None
        conditions = []
        if paper_id:
            conditions.append({"paper_id": paper_id})
        if category:
            conditions.append({"category": category})
        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

        data = raw_col.get(
            where=where_filter,
            include=["documents", "metadatas"],
        )

        total = len(data["ids"])
        start = (page - 1) * limit
        end = start + limit

        chunks = []
        for i in range(start, min(end, total)):
            chunks.append({
                "id": data["ids"][i],
                "content": data["documents"][i][:300] if data["documents"][i] else "",
                "metadata": data["metadatas"][i] if data["metadatas"] else {},
            })

        return {
            "chunks": chunks,
            "total": total,
            "page": page,
            "limit": limit,
            "totalPages": (total + limit - 1) // limit,
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to get chunks: {str(e)}")


@app.get("/inspect/duplicates")
async def get_duplicates():
    """Find papers with duplicate content hashes."""
    try:
        store = VECTOR_BACKEND._get_store("content_store")
        raw_col = store._collection
        data = raw_col.get(include=["metadatas"])

        hash_map: dict = {}
        for meta in data.get("metadatas", []):
            if meta:
                file_hash = meta.get("file_hash") or meta.get("content_hash")
                paper_id = meta.get("paper_id")
                if file_hash and paper_id:
                    if file_hash not in hash_map:
                        hash_map[file_hash] = set()
                    hash_map[file_hash].add(paper_id)

        duplicates = [
            {"hash": h, "paper_ids": list(pids)}
            for h, pids in hash_map.items()
            if len(pids) > 1
        ]

        return {"duplicates": duplicates, "total": len(duplicates)}
    except Exception as e:
        raise HTTPException(500, f"Failed to check duplicates: {str(e)}")


@app.delete("/papers/{paper_id}/chunks")
async def delete_paper_chunks(paper_id: str):
    """Delete all chunks for a paper from all vector store collections."""
    try:
        deleted_from = []
        for col_name in ["content_store", "system_knowledge_base"]:
            try:
                VECTOR_BACKEND.delete_where(col_name, {"paper_id": paper_id})
                deleted_from.append(col_name)
            except Exception:
                pass  # Collection may not exist or paper not in it

        # Clean up pipeline cache
        if paper_id in pipelines:
            del pipelines[paper_id]

        # Clean up file hash from RAG database
        try:
            db = get_database()
            db.delete_file_hash(paper_id)
        except Exception:
            pass

        return {
            "success": True,
            "paper_id": paper_id,
            "deleted_from": deleted_from,
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to delete paper chunks: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

#Happy new year 2026!