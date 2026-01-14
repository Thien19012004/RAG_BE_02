# -*- coding: utf-8 -*-
"""
Document ingestion pipeline responsible for:
- dual extraction (GROBID text + PyMuPDF/Camelot layout)
- multimodal summarization with caching
- semantic node construction
- writing abstract/content docs into the configured vector backend
"""
import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from typing import Any, Dict, List

from langchain_core.documents import Document

from pdf_extract import (
    build_semantic_nodes,
    run_grobid,
    attach_layout_to_nodes,
    extract_layout_blocks,
    extract_table_blocks,
    extract_image_blocks,
)
from summarization import (
    build_text_summarizer,
    build_vision_summarizer,
)
from parallel_processing import (
    summarize_texts_parallel,
    summarize_images_parallel,
)
from vectorstore_setup import (
    CONTENT_COLLECTION,
    VectorStoreBackend,
    get_embedding_model,
)


@dataclass
class IngestionResult:
    paper_id: str
    title: str
    abstract: str
    node_count: int
    table_count: int
    image_count: int
    metadata_path: str


def _save_metadata(cache_dir, payload: Dict[str, Any]) -> str:
    cache_dir.mkdir(exist_ok=True, parents=True)
    meta_path = cache_dir / "paper_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(meta_path)


def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure all metadata values are simple JSON / Chroma-compatible types:
    str, int, float, bool, None, or dict với value đơn giản.
    Lists và kiểu phức tạp khác sẽ stringify.
    """
    cleaned: Dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            cleaned[k] = v
        elif isinstance(v, dict):
            # Chroma không nhận dict; stringify JSON an toàn
            try:
                cleaned[k] = json.dumps(v)
            except Exception:
                cleaned[k] = str(v)
        elif isinstance(v, (list, tuple, set)):
            simple_vals = [x for x in v if isinstance(x, (str, int, float, bool))]
            cleaned[k] = ", ".join(str(x) for x in simple_vals) if simple_vals else None
        else:
            cleaned[k] = str(v)
    return cleaned

def _norm_text(s: str) -> List[str]:
    s = re.sub(r"\s+", " ", s.strip().lower())
    return [w for w in s.split(" ") if w]

def _overlap_score(a_words: List[str], b_words: List[str]) -> float:
    if not a_words or not b_words: return 0.0
    a_set, b_set = set(a_words), set(b_words)
    inter = len(a_set & b_set)
    return inter / max(1, len(a_set))

async def ingest_document(file_config, backend: VectorStoreBackend) -> IngestionResult:
    """Ingest a PDF into the abstract/content stores while keeping caches updated."""
    print(f"🔧 [PIPELINE] Start ingest for paper_id={file_config.file_id}")
    need_rebuild, pdf_hash = file_config.needs_rebuild()
    print(f"📄 [PIPELINE] PDF hash={pdf_hash[:8]}, need_rebuild={need_rebuild}")

    pdf_path_str = str(file_config.pdf_path)

    # -------------------------------------------------------------------------
    # 1) Dual extraction: GROBID (semantic sections) + PyMuPDF/Camelot layout
    # -------------------------------------------------------------------------
    with ThreadPoolExecutor(max_workers=4) as executor:
        loop = asyncio.get_event_loop()
        f_grobid = loop.run_in_executor(executor, run_grobid, pdf_path_str, file_config.file_id)
        f_layout = loop.run_in_executor(executor, extract_layout_blocks, pdf_path_str)
        f_tables = loop.run_in_executor(executor, extract_table_blocks, pdf_path_str)
        f_images = loop.run_in_executor(executor, extract_image_blocks, pdf_path_str)

        grobid_payload, layout_blocks, table_blocks, image_blocks = await asyncio.gather(
            f_grobid, f_layout, f_tables, f_images
        )
        
    print(
        f"🧠 [PIPELINE] GROBID sections={len(grobid_payload.sections)} "
        f"title={grobid_payload.title}"
    )
    print(
        "🧩 [PIPELINE] Layout blocks -> "
        f"tables={len(table_blocks)}, texts={len(layout_blocks)}, images={len(image_blocks)}"
    )

    # -------------------------------------------------------------------------
    # 2) Semantic nodes from GROBID sections + embedding
    #    (semantic chunking theo section)
    # -------------------------------------------------------------------------
    embedding_model = get_embedding_model()
    nodes = build_semantic_nodes(
        paper_id=file_config.file_id,
        sections=grobid_payload.sections,
        embed_fn=embedding_model.embed_documents,
    )
    print(f"🧱 [PIPELINE] Semantic nodes built={len(nodes)}")

    # align semantic nodes với layout (binary search/matching inside pdf_extract)
    # để lấy page + raw bbox (+ layout_width/height khi có)
    nodes = attach_layout_to_nodes(nodes, layout_blocks)

    # -------------------------------------------------------------------------
    # 3) Multimodal summarization (only tables + images). Text nodes keep raw text.
    # -------------------------------------------------------------------------
    text_summarizer = build_text_summarizer()
    vision_summarizer = build_vision_summarizer()

    table_task = summarize_texts_parallel(
        [tbl.plaintext for tbl in table_blocks],
        str(file_config.cache_dir / "table_summaries.json"),
        text_summarizer, use_cache=not need_rebuild
    )
    image_task = summarize_images_parallel(
        [img.image_b64 for img in image_blocks],
        str(file_config.cache_dir / "image_summaries.json"),
        vision_summarizer, use_cache=not need_rebuild
    )
    table_summaries, image_summaries = await asyncio.gather(table_task, image_task)
    print(
        "📈 [PIPELINE] Summaries -> "
        f"tables={len(table_summaries)}, images={len(image_summaries)}"
    )

    # -------------------------------------------------------------------------
    # 4) Ghi vào vector store (content + abstract)
    # -------------------------------------------------------------------------
    backend.delete_where(CONTENT_COLLECTION, {"paper_id": file_config.file_id})

    documents: List[Document] = []

    # 4a) Thêm Abstract như một node bình thường với modality="abstract"
    abstract_text = grobid_payload.abstract or ""
    abstract_page = None
    abstract_bbox = None

    if abstract_text.strip():
        abs_words = _norm_text(abstract_text[:500])
        best_score = 0.0
        for blk in layout_blocks:
            if blk.page_number > 2: continue
            score = _overlap_score(abs_words, _norm_text(blk.text))
            if score > best_score:
                best_score = score
                abstract_page, abstract_bbox = blk.page_number, blk.bbox
        
        if best_score < 0.2: abstract_page = 1

        abs_meta = _sanitize_metadata({
            "paper_id": file_config.file_id,
            "title": grobid_payload.title,
            "section_title": "Abstract",
            "modality": "abstract",
            "page_label": abstract_page,
            "bbox": abstract_bbox,
        })
        documents.append(Document(page_content=abstract_text, metadata=abs_meta))

    # 4b) Thêm Semantic Nodes
    for node in nodes:
        page_val = node.page_number or node.approx_page_start
        metadata = _sanitize_metadata({
            "paper_id": file_config.file_id,
            "section_title": node.section_title,
            "modality": "text",
            "page_label": page_val,
            "bbox": node.bbox,
        })
        documents.append(Document(page_content=node.text, metadata=metadata))

    # 4c) Tables – mỗi bảng có bbox riêng để highlight
    for idx, (tbl, summary) in enumerate(zip(table_blocks, table_summaries)):
        if not summary:
            continue
        page_number = getattr(tbl, "page_number", None)
        metadata = _sanitize_metadata(
            {
                "paper_id": file_config.file_id,
                "modality": "table",
                "section_title": getattr(tbl, "section_title", None)
                or f"Table {idx+1}",
                "page_label": page_number,
                "page_start": page_number,
                "page_end": page_number,
                "table_html": tbl.html,
                "bbox": getattr(tbl, "bbox", None),
            }
        )
        documents.append(Document(page_content=summary, metadata=metadata))

    # 4d) Images / Figures – kèm image_b64 + bbox để FE highlight region
    for idx, (img, summary) in enumerate(zip(image_blocks, image_summaries)):
        if not summary:
            continue
        metadata = _sanitize_metadata(
            {
                "paper_id": file_config.file_id,
                "modality": "image",
                "section_title": f"Figure {idx+1}",
                "page_label": getattr(img, "page_number", None),
                "image_b64": img.image_b64,
                "bbox": getattr(img, "bbox", None),
            }
        )
        documents.append(Document(page_content=summary, metadata=metadata))

    backend.add_documents(documents, CONTENT_COLLECTION)
    print(f"📦 [PIPELINE] Added {len(documents)} docs to content collection")


    # -------------------------------------------------------------------------
    # 6) Cache metadata + hash
    # -------------------------------------------------------------------------
    file_config.save_hash(pdf_hash)

    metadata_payload = {
        "paper_id": file_config.file_id,
        "title": grobid_payload.title,
        "authors": grobid_payload.authors,
        "abstract": abstract_text,
        "node_count": len(nodes),
        "table_count": len(table_blocks),
        "image_count": len(image_blocks),
    }
    metadata_path = _save_metadata(file_config.cache_dir, metadata_payload)

    return IngestionResult(
        paper_id=file_config.file_id,
        title=grobid_payload.title,
        abstract=abstract_text,
        node_count=len(nodes),
        table_count=len(table_blocks),
        image_count=len(image_blocks),
        metadata_path=metadata_path,
    )
