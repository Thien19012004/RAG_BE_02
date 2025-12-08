# -*- coding: utf-8 -*-
"""
Document ingestion pipeline responsible for:
- dual extraction (GROBID text + PyMuPDF/Camelot layout)
- multimodal summarization with caching
- semantic node construction
- writing abstract/content docs into the configured vector backend
"""

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
    ABSTRACT_COLLECTION,
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


def ingest_document(file_config, backend: VectorStoreBackend) -> IngestionResult:
    """Ingest a PDF into the abstract/content stores while keeping caches updated."""
    print(f"🔧 [PIPELINE] Start ingest for paper_id={file_config.file_id}")
    need_rebuild, pdf_hash = file_config.needs_rebuild()
    print(f"📄 [PIPELINE] PDF hash={pdf_hash[:8]}, need_rebuild={need_rebuild}")

    pdf_path_str = str(file_config.pdf_path)

    # -------------------------------------------------------------------------
    # 1) Dual extraction: GROBID (semantic sections) + PyMuPDF/Camelot layout
    # -------------------------------------------------------------------------
    grobid_payload = run_grobid(pdf_path_str, paper_id=file_config.file_id)
    print(
        f"🧠 [PIPELINE] GROBID sections={len(grobid_payload.sections)} "
        f"title={grobid_payload.title}"
    )

    # Layout text blocks from PyMuPDF
    layout_blocks = extract_layout_blocks(pdf_path_str)
    # Tables from Camelot
    table_blocks = extract_table_blocks(pdf_path_str)
    # Figures/images from PyMuPDF
    image_blocks = extract_image_blocks(pdf_path_str)

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

    table_summaries = summarize_texts_parallel(
        [tbl.plaintext for tbl in table_blocks],
        str(file_config.cache_dir / "table_summaries.json"),
        text_summarizer,
        to_str=lambda x: x,
        use_cache=not need_rebuild,
        batch_size=6,
        max_workers=4,
    )

    image_summaries = summarize_images_parallel(
        [img.image_b64 for img in image_blocks],
        str(file_config.cache_dir / "image_summaries.json"),
        vision_summarizer,
        use_cache=not need_rebuild,
        batch_size=3,
        max_workers=2,
    )
    print(
        "📈 [PIPELINE] Summaries -> "
        f"tables={len(table_summaries)}, images={len(image_summaries)}"
    )

    # -------------------------------------------------------------------------
    # 4) Ghi vào vector store (content + abstract)
    # -------------------------------------------------------------------------
    backend.delete_where(CONTENT_COLLECTION, {"paper_id": file_config.file_id})
    backend.delete_where(ABSTRACT_COLLECTION, {"paper_id": file_config.file_id})

    documents: List[Document] = []

    # 4a) Ưu tiên semantic nodes: mỗi node là 1 retrieval unit có bbox
    if nodes:
        for node in nodes:
            page_label = (
                node.page_number
                or node.approx_page_start
                or node.approx_page_end
            )

            metadata = _sanitize_metadata(
                {
                    "paper_id": file_config.file_id,
                    "section_title": node.section_title,
                    "modality": "text",
                    "order_idx": node.order_idx,
                    "page_label": page_label,
                    "page_start": node.approx_page_start,
                    "page_end": node.approx_page_end,
                    # raw PDF bbox + layout_size (để FE normalize highlight)
                    "bbox": node.bbox,
                }
            )
            documents.append(Document(page_content=node.text, metadata=metadata))
    else:
        # 4b) Fallback khi không build được semantic nodes:
        # dùng raw layout block text làm retrieval unit, vẫn gắn page + bbox
        print(
            "⚠️ [PIPELINE] No semantic nodes; falling back to layout blocks "
            "as retrieval units."
        )
        for idx, blk in enumerate(layout_blocks):
            if not getattr(blk, "text", "").strip():
                continue
            page_label = getattr(blk, "page_number", None)
            metadata = _sanitize_metadata(
                {
                    "paper_id": file_config.file_id,
                    "section_title": f"Block {idx+1}",
                    "modality": "text",
                    "order_idx": idx,
                    "page_label": page_label,
                    "page_start": page_label,
                    "page_end": page_label,
                    "bbox": getattr(blk, "bbox", None),
                }
            )
            documents.append(Document(page_content=blk.text, metadata=metadata))

    # 4c) Tables (Camelot) – mỗi bảng có bbox riêng để highlight
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
    # 5) Abstract store: ưu tiên abstract từ GROBID
    # -------------------------------------------------------------------------
    if grobid_payload.abstract:
        abstract_text = grobid_payload.abstract
    elif nodes:
        abstract_text = nodes[0].text
    else:
        # Join a few non-empty table summaries to approximate an abstract
        non_empty_tables = [
            s for s in table_summaries if isinstance(s, str) and s.strip()
        ]
        joined = " ".join(non_empty_tables[:3])
        abstract_text = joined[:2000]

    abstract_metadata = _sanitize_metadata(
        {
            "paper_id": file_config.file_id,
            "title": grobid_payload.title,
            "authors": grobid_payload.authors,
            "section_title": "Abstract",
            "modality": "abstract",
        }
    )
    abstract_doc = Document(page_content=abstract_text, metadata=abstract_metadata)
    backend.add_documents([abstract_doc], ABSTRACT_COLLECTION)
    print("📚 [PIPELINE] Abstract added to abstract collection")

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
