# -*- coding: utf-8 -*-
"""
Document ingestion pipeline responsible for:
- dual extraction (GROBID text + unstructured layout)
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
    get_images_base64,
    partition_pdf_into_chunks,
    remove_repeated_headers,
    run_grobid,
    split_tables_and_texts,
    attach_layout_to_nodes,  # <- thêm
)
from summarization import (
    build_text_summarizer,
    build_vision_summarizer,
)
from parallel_processing import process_all_content_parallel
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
            # chỉ giữ dict đơn giản (không lồng object / list)
            simple_dict: Dict[str, Any] = {}
            for dk, dv in v.items():
                if isinstance(dv, (str, int, float, bool)) or dv is None:
                    simple_dict[dk] = dv
                else:
                    simple_dict[dk] = str(dv)
            cleaned[k] = simple_dict
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

    # Dual extraction -----------------------------------------------------
    grobid_payload = run_grobid(str(file_config.pdf_path), paper_id=file_config.file_id)
    print(f"🧠 [PIPELINE] GROBID sections={len(grobid_payload.sections)} title={grobid_payload.title}")

    chunks = partition_pdf_into_chunks(str(file_config.pdf_path))
    tables, texts = split_tables_and_texts(chunks)
    texts = remove_repeated_headers(texts)
    images = get_images_base64(chunks)
    print(
        f"🧩 [PIPELINE] Layout chunks -> tables={len(tables)}, texts={len(texts)}, images={len(images)}"
    )

    # Semantic nodes ------------------------------------------------------
    embedding_model = get_embedding_model()
    nodes = build_semantic_nodes(
        paper_id=file_config.file_id,
        sections=grobid_payload.sections,
        embed_fn=embedding_model.embed_documents,
    )
    print(f"🧱 [PIPELINE] Semantic nodes built={len(nodes)}")
    # 🔍 align với layout để lấy page + bbox
    nodes = attach_layout_to_nodes(nodes, texts)

    # Summarization -------------------------------------------------------
    text_summarizer = build_text_summarizer()
    vision_summarizer = build_vision_summarizer()
    cache_files = {
        "text_summaries": str(file_config.cache_dir / "text_summaries.json"),
        "table_summaries": str(file_config.cache_dir / "table_summaries.json"),
        "image_summaries": str(file_config.cache_dir / "image_summaries.json"),
    }
    text_summaries, table_summaries, image_summaries = process_all_content_parallel(
        texts,
        [t.metadata.text_as_html for t in tables],
        images,
        cache_files,
        text_summarizer,
        vision_summarizer,
        use_cache=not need_rebuild,
    )
    print(
        f"📈 [PIPELINE] Summaries -> text={len(text_summaries)}, tables={len(table_summaries)}, images={len(image_summaries)}"
    )

    # Vector store writes -------------------------------------------------
    backend.delete_where(CONTENT_COLLECTION, {"paper_id": file_config.file_id})
    backend.delete_where(ABSTRACT_COLLECTION, {"paper_id": file_config.file_id})

    documents: List[Document] = []

    # Prefer semantic nodes if available; otherwise, fall back to text summaries
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
                    "bbox": node.bbox,  # <- raw PDF bbox + layout_size
                }
            )
            documents.append(Document(page_content=node.text, metadata=metadata))
    else:
        # Fallback path when GROBID sections / semantic nodes are unavailable.
        print("⚠️ [PIPELINE] No semantic nodes; falling back to text_summaries as retrieval units.")
        for idx, (orig, summary) in enumerate(zip(texts, text_summaries)):
            if not summary:
                continue
            section_title = getattr(getattr(orig, "metadata", None), "section", None)
            page_label = getattr(getattr(orig, "metadata", None), "page_number", None)
            metadata = _sanitize_metadata(
                {
                    "paper_id": file_config.file_id,
                    "section_title": section_title or f"Chunk {idx+1}",
                    "modality": "text",
                    "order_idx": idx,
                    "page_label": page_label,
                    "page_start": page_label,
                    "page_end": page_label,
                }
            )
            documents.append(Document(page_content=summary, metadata=metadata))

    for table, summary in zip(tables, table_summaries):
        if not summary:
            continue
        page_number = getattr(table.metadata, "page_number", None)
        metadata = _sanitize_metadata(
            {
                "paper_id": file_config.file_id,
                "modality": "table",
                "section_title": getattr(table.metadata, "section", None),
                "page_label": page_number,
                "page_start": page_number,
                "page_end": page_number,
                "table_html": table.metadata.text_as_html,
            }
        )
        documents.append(Document(page_content=summary, metadata=metadata))

    for image_b64, summary in zip(images, image_summaries):
        if not summary:
            continue
        metadata = _sanitize_metadata(
            {
            "paper_id": file_config.file_id,
            "modality": "image",
            "section_title": "Figure",
            "page_label": None,
            "image_b64": image_b64,
            }
        )
        documents.append(Document(page_content=summary, metadata=metadata))

    backend.add_documents(documents, CONTENT_COLLECTION)
    print(f"📦 [PIPELINE] Added {len(documents)} docs to content collection")

    # Abstract text: prefer GROBID abstract, then first semantic node,
    # finally fall back to concatenated text summaries.
    if grobid_payload.abstract:
        abstract_text = grobid_payload.abstract
    elif nodes:
        abstract_text = nodes[0].text
    else:
        # Join a few non-empty summaries to approximate an abstract
        non_empty_summaries = [s for s in text_summaries if isinstance(s, str) and s.strip()]
        joined = " ".join(non_empty_summaries[:5])
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

    file_config.save_hash(pdf_hash)

    metadata_payload = {
        "paper_id": file_config.file_id,
        "title": grobid_payload.title,
        "authors": grobid_payload.authors,
        "abstract": abstract_text,
        "node_count": len(nodes),
        "table_count": len(tables),
        "image_count": len(images),
    }
    metadata_path = _save_metadata(file_config.cache_dir, metadata_payload)

    return IngestionResult(
        paper_id=file_config.file_id,
        title=grobid_payload.title,
        abstract=abstract_text,
        node_count=len(nodes),
        table_count=len(tables),
        image_count=len(images),
        metadata_path=metadata_path,
    )
