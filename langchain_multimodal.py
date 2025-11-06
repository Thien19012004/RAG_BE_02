# -*- coding: utf-8 -*-
"""
LangChain Multi-modal RAG Pipeline
- Groq for text/table summarization
- OpenAI gpt-4o-mini for image understanding
- OpenAI embeddings + Chroma for retrieval
- Automatic caching for summaries + embeddings
"""

from pdf_extract import (
    partition_pdf_into_chunks,
    split_tables_and_texts,
    remove_repeated_headers,
    get_images_base64,
)
from summarization import (
    build_text_summarizer,
    build_vision_summarizer,
    TEXT_SLEEP_SECONDS,
    VISION_SLEEP_SECONDS,
)
from parallel_processing import process_all_content_parallel
from vectorstore_setup import (
    build_vectorstore,
    add_group_to_store,
    persist_docstore_index,
    retrieve_parents as retrieve_parents_vs,
)
from rag_pipeline import build_rag_chains


def build_pipeline(file_config):
    """Build RAG pipeline for a specific file"""
    need_rebuild, pdf_hash = file_config.needs_rebuild()
    print("🔍 PDF changed:", need_rebuild)

    # Extract content from PDF
    chunks = partition_pdf_into_chunks(str(file_config.pdf_path))
    tables, texts = split_tables_and_texts(chunks)
    texts = remove_repeated_headers(texts)
    images = get_images_base64(chunks)

    # Build summarizers
    text_summarizer = build_text_summarizer()
    vision_summarizer = build_vision_summarizer()

    # Prepare cache files
    cache_files = {
        "text_summaries": str(file_config.cache_dir / "text_summaries.json"),
        "table_summaries": str(file_config.cache_dir / "table_summaries.json"),
        "image_summaries": str(file_config.cache_dir / "image_summaries.json"),
    }
    
    # Convert tables to HTML
    tables_html = [t.metadata.text_as_html for t in tables]
    
    # Process all content in parallel for better performance
    text_summaries, table_summaries, image_summaries = process_all_content_parallel(
        texts,
        tables_html,
        images,
        cache_files,
        text_summarizer,
        vision_summarizer,
        use_cache=not need_rebuild,
    )

    # Build vector store
    vectorstore, docstore = build_vectorstore(str(file_config.chroma_path))
    
    if need_rebuild:
        print("📗 Rebuilding Chroma store...")
        add_group_to_store(vectorstore, docstore, texts, text_summaries)
        add_group_to_store(vectorstore, docstore, tables, table_summaries)
        add_group_to_store(vectorstore, docstore, images, image_summaries)
        persist_docstore_index(docstore, str(file_config.cache_dir / "docstore_index.json"))
        file_config.save_hash(pdf_hash)
    else:
        print("📗 Using existing Chroma (no rebuild)")

    # Build retrieval function and RAG chains
    retrieve_fn = lambda q: retrieve_parents_vs(vectorstore, docstore, q, k=6)
    rag_chain, rag_chain_with_ctx = build_rag_chains(retrieve_fn)
    
    return rag_chain, rag_chain_with_ctx


# ================= DEMO =================
if __name__ == "__main__":
    from config import FileConfig
    
    # Demo with a test file
    file_config = FileConfig("test_file")
    rag_chain, rag_chain_with_ctx = build_pipeline(file_config)

    q1 = "What is the main idea of the paper?"
    print("\nQ:", q1)
    print("A:", rag_chain.invoke(q1))

    q2 = "What are the most frequent words?"
    print("\nQ:", q2)
    out = rag_chain_with_ctx.invoke(q2)
    print("A:", out["response"])

    print("\n--- Context preview ---")
    for t in out["context"]["texts"][:2]:
        try:
            print((getattr(t, "text", "") or "")[:400], "\n-----")
        except Exception:
            pass
