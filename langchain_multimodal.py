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
    print(f"🔧 [PIPELINE] Start building pipeline for file_id={file_config.file_id}")
    need_rebuild, pdf_hash = file_config.needs_rebuild()
    print(f"📄 [PIPELINE] PDF hash={pdf_hash[:8]}..., need_rebuild={need_rebuild}")

    # Extract content from PDF
    print(f"🗂️  [PIPELINE] Partition PDF into chunks: {file_config.pdf_path}")
    chunks = partition_pdf_into_chunks(str(file_config.pdf_path))
    print(f"🧩 [PIPELINE] Extracted chunks: {len(chunks)}")
    tables, texts = split_tables_and_texts(chunks)
    print(f"📊 [PIPELINE] Tables={len(tables)}, Texts(before clean)={len(texts)}")
    texts = remove_repeated_headers(texts)
    print(f"🧹 [PIPELINE] Texts(after header clean)={len(texts)}")
    images = get_images_base64(chunks)
    print(f"🖼️  [PIPELINE] Images extracted={len(images)}")

    # Build summarizers
    print("🧠 [PIPELINE] Build summarizers (text + vision)")
    text_summarizer = build_text_summarizer()
    vision_summarizer = build_vision_summarizer()

    # Prepare cache files
    cache_files = {
        "text_summaries": str(file_config.cache_dir / "text_summaries.json"),
        "table_summaries": str(file_config.cache_dir / "table_summaries.json"),
        "image_summaries": str(file_config.cache_dir / "image_summaries.json"),
    }
    print(f"🗃️  [PIPELINE] Cache files: {cache_files}")
    
    # Convert tables to HTML
    tables_html = [t.metadata.text_as_html for t in tables]
    print(f"🧾 [PIPELINE] Converted tables to HTML: {len(tables_html)}")
    
    # Process all content in parallel for better performance
    print("🚀 [PIPELINE] Start parallel summarization (texts/tables/images)")
    text_summaries, table_summaries, image_summaries = process_all_content_parallel(
        texts,
        tables_html,
        images,
        cache_files,
        text_summarizer,
        vision_summarizer,
        use_cache=not need_rebuild,
    )
    print(f"📈 [PIPELINE] Summaries: text={len(text_summaries)}, table={len(table_summaries)}, image={len(image_summaries)}")

    # Build vector store
    print(f"📦 [PIPELINE] Build/Load vector store at {file_config.chroma_path}")
    vectorstore, docstore = build_vectorstore(str(file_config.chroma_path))
    
    if need_rebuild:
        print("📗 [PIPELINE] Rebuilding Chroma store (adding documents)")
        add_group_to_store(vectorstore, docstore, texts, text_summaries)
        add_group_to_store(vectorstore, docstore, tables, table_summaries)
        add_group_to_store(vectorstore, docstore, images, image_summaries)
        index_path = str(file_config.cache_dir / "docstore_index.json")
        persist_docstore_index(docstore, index_path)
        print(f"💾 [PIPELINE] Persisted docstore index to {index_path}")
        file_config.save_hash(pdf_hash)
        print("🔐 [PIPELINE] Saved current PDF hash")
    else:
        print("📗 [PIPELINE] Using existing Chroma (no rebuild)")

    # Build retrieval function and RAG chains
    print("🔎 [PIPELINE] Build retrieval fn and RAG chains")
    retrieve_fn = lambda q: retrieve_parents_vs(vectorstore, docstore, q, k=6)
    rag_chain, rag_chain_with_ctx = build_rag_chains(retrieve_fn)
    print("✅ [PIPELINE] Pipeline build finished")
    
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
