# -*- coding: utf-8 -*-
"""
LangChain Multi-modal RAG (with cache)
- Groq for text/table summarization
- OpenAI gpt-4o-mini for image understanding
- OpenAI embeddings + Chroma for retrieval
- Automatic caching for summaries + embeddings
"""

import config as cfg

from pdf_extract import (
    partition_pdf_into_chunks,
    split_tables_and_texts,
    remove_repeated_headers,
    get_images_base64,
)
from summarization import (
    build_text_summarizer,
    build_vision_summarizer,
    summarize_with_cache,
    summarize_images_with_cache,
    TEXT_SLEEP_SECONDS,
    VISION_SLEEP_SECONDS,
)
from vectorstore_setup import (
    build_vectorstore,
    add_group_to_store,
    persist_docstore_index,
    retrieve_parents as retrieve_parents_vs,
)
from rag_pipeline import build_rag_chains


def _build_pipeline():
    need_rebuild, pdf_hash = cfg.compute_rebuild_flag(cfg.PDF_PATH, cfg.HASH_FILE)
    print("🔍 PDF changed:", need_rebuild)

    chunks = partition_pdf_into_chunks(cfg.PDF_PATH)
    #print(chunks)
    tables, texts = split_tables_and_texts(chunks)
    #print(tables)
    #print(texts)
    texts = remove_repeated_headers(texts)
    #print(texts)
    images = get_images_base64(chunks)
    #print(images)

    text_summarizer = build_text_summarizer()
    vision_summarizer = build_vision_summarizer()

    text_summaries = summarize_with_cache(
        texts,
        f"{cfg.CACHE_DIR}/text_summaries.json",
        text_summarizer,
        to_str=lambda x: x.text,
        sleep_s=TEXT_SLEEP_SECONDS,
        use_cache=not need_rebuild,
    )
    tables_html = [t.metadata.text_as_html for t in tables]
    #print(tables_html)
    table_summaries = summarize_with_cache(
        tables_html,
        f"{cfg.CACHE_DIR}/table_summaries.json",
        text_summarizer,
        to_str=lambda x: x,
        sleep_s=TEXT_SLEEP_SECONDS,
        use_cache=not need_rebuild,
    )
    image_summaries = summarize_images_with_cache(
        images,
        f"{cfg.CACHE_DIR}/image_summaries.json",
        vision_summarizer,
        sleep_s=VISION_SLEEP_SECONDS,
        use_cache=not need_rebuild,
    )

    vectorstore, docstore = build_vectorstore()
    if need_rebuild:
        print("📗 Rebuilding Chroma store...")
        add_group_to_store(vectorstore, docstore, texts, text_summaries)
        add_group_to_store(vectorstore, docstore, tables, table_summaries)
        add_group_to_store(vectorstore, docstore, images, image_summaries)
        persist_docstore_index(docstore)
        open(cfg.HASH_FILE, "w").write(pdf_hash)
    else:
        print("📗 Using existing Chroma (no rebuild)")

    retrieve_fn = lambda q: retrieve_parents_vs(vectorstore, docstore, q, k=6)
    rag_chain, rag_chain_with_ctx = build_rag_chains(retrieve_fn)
    return rag_chain, rag_chain_with_ctx


# ================= DEMO =================
if __name__ == "__main__":
    rag_chain, rag_chain_with_ctx = _build_pipeline()

    #q1 = "What are the tools that PaperQA have?"
    #q1 = "What is pythagorean theorem?"
    #q1 = "What is Ptolemy's generalization of Pythagorean theorem?"
    q1 = "What is the main idea of the paper?"
    print("\nQ:", q1)
    print("A:", rag_chain.invoke(q1))

    #2 = "How does PaperQA compare to expert humans?"
    #q2 = "What is Ptolemy's generalization of Pythagorean theorem?"
    q2 = "what is The most frequent words?"
    print("\nQ:", q2)
    out = rag_chain_with_ctx.invoke(q2)
    print("A:", out["response"])

    print("\n--- Context preview ---")
    for t in out["context"]["texts"][:2]:
        try:
            print((getattr(t, "text", "") or "")[:400], "\n-----")
        except Exception:
            pass
