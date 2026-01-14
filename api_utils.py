# -*- coding: utf-8 -*-
"""
API utility functions for reusable logic across endpoints.
"""

import copy
from typing import Dict, Any, Optional

from vectorstore_setup import VectorStoreBackend, CONTENT_COLLECTION
from rag_pipeline import document_retrieve, split_docs


def retrieve_context_for_explain(
    file_id: str,
    backend: VectorStoreBackend,
    page_number: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Retrieves context for visual explanation with a strict focus on the target page.
    Strategy:
    1. Page-Specific: Fetch ALL chunks (text, tables, images) belonging to the target page using metadata filtering.
    2. Global Semantic: Fetch top relevant cross-references (definitions, related figures) from the whole paper.
    3. Merge & Deduplicate.
    """
    
    combined_docs = []
    seen_content = set()

    # --- 1. STRICT PAGE RETRIEVAL (The "All chunks on this page" requirement) ---
    if page_number is not None:
        # FIX: ChromaDB yêu cầu dùng $and rõ ràng khi filter nhiều điều kiện
        where_filter = {
            "$and": [
                {"paper_id": file_id},
                {"page_label": page_number}
            ]
        }
          
        # We use a neutral query "." combined with a strict metadata filter.
        # We set k=30 to ensure we catch every single chunk on that page.
        page_docs = backend.similarity_search(
            query=".",  # Dummy query to satisfy interface, relies on filter
            k=30,       # High limit to capture dense pages
            collection=CONTENT_COLLECTION,
            where=where_filter
        )
        for doc in page_docs:
            combined_docs.append(doc)
            # Use content hash or snippet as key for deduplication
            seen_content.add(doc.page_content[:100])

    # --- 2. GLOBAL SEMANTIC RETRIEVAL (Context from other pages) ---
    # We search for generic scientific terms to find definitions/formulas that might be 
    # referenced in the crop but defined elsewhere.
    global_docs = document_retrieve(
        query="formula equation figure table definition theory",
        paper_id=file_id,
        backend=backend,
        k=5 # Keep it focused
    )

    # --- 3. MERGE & DEDUPLICATE ---
    for doc in global_docs:
        key = doc.page_content[:100]
        # Only add if we haven't seen this content from the page retrieval step
        if key not in seen_content:
            combined_docs.append(doc)
            seen_content.add(key)

    # Convert to standard format (split into texts/images/tables)
    structured_ctx = split_docs(combined_docs)

    return structured_ctx


def format_explain_context_for_chain(
    context: Dict[str, Any],
    image_b64: str,
    page_number: Optional[int] = None,
) -> Dict[str, Any]:
    # Deep copy to avoid mutating original for chain
    final_ctx = copy.deepcopy(context)
    
    return final_ctx