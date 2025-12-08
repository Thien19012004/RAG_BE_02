import base64
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from vectorstore_setup import (
    ABSTRACT_COLLECTION,
    CONTENT_COLLECTION,
    VectorStoreBackend,
)


@dataclass
class PromptConfig:
    mode: str  # "document" | "corpus"
    paper_id: Optional[str] = None
    paper_title: Optional[str] = None
    instructions: Optional[str] = None


def split_docs(docs: List[Document]):
    """Separate retrieved docs into modality buckets + inject source ids."""

    def _extract_page(meta: Dict[str, Any]) -> Optional[int]:
        def _to_int(val: Any) -> Optional[int]:
            if isinstance(val, int):
                return val
            if isinstance(val, str):
                stripped = val.strip()
                if stripped.isdigit():
                    return int(stripped)
            return None

        return _to_int(meta.get("page_label")) or _to_int(meta.get("page_start")) or _to_int(
            meta.get("page_end")
        )

    grouped = {"images": [], "texts": [], "tables": []}
    for idx, doc in enumerate(docs, start=1):
        source_id = f"S{idx}"
        metadata = doc.metadata or {}
        modality = metadata.get("modality", "text")
        page_number = _extract_page(metadata)

        payload: Dict[str, Any] = {
            "source_id": source_id,
            "metadata": metadata,
            "text": doc.page_content,
            "type": metadata.get("section_title") or modality.title(),
            "page": page_number,
            "locator": {
                "paper_id": metadata.get("paper_id"),
                "section_title": metadata.get("section_title"),
                "page_label": metadata.get("page_label"),
                "page_start": metadata.get("page_start"),
                "page_end": metadata.get("page_end"),
                # mới thêm:
                "bbox": metadata.get("bbox"),
            },
        }

        if modality == "image":
            payload["image_b64"] = metadata.get("image_b64")
            grouped["images"].append(payload)
        elif modality == "table":
            payload["table_html"] = metadata.get("table_html", "")
            grouped["tables"].append(payload)
        else:
            grouped["texts"].append(payload)

    return grouped



def build_mm_prompt(kwargs: Dict[str, Any]):
    ctx = kwargs["context"]
    question = kwargs["question"]
    prompt_cfg: PromptConfig = kwargs["prompt_cfg"]

    instructions = [
        "You are the best scientific research assistant. I will tip you 1000 dollars if you answer perfectly.",
        "Use ONLY the provided context to answer. Synthesize information from multiple sources when needed.",
        "If asked for a summary or overview, combine information from different sections to create a comprehensive answer.",
        "Keep answers concise (3-5 sentences for simple questions, up to 2 paragraphs for summaries) and cite sources as [S1], [S2], etc.",
        "Ground every claim in the context. If context is truly insufficient after synthesizing all available sources, acknowledge limitations but still provide the best answer possible.",
    ]
    if prompt_cfg.mode == "document" and prompt_cfg.paper_id:
        instructions.append(
            f"This question is about a single paper (paper_id={prompt_cfg.paper_id}). "
            "Restrict your answer to this paper unless explicitly told otherwise."
        )
    if prompt_cfg.instructions:
        instructions.append(prompt_cfg.instructions)

    ctx_lines: List[str] = []
    # Prioritize abstract if present (usually from abstract_store)
    abstract_items = []
    other_items = []
    
    for bucket in ("texts", "tables"):
        for item in ctx[bucket]:
            modality = item["metadata"].get("modality", "")
            section = item["metadata"].get("section_title") or item["metadata"].get("source", "")
            # Check if this is an abstract (from abstract_store or section_title indicates abstract)
            is_abstract = (
                modality == "abstract"
                or "abstract" in section.lower()
                or (not section and len(item["text"]) > 200)  # Long text without section might be abstract
            )
            if is_abstract:
                abstract_items.append(item)
            else:
                other_items.append(item)
    
    # Format: abstract first, then other content
    for item in abstract_items:
        section = item["metadata"].get("section_title") or "Abstract"
        pages = item["metadata"].get("page_label") or item["metadata"].get("page", "")
        ctx_lines.append(
            f"[{item['source_id']}] {section} (p.{pages}):\n{item['text']}"
        )
    
    for item in other_items:
        section = item["metadata"].get("section_title") or item["metadata"].get("source", "")
        pages = item["metadata"].get("page_label") or item["metadata"].get("page", "")
        ctx_lines.append(
            f"[{item['source_id']}] {section or 'Context'} (p.{pages}): {item['text']}"
        )

    content = [
        {
            "type": "text",
            "text": (
                "\n".join(instructions)
                + "\n\nContext:\n"
                + ("\n".join(ctx_lines) if ctx_lines else "No textual context available.")
                + f"\n\nQuestion: {question}\nAnswer:"
            ),
        }
    ]

    for image_item in ctx["images"]:
        img_b64 = image_item.get("image_b64")
        if not img_b64:
            continue
        caption = image_item["text"] or "Figure from the paper."
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            }
        )
        content.append(
            {
                "type": "text",
                "text": f"[{image_item['source_id']}] Figure description: {caption}",
            }
        )

    return [HumanMessage(content=content)]


def build_rag_chains(retrieve_fn, prompt_cfg: PromptConfig):
    final_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    def _splitter(docs: List[Document]):
        return split_docs(docs)

    def _prompt_builder(payload: Dict[str, Any]):
        payload["prompt_cfg"] = prompt_cfg
        return build_mm_prompt(payload)

    rag_chain = (
        {
            "context": RunnableLambda(lambda x: retrieve_fn(x)) | RunnableLambda(_splitter),
            "question": RunnablePassthrough(),
            "prompt_cfg": RunnableLambda(lambda _: prompt_cfg),
        }
        | RunnableLambda(_prompt_builder)
        | final_llm
        | StrOutputParser()
    )

    rag_chain_with_ctx = (
        {
            "context": RunnableLambda(lambda x: retrieve_fn(x)) | RunnableLambda(_splitter),
            "question": RunnablePassthrough(),
            "prompt_cfg": RunnableLambda(lambda _: prompt_cfg),
        }
        | RunnablePassthrough().assign(
            response=(RunnableLambda(_prompt_builder) | final_llm | StrOutputParser())
        )
    )

    return rag_chain, rag_chain_with_ctx


def _get_rewrite_chain():
    prompt = ChatPromptTemplate.from_template(
        "Condense the academic question below into a <=10 word semantic search query.\n"
        "Question: {question}\nSearch query:"
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    return prompt | llm | StrOutputParser()


def condense_query(question: str) -> str:
    """Use a small LLM to rewrite the user query for abstract retrieval."""
    if not hasattr(condense_query, "_chain"):
        condense_query._chain = _get_rewrite_chain()
    try:
        condensed = condense_query._chain.invoke({"question": question}).strip()
        return condensed or question
    except Exception:
        return question


def corpus_retrieve(
    query: str,
    backend: VectorStoreBackend,
    abstract_collection: str = ABSTRACT_COLLECTION,
    content_collection: str = CONTENT_COLLECTION,
    k_abstracts: int = 50,
    k_content: int = 12,
) -> List[Document]:
    search_query = condense_query(query)
    abstract_hits = backend.similarity_search(search_query, k=k_abstracts, collection=abstract_collection)
    paper_ids = list({hit.metadata.get("paper_id") for hit in abstract_hits if hit.metadata.get("paper_id")})

    where = {"paper_id": {"$in": paper_ids}} if paper_ids else None
    content_hits = backend.similarity_search(
        query=query,
        k=k_content,
        collection=content_collection,
        where=where,
    )
    return content_hits


def document_retrieve(
    query: str,
    paper_id: str,
    backend: VectorStoreBackend,
    content_collection: str = CONTENT_COLLECTION,
    abstract_collection: str = ABSTRACT_COLLECTION,
    k: int = 12,
) -> List[Document]:
    """
    Retrieve documents for a specific paper. For summary queries, always include abstract first.
    """
    results: List[Document] = []
    
    # Check if query is about summary/overview - if so, prioritize abstract
    query_lower = query.lower()
    is_summary_query = any(term in query_lower for term in ["summary", "summarize", "overview", "what is this paper", "main idea", "contribution"])
    
    if is_summary_query:
        # Always include abstract for summary queries
        abstract_docs = backend.similarity_search(
            query=query,
            k=1,
            collection=abstract_collection,
            where={"paper_id": paper_id},
        )
        results.extend(abstract_docs)
        # Reduce k for content since we already have abstract
        k_content = max(6, k - 1)
    else:
        k_content = k
    
    # Retrieve content chunks
    where = {"paper_id": paper_id}
    content_docs = backend.similarity_search(
        query=query,
        k=k_content,
        collection=content_collection,
        where=where,
    )
    results.extend(content_docs)
    
    # Deduplicate by page_content (in case abstract was also in content)
    seen = set()
    unique_results = []
    for doc in results:
        content_key = doc.page_content[:100]  # Use first 100 chars as key
        if content_key not in seen:
            seen.add(content_key)
            unique_results.append(doc)
    
    return unique_results[:k]


def build_document_rag(
    paper_id: str,
    backend: VectorStoreBackend,
    content_collection: str = CONTENT_COLLECTION,
    abstract_collection: str = ABSTRACT_COLLECTION,
    prompt_cfg: Optional[PromptConfig] = None,
):
    prompt_cfg = prompt_cfg or PromptConfig(mode="document", paper_id=paper_id)

    def _retrieve(question: str):
        return document_retrieve(
            question, 
            paper_id, 
            backend, 
            content_collection=content_collection,
            abstract_collection=abstract_collection,
        )

    return build_rag_chains(_retrieve, prompt_cfg)


def build_corpus_rag(
    backend: VectorStoreBackend,
    abstract_collection: str = ABSTRACT_COLLECTION,
    content_collection: str = CONTENT_COLLECTION,
    prompt_cfg: Optional[PromptConfig] = None,
):
    prompt_cfg = prompt_cfg or PromptConfig(mode="corpus")

    def _retrieve(question: str):
        return corpus_retrieve(
            question,
            backend,
            abstract_collection=abstract_collection,
            content_collection=content_collection,
        )

    return build_rag_chains(_retrieve, prompt_cfg)
