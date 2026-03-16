from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from model_factory import get_llm

from vectorstore_setup import (
    CONTENT_COLLECTION,
    VectorStoreBackend,
)


@dataclass
class PromptConfig:
    paper_id: Optional[str] = None
    paper_title: Optional[str] = None
    system_instructions: Optional[str] = None
    # Biến cờ để xác định chế độ giải thích vùng ảnh
    is_visual_explanation: bool = False


# --- Default Prompts ---
DEFAULT_RAG_INSTRUCTIONS = (
    "You are the best scientific research assistant. "
    "Use ONLY the provided context to answer. Synthesize information from multiple sources when needed. "
    "If asked for a summary, combine information to create a comprehensive answer. "
    "Keep answers concise and cite sources as [S1], [S2], etc. "
    "Ground every claim in the context.\n\n"
    "CRITICAL LaTeX Formatting Rules (MUST follow exactly):\n"
    "- For inline math, use single dollar signs: $E = mc^2$\n"
    "- For block/display math, use double dollar signs on their own lines:\n"
    "$$\n"
    "\\frac{a}{b} = c\n"
    "$$\n"
    "- Always use backslash for LaTeX commands: \\frac, \\sum, \\int, \\sqrt, \\alpha, \\beta\n"
    "- ALWAYS close every math delimiter: if you open $, you must close with $\n"
    "- Use \\text{} for words inside math: $P(\\text{event}) = 0.5$\n"
    "- Fractions: \\frac{numerator}{denominator}\n"
    "- Subscripts: x_i or x_{ij}, Superscripts: x^2 or x^{n+1}\n"
    "- DO NOT leave unbalanced $ signs in your response"
)

REGION_EXPLAIN_INSTRUCTIONS = (
    "You are an expert scientific assistant analyzing a specific region from a research paper. "
    "You are provided with a CROPPED IMAGE of the region and textual context from the paper. "
    "1. Identify if the region is a math formula, table, figure/plot, or text. "
    "2. Explain it clearly for a student audience. "
    "   - Formulas: Define symbols, explain terms/intuition. "
    "   - Tables: Describe columns, units, trends. "
    "   - Plots: Describe axes, variables, insights. "
    "3. Use the provided Textual Context to reduce hallucination. Cite it if helpful.\n\n"
    "CRITICAL LaTeX Formatting Rules (MUST follow exactly):\n"
    "- For inline math, use single dollar signs: $E = mc^2$\n"
    "- For block/display math, use double dollar signs on their own lines:\n"
    "$$\n"
    "\\int_{a}^{b} f(x) dx\n"
    "$$\n"
    "- Use backslash for all LaTeX commands: \\frac, \\sum, \\int, \\sqrt, \\alpha, \\beta, \\theta\n"
    "- ALWAYS balance math delimiters: every $ must have a closing $\n"
    "- Define each variable after equations: 'where $x$ represents...'\n"
    "- Fractions: \\frac{a}{b}, Roots: \\sqrt{x}, \\sqrt[n]{x}\n"
    "- Sums: \\sum_{i=1}^{n}, Products: \\prod_{i=1}^{n}\n"
    "- Integrals: \\int_{a}^{b}, Limits: \\lim_{x \\to \\infty}\n"
    "- DO NOT leave unbalanced $ signs"
)


def split_docs(docs: List[Document]):
    """Separate retrieved docs into modality buckets + inject source ids."""

    def _extract_page(meta: Dict[str, Any]) -> Optional[int]:
        def _to_int(val: Any) -> Optional[int]:
            if isinstance(val, int): return val
            if isinstance(val, str) and val.strip().isdigit(): return int(val.strip())
            return None
        return _to_int(meta.get("page_label")) or _to_int(meta.get("page_start")) or _to_int(meta.get("page_end"))

    grouped = {"images": [], "texts": [], "tables": []}
    for idx, doc in enumerate(docs, start=1):
        source_id = f"S{idx}"
        metadata = doc.metadata or {}
        modality = metadata.get("modality", "text")

        payload: Dict[str, Any] = {
            "source_id": source_id,
            "metadata": metadata,
            "text": doc.page_content,
            "type": metadata.get("section_title") or modality.title(),
            "page": _extract_page(metadata),
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
    """
    Unified prompt builder for both General Query and Region Explanation.
    Accepts:
    - context: Dict (grouped docs)
    - question: str
    - prompt_cfg: PromptConfig
    - focus_image_b64: Optional[str] (The specific region to explain)
    """
    ctx = kwargs.get("context", {})
    question = kwargs.get("question", "")
    focus_image_b64 = kwargs.get("focus_image_b64")  # New: Specific image to focus on
    prompt_cfg: PromptConfig = kwargs.get("prompt_cfg", PromptConfig())

    # 1. Determine System Instructions
    base_instruction = prompt_cfg.system_instructions or DEFAULT_RAG_INSTRUCTIONS

    instructions = [base_instruction]

    if prompt_cfg.paper_id and not prompt_cfg.is_visual_explanation:
        instructions.append(f"Focus on paper ID: {prompt_cfg.paper_id}.")

    # 2. Build Text Context
    ctx_lines: List[str] = []

    # Merge texts and tables for context
    combined_text_items = ctx.get("texts", []) + ctx.get("tables", [])

    # Sort: Abstracts first, then others
    def sort_key(item):
        sec = (item["metadata"].get("section_title") or "").lower()
        return 0 if "abstract" in sec else 1

    for item in sorted(combined_text_items, key=sort_key):
        sid = item['source_id']
        section = item["metadata"].get("section_title") or "Context"
        page = item.get("page") or "?"
        ctx_lines.append(f"[{sid}] {section} (p.{page}): {item['text']}")

    context_str = "\n".join(ctx_lines) if ctx_lines else "No textual context available."

    # 3. Construct Message Content
    content = []

    # Text Part
    text_content = (
        f"{'\n'.join(instructions)}\n\n"
        f"--- CONTEXT START ---\n{context_str}\n--- CONTEXT END ---\n\n"
        f"User Question: {question}\n"
    )

    if focus_image_b64:
        text_content += "Note: The user has provided a specific image region to analyze below.\n"

    content.append({"type": "text", "text": text_content})

    # 4. Add Images (Context Images + Focus Image)

    # If we have a focus image (Region Explain), it usually comes LAST or FIRST.
    # Let's put it last to ensure the model focuses on it.
    if focus_image_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{focus_image_b64}"}
        })
        content.append({"type": "text", "text": "Image: The cropped region to explain."})

    # Add other context images (retrieved figures)
    for img_item in ctx.get("images", []):
        # Skip if this is the same as focus image (naive check)
        if focus_image_b64 and img_item.get("image_b64") == focus_image_b64:
            continue

        b64 = img_item.get("image_b64")
        if b64:
            caption = img_item.get("text") or "Figure"
            sid = img_item.get("source_id")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
            content.append({"type": "text", "text": f"[{sid}] Context Figure: {caption}"})

    content.append({"type": "text", "text": "Answer:"})

    return [HumanMessage(content=content)]


def build_generative_chain():
    """
    Builds the pure generation chain: Input -> Prompt -> LLM -> String.
    Input Expected: {
        "context": Dict (from split_docs),
        "question": str,
        "prompt_cfg": PromptConfig,
        "focus_image_b64": Optional[str]
    }
    """
    final_llm = get_llm("generation")

    chain = (
        RunnableLambda(build_mm_prompt)
        | final_llm
        | StrOutputParser()
    )
    return chain


# --- HyDE Query Transformation (F2) ---

def hyde_transform(question: str) -> str:
    """
    HyDE: Hypothetical Document Embeddings.
    Generate a hypothetical answer paragraph to improve retrieval embedding.

    Short/ambiguous queries embed poorly. This generates a longer, more specific
    text that embeds closer to relevant chunks.

    Returns the hypothetical answer, or the original question on error.
    """
    if not question or not question.strip():
        return question

    try:
        llm = get_llm("hyde")
        prompt = (
            "You are a scientific expert. Given the following question, write a short "
            "hypothetical paragraph (100-200 words) that would be a good answer. "
            "Write as if you found this text in a research paper.\n\n"
            f"Question: {question}\n\n"
            "Hypothetical answer paragraph:"
        )
        response = llm.invoke(prompt)
        result = response.content if hasattr(response, 'content') else str(response)
        return result.strip() if result and result.strip() else question
    except Exception as e:
        print(f"[HyDE] Error: {e}, falling back to original query")
        return question


# --- Multi-Paper Query Decomposition ---

def decompose_multi_query(
    question: str,
    paper_titles: Dict[str, str],
) -> Dict[str, Any]:
    """
    Analyze a multi-paper question and decompose it into sub-queries.

    For comparative/meta questions (e.g. "Are these papers related?"),
    the original question embeds poorly against specific document chunks.
    This function decomposes it into per-paper sub-queries that retrieve
    relevant chunks from each paper independently.

    Args:
        question: The user's original question
        paper_titles: Dict mapping paper_id -> title

    Returns:
        {
            "type": "COMPARATIVE" | "SUMMARY" | "DIRECT",
            "sub_queries": list of specific retrieval queries,
            "needs_summaries": bool,
            "original": original question
        }
    """
    if not question or not question.strip() or len(paper_titles) < 2:
        return {
            "type": "DIRECT",
            "sub_queries": [question],
            "needs_summaries": False,
            "original": question,
        }

    paper_list = "\n".join(
        [f"- [{pid[:8]}] {title}" for pid, title in paper_titles.items()]
    )

    try:
        llm = get_llm("condense")
        prompt = (
            "You are a research assistant analyzing a question about multiple papers.\n\n"
            f"Papers:\n{paper_list}\n\n"
            f"User question: \"{question}\"\n\n"
            "Classify this question into ONE of these types:\n"
            "1. COMPARATIVE — comparing, contrasting, or asking about relationships between papers\n"
            "2. SUMMARY — requesting an overview or summary of multiple papers\n"
            "3. DIRECT — a specific technical question that can be answered by searching chunks\n\n"
            "Then generate sub-queries optimized for semantic search.\n"
            "For COMPARATIVE: generate one specific query per paper that extracts the aspect being compared.\n"
            "For SUMMARY: generate one query per paper asking for main contributions.\n"
            "For DIRECT: return the original question as-is.\n\n"
            "Respond in EXACTLY this format (no extra text):\n"
            "TYPE: <COMPARATIVE|SUMMARY|DIRECT>\n"
            "QUERIES:\n"
            "- <query 1>\n"
            "- <query 2>\n"
            "..."
        )

        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        text = text.strip()

        # Parse response
        query_type = "DIRECT"
        sub_queries = [question]

        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if line.upper().startswith("TYPE:"):
                parsed = line.split(":", 1)[1].strip().upper()
                if parsed in ("COMPARATIVE", "SUMMARY", "DIRECT"):
                    query_type = parsed

        # Extract sub-queries
        in_queries = False
        parsed_queries: List[str] = []
        for line in lines:
            line = line.strip()
            if line.upper().startswith("QUERIES:"):
                in_queries = True
                continue
            if in_queries and line.startswith("- "):
                q = line[2:].strip()
                if q:
                    parsed_queries.append(q)

        if parsed_queries:
            sub_queries = parsed_queries

        needs_summaries = query_type in ("COMPARATIVE", "SUMMARY")

        print(
            f"[DecomposeMultiQuery] type={query_type}, "
            f"sub_queries={len(sub_queries)}, needs_summaries={needs_summaries}"
        )

        return {
            "type": query_type,
            "sub_queries": sub_queries,
            "needs_summaries": needs_summaries,
            "original": question,
        }

    except Exception as e:
        print(f"[DecomposeMultiQuery] Error: {e}, falling back to DIRECT")
        return {
            "type": "DIRECT",
            "sub_queries": [question],
            "needs_summaries": False,
            "original": question,
        }


# --- Conversation Memory / Question Condensing (F3) ---

def condense_question(
    chat_history: Optional[List[Dict[str, str]]],
    question: str,
    summary: str = "",
    max_history: int = 10,
    custom_condense_prompt: str = "",
) -> str:
    """
    Condense a follow-up question using chat history + rolling summary
    into a standalone question.

    Args:
        chat_history: Recent messages within the sliding window
        question: The follow-up question
        summary: Rolling summary of older messages beyond the window
        max_history: Max messages to include from history
        custom_condense_prompt: Optional custom prompt text (from admin config)

    Returns standalone question, or original question if no history or on error.
    """
    if not chat_history and not summary:
        return question

    # Limit history to most recent messages
    recent = (chat_history or [])[-max_history:]

    try:
        llm = get_llm("condense")
        history_str = "\n".join(
            f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}"
            for msg in recent
        )

        summary_block = ""
        if summary:
            summary_block = (
                f"Summary of earlier conversation:\n{summary}\n\n"
            )

        # Use custom prompt if provided, otherwise default
        condense_instruction = custom_condense_prompt or (
            "Given the following conversation context and a follow-up question, "
            "rephrase the follow-up question to be a standalone question that "
            "can be understood without the conversation history."
        )

        prompt = (
            f"{condense_instruction}\n\n"
            f"{summary_block}"
            f"Recent Chat History:\n{history_str}\n\n"
            f"Follow-up Question: {question}\n\n"
            "Standalone Question:"
        )
        response = llm.invoke(prompt)
        result = response.content if hasattr(response, 'content') else str(response)
        condensed = result.strip() if result and result.strip() else question
        print(f"[Condense] '{question}' → '{condensed}'")
        return condensed
    except Exception as e:
        print(f"[Condense] Error: {e}, falling back to original question")
        return question


def summarize_conversation(
    old_summary: str,
    overflow_messages: List[Dict[str, str]],
) -> str:
    """
    Summarize overflow messages (pushed out of the sliding window)
    into a rolling summary that preserves key context.

    Extracts: key topics, important numbers/data, conclusions,
    and user preferences mentioned in the conversation.

    Args:
        old_summary: Previous rolling summary (may be empty)
        overflow_messages: Messages that just fell out of the window

    Returns: Updated summary string
    """
    if not overflow_messages:
        return old_summary

    try:
        llm = get_llm("summarization")

        msgs_str = "\n".join(
            f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}"
            for msg in overflow_messages
        )

        old_block = ""
        if old_summary:
            old_block = f"Previous summary:\n{old_summary}\n\n"

        prompt = (
            "You are maintaining a rolling memory of a research conversation. "
            "Combine the previous summary with the new messages below into a "
            "concise updated summary.\n\n"
            "IMPORTANT — preserve these elements:\n"
            "- Key topics and concepts discussed\n"
            "- Important numbers, statistics, and data points\n"
            "- Conclusions and insights reached\n"
            "- User's research interests and preferences\n"
            "- Any specific papers, methods, or terms mentioned\n\n"
            f"{old_block}"
            f"New messages to incorporate:\n{msgs_str}\n\n"
            "Updated summary (keep under 300 words, be concise):"
        )

        response = llm.invoke(prompt)
        result = response.content if hasattr(response, 'content') else str(response)
        new_summary = result.strip() if result and result.strip() else old_summary
        print(f"[Memory] Summary updated: {len(new_summary)} chars")
        return new_summary
    except Exception as e:
        print(f"[Memory] Summarize failed: {e}, keeping old summary")
        return old_summary


# --- RRF Merge for Multi-Source Retrieval (F1) ---

def rrf_merge(
    user_docs: List[Document],
    system_docs: List[Document],
    k: int = 60,
    user_boost: float = 1.2,
    total_k: int = 15,
) -> List[Document]:
    """
    Reciprocal Rank Fusion — merge docs from user paper + system KB.

    Args:
        user_docs: Documents from user's paper
        system_docs: Documents from system knowledge base
        k: RRF constant (higher = more equal weight)
        user_boost: Boost factor for user docs (>1 = prioritize user paper)
        total_k: Maximum total docs to return

    Returns:
        Merged, deduplicated, and ranked list of documents
    """
    if not user_docs and not system_docs:
        return []
    if not system_docs:
        return user_docs[:total_k]
    if not user_docs:
        return system_docs[:total_k]

    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    # Score user docs with boost
    for rank, doc in enumerate(user_docs):
        content_key = doc.page_content[:100].strip()
        rrf_score = user_boost / (k + rank + 1)
        scores[content_key] = scores.get(content_key, 0) + rrf_score
        doc_map[content_key] = doc

    # Score system docs
    for rank, doc in enumerate(system_docs):
        content_key = doc.page_content[:100].strip()
        rrf_score = 1.0 / (k + rank + 1)
        scores[content_key] = scores.get(content_key, 0) + rrf_score
        if content_key not in doc_map:
            doc_map[content_key] = doc

    # Sort by combined RRF score
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    return [doc_map[key] for key in sorted_keys[:total_k]]


# Similarity threshold: documents below this are considered irrelevant
RELEVANCE_THRESHOLD = 0.45  # For converted similarity (1/(1+distance))


def document_retrieve(
    query: str,
    paper_id: str,
    backend: VectorStoreBackend,
    content_collection: str = CONTENT_COLLECTION,
    k: int = 12,
    use_hyde: bool = False,
    include_system_kb: bool = False,
    kb_categories: List[str] | None = None,
) -> List[Document]:
    """
    Chiến lược retrieval tối ưu với similarity threshold:
    1. Nếu là summary: Lấy 1-2 bản ghi Abstract + (k-2) bản ghi Content liên quan nhất.
    2. Nếu là query thường: Lấy k bản ghi tốt nhất từ toàn bộ Store (bao gồm cả Abstract).
    3. Lọc các kết quả có similarity score thấp hơn RELEVANCE_THRESHOLD.
    4. Nếu use_hyde=True: Transform query bằng HyDE trước khi embed.
    5. Nếu include_system_kb=True: Song song search system KB, merge bằng RRF.
    """
    # HyDE transform if enabled
    effective_query = hyde_transform(query) if use_hyde else query

    query_lower = query.lower()
    is_summary = any(x in query_lower for x in ["summary", "overview", "main idea", "summarize", "abstract", "conclusion", "contribution"])

    results_with_scores: List[tuple] = []

    if is_summary:
        # PATH A: Lấy Abstract để nắm ý chính
        abstract_docs = backend.similarity_search_with_score(
            query=effective_query,
            k=1,
            collection=content_collection,
            where={"$and": [{"paper_id": paper_id}, {"modality": "abstract"}]}
        )
        results_with_scores.extend(abstract_docs)

        # PATH B: Lấy thêm Content để có chi tiết
        search_k = max(8, k - len(results_with_scores))
        content_docs = backend.similarity_search_with_score(
            query=effective_query,
            k=search_k,
            collection=content_collection,
            where={"paper_id": paper_id}
        )
        results_with_scores.extend(content_docs)
    else:
        # Truy vấn bình thường: Tìm kiếm dựa trên độ tương đồng với scores
        results_with_scores = backend.similarity_search_with_score(
            query=effective_query,
            k=k,
            collection=content_collection,
            where={"paper_id": paper_id}
        )

    # Filter by relevance threshold
    filtered = [
        (doc, score) for doc, score in results_with_scores
        if score >= RELEVANCE_THRESHOLD
    ]

    # Tag each document with its relevance score
    user_docs = []
    seen_content = set()
    for doc, score in filtered:
        if doc.metadata is None:
            doc.metadata = {}
        doc.metadata["relevance_score"] = score
        content_key = doc.page_content[:150].strip()
        if content_key not in seen_content:
            user_docs.append(doc)
            seen_content.add(content_key)

    # System KB retrieval (if enabled)
    if include_system_kb:
        try:
            kb_where: dict | None = None
            if kb_categories:
                if len(kb_categories) == 1:
                    kb_where = {"category": kb_categories[0]}
                else:
                    kb_where = {"category": {"$in": kb_categories}}

            kb_results = backend.similarity_search_with_score(
                query=effective_query,
                k=k,
                collection="system_knowledge_base",
                where=kb_where,
            )

            system_docs = []
            for doc, score in kb_results:
                if score >= RELEVANCE_THRESHOLD:
                    if doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata["relevance_score"] = score
                    doc.metadata["source"] = "system"
                    system_docs.append(doc)

            # Merge using RRF
            return rrf_merge(user_docs, system_docs, total_k=k)
        except Exception as e:
            print(f"[document_retrieve] System KB search failed: {e}")
            # Fallback to user docs only

    return user_docs[:k]


def multi_document_retrieve(
    query: str,
    paper_ids: List[str],
    backend: VectorStoreBackend,
    content_collection: str = CONTENT_COLLECTION,
    k_per_paper: int = 6,
    total_k: int = 15,
    relevance_threshold: float = 0.5,
) -> List[Document]:
    """
    Retrieve documents from multiple papers with strict per-paper scoring.

    Strategy:
    1. Retrieve from each paper independently with scores
    2. Apply absolute threshold to filter irrelevant chunks
    3. Compute per-paper aggregated relevance to identify primary paper(s)
    4. Re-rank: strongly penalize docs from weakly-matching papers
    5. Return only grounded, relevant results

    Args:
        query: The user's question
        paper_ids: List of paper IDs to search across
        backend: Vector store backend
        k_per_paper: How many docs to retrieve per paper initially
        total_k: Final number of docs to return after merging
        relevance_threshold: Ratio threshold for including a paper's docs
    """
    all_docs_with_scores: List[tuple[Document, float]] = []

    # Step 1: Search each paper and track relevance scores
    paper_scores: Dict[str, List[float]] = {}
    paper_max_scores: Dict[str, float] = {}

    for paper_id in paper_ids:
        try:
            docs_with_scores = backend.similarity_search_with_score(
                query=query,
                k=k_per_paper,
                collection=content_collection,
                where={"paper_id": paper_id}
            )

            if docs_with_scores:
                scores = [score for _, score in docs_with_scores]
                paper_scores[paper_id] = scores
                paper_max_scores[paper_id] = max(scores)

                for doc, score in docs_with_scores:
                    if doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata["source_paper_id"] = paper_id
                    doc.metadata["relevance_score"] = score
                    all_docs_with_scores.append((doc, score))

        except Exception as e:
            print(f"Error retrieving from paper {paper_id}: {e}")
            continue

    if not all_docs_with_scores:
        return []

    # Step 2: Apply ABSOLUTE threshold first - remove clearly irrelevant chunks
    all_docs_with_scores = [
        (doc, score) for doc, score in all_docs_with_scores
        if score >= RELEVANCE_THRESHOLD
    ]

    if not all_docs_with_scores:
        return []

    # Step 3: Per-paper scoring — identify primary vs secondary papers
    # Compute mean of top-3 scores per paper as aggregate relevance
    paper_agg_scores: Dict[str, float] = {}
    for paper_id, scores in paper_scores.items():
        top_scores = sorted(scores, reverse=True)[:3]
        paper_agg_scores[paper_id] = sum(top_scores) / len(top_scores) if top_scores else 0

    best_paper_score = max(paper_agg_scores.values()) if paper_agg_scores else 0

    # Step 4: Dynamic per-paper threshold — papers with aggregate score
    # below (best_paper * relevance_threshold) are heavily penalized
    dynamic_paper_threshold = best_paper_score * relevance_threshold

    filtered_docs: List[tuple[Document, float]] = []
    for doc, score in all_docs_with_scores:
        paper_id = doc.metadata.get("source_paper_id")
        paper_agg = paper_agg_scores.get(paper_id, 0)

        if paper_agg >= dynamic_paper_threshold:
            # Paper is relevant — include at original score
            filtered_docs.append((doc, score))
        else:
            # Paper is weakly relevant — only include if this specific chunk
            # has very high individual score (above 90% of best paper's max)
            if score >= best_paper_score * 0.9:
                filtered_docs.append((doc, score))
            # Otherwise skip entirely — prevents citations from irrelevant papers

    # Sort by score (higher is better)
    filtered_docs.sort(key=lambda x: x[1], reverse=True)

    # Step 5: Deduplicate by content
    seen_content = set()
    unique_docs: List[Document] = []

    for doc, score in filtered_docs:
        content_key = doc.page_content[:150].strip()
        if content_key not in seen_content:
            unique_docs.append(doc)
            seen_content.add(content_key)

    return unique_docs[:total_k]


# --- Grounding check ---

UNGROUNDED_INSTRUCTIONS = (
    "You are a helpful AI assistant. The user asked a question that is NOT covered by "
    "the provided research paper(s). No relevant context was found in the document(s).\n\n"
    "Answer the question using your general knowledge.\n"
    "Do NOT invent citations. Do NOT use [S1], [S2] or any source markers.\n"
    "Make it clear that this answer is from general knowledge, not from the paper.\n"
    "Start your response with: \"**Note:** This answer is based on general knowledge, "
    "not on the content of the provided document(s).\"\n\n"
    "CRITICAL LaTeX Formatting Rules (MUST follow exactly):\n"
    "- For inline math, use single dollar signs: $E = mc^2$\n"
    "- For block/display math, use double dollar signs on their own lines:\n"
    "$$\n"
    "\\frac{a}{b} = c\n"
    "$$\n"
    "- Always use backslash for LaTeX commands: \\frac, \\sum, \\int, \\sqrt, \\alpha, \\beta\n"
    "- ALWAYS close every math delimiter: if you open $, you must close with $\n"
    "- DO NOT leave unbalanced $ signs in your response"
)


def check_grounding(docs: List[Document], threshold: float = RELEVANCE_THRESHOLD) -> bool:
    """
    Check if any retrieved document is sufficiently relevant to ground an answer.
    Returns True if the query IS grounded (i.e. we found relevant context).
    Returns False if the query is NOT grounded (no relevant context).
    """
    if not docs:
        return False

    for doc in docs:
        score = (doc.metadata or {}).get("relevance_score", 0)
        if score >= threshold:
            return True

    return False


def build_document_rag_chain(
    paper_id: str,
    backend: VectorStoreBackend,
    prompt_cfg: Optional[PromptConfig] = None,
):
    """
    Builds a standard RAG chain for the /query endpoint.
    Includes grounding check — if no relevant docs found, still answers
    from general knowledge but returns 0 citations.
    """
    prompt_cfg = prompt_cfg or PromptConfig(paper_id=paper_id)

    # 1. Retrieval Step
    def _retrieve_step(input_dict):
        q = input_dict["question"]
        docs = document_retrieve(q, paper_id, backend)
        return docs

    # 2. Generation Step
    gen_chain = build_generative_chain()

    # 3. Compose with grounding check
    def _run_chain(input_dict):
        question = input_dict["question"]
        docs = _retrieve_step(input_dict)

        if check_grounding(docs):
            # Grounded: answer from paper context with citations
            context = split_docs(docs)
            response = gen_chain.invoke({
                "context": context,
                "question": question,
                "prompt_cfg": prompt_cfg,
                "focus_image_b64": None,
            })
            return {
                "response": response,
                "context": context,
                "grounded": True,
            }
        else:
            # Ungrounded: answer from general knowledge, 0 citations
            ungrounded_cfg = PromptConfig(
                paper_id=paper_id,
                system_instructions=UNGROUNDED_INSTRUCTIONS,
            )
            empty_context = {"texts": [], "tables": [], "images": []}
            response = gen_chain.invoke({
                "context": empty_context,
                "question": question,
                "prompt_cfg": ungrounded_cfg,
                "focus_image_b64": None,
            })
            return {
                "response": response,
                "context": empty_context,
                "grounded": False,
            }

    return RunnableLambda(_run_chain)


def brainstorm_questions_chain(
    title: str,
    abstract: str,
    text_input: Optional[str] = None,
    relevant_context: Optional[str] = None,
) -> List[str]:
    """
    Sử dụng LLM để tạo ra các câu hỏi gợi ý dựa trên Title, Abstract,
    và tùy chọn text_input từ user + relevant context từ vector store.

    Nếu text_input được cung cấp, câu hỏi sẽ được sinh ra phù hợp với
    ý định của user và nội dung bài báo.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Build dynamic prompt based on whether text_input is provided
    if text_input and text_input.strip():
        prompt = (
            f"You are a research expert analyzing a scientific paper.\n\n"
            f"Paper Title: {title}\n"
            f"Paper Abstract: {abstract}\n\n"
        )

        if relevant_context:
            prompt += (
                f"Relevant content from the paper related to the user's interest:\n"
                f"---\n{relevant_context}\n---\n\n"
            )

        prompt += (
            f"The user is interested in the following topic/direction:\n"
            f"\"{text_input}\"\n\n"
            f"Based on the paper's content AND the user's interest, generate 3-5 "
            f"highly targeted research questions that:\n"
            f"1. Are directly relevant to both the paper's content and the user's interest\n"
            f"2. Help the user explore the specific aspect they care about\n"
            f"3. Connect the user's interest with the paper's methodology, results, or findings\n"
            f"4. Range from specific factual questions to deeper analytical ones\n"
            f"5. Are answerable from the paper's content\n\n"
            f"Output requirements:\n"
            f"- Return ONLY a JSON list of strings.\n"
            f"- Format: [\"question 1\", \"question 2\", ...]\n"
            f"- Questions should be concise, professional, and in the same language as the user's input when possible."
        )
    else:
        prompt = (
            f"You are a research expert. Based on the following research paper metadata, "
            f"suggest 3-5 thought-provoking and diverse questions that a reader should ask "
            f"to understand the paper's contributions, methodology, and results more deeply.\n\n"
            f"Title: {title}\n"
            f"Abstract: {abstract}\n\n"
            f"Output requirements:\n"
            f"- Return ONLY a JSON list of strings.\n"
            f"- Format: [\"question 1\", \"question 2\", ...]\n"
            f"- Questions should be concise and professional."
        )

    response = llm.invoke(prompt)

    try:
        import json
        content = response.content.strip()
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        print(f"Error parsing brainstormed questions: {e}")
        return [line.strip("- ") for line in response.content.split("\n") if len(line) > 10][:6]


def summarize_paper_chain(
    title: str,
    abstract: str,
    context: str,
) -> str:
    """
    Generate a comprehensive, structured summary of a research paper
    using its title, abstract, and retrieved content sections.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    prompt = (
        f"You are an expert scientific paper summarizer. Generate a comprehensive, "
        f"well-structured summary of the following research paper.\n\n"
        f"Paper Title: {title}\n"
        f"Paper Abstract: {abstract}\n\n"
        f"Key Content from the Paper:\n"
        f"---\n{context}\n---\n\n"
        f"Write a detailed summary covering these aspects (use markdown formatting):\n"
        f"## Overview\n"
        f"A 2-3 sentence high-level overview of what this paper is about.\n\n"
        f"## Key Contributions\n"
        f"Bullet points of the main contributions and novel aspects.\n\n"
        f"## Methodology\n"
        f"Brief description of the approach/methods used.\n\n"
        f"## Main Results\n"
        f"Key findings, performance metrics, or experimental results.\n\n"
        f"## Significance & Implications\n"
        f"Why this work matters and potential impact.\n\n"
        f"Requirements:\n"
        f"- Be accurate and grounded in the provided content\n"
        f"- Use clear, academic language\n"
        f"- Keep the total summary between 300-600 words\n"
        f"- Use LaTeX for any mathematical expressions: $formula$\n"
        f"- Do not invent information not present in the content"
    )

    response = llm.invoke(prompt)
    return response.content.strip()