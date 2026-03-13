from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from vectorstore_setup import (
    CONTENT_COLLECTION,
    VectorStoreBackend,
)


@dataclass
class PromptConfig:
    paper_id: Optional[str] = None
    paper_title: Optional[str] = None
    paper_abstract: Optional[str] = None
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

    # Inject global context (Title & Abstract)
    global_context = ""
    if prompt_cfg.paper_title or prompt_cfg.paper_abstract:
        global_context += "--- PAPER OVERVIEW (Use this to understand the big picture) ---\n"
        if prompt_cfg.paper_title:
            global_context += f"Title: {prompt_cfg.paper_title}\n"
        if prompt_cfg.paper_abstract:
            global_context += f"Abstract: {prompt_cfg.paper_abstract}\n"
        global_context += "\n"

    # Text Part
    text_content = (
        f"{'\n'.join(instructions)}\n\n"
        f"{global_context}"
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
    final_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    chain = (
        RunnableLambda(build_mm_prompt)
        | final_llm
        | StrOutputParser()
    )
    return chain


# Similarity threshold: documents below this are considered irrelevant
RELEVANCE_THRESHOLD = 0.45  # For converted similarity (1/(1+distance))


def analyze_user_query(query: str) -> dict:
    import json
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import StrOutputParser

    if len(query) > 500:
        # Avoid running analysis on giant text blocks (e.g. from /brainstorm-questions)
        print(f"\n[QUERY ROUTER] Query too long ({len(query)} chars). Skipping analysis.")
        return {"intent": "SPECIFIC", "search_queries": [query[:200]]}

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = PromptTemplate.from_template(
        "You are a routing assistant for a scientific RAG system.\n"
        "Analyze the user's query and output a JSON dictionary with two keys: 'intent' and 'search_queries'.\n"
        "1. 'intent': 'GLOBAL' if the user asks for high-level summaries, core concepts, or the abstract. 'SPECIFIC' if they ask about details.\n"
        "2. 'search_queries': Generate 1 to 3 optimized queries to find relevant text INSIDE the user's paper. "
        "CRITICAL: If the user asks 'Explain the abstract' or 'Summarize', DO NOT generate general queries like 'how to write an abstract'. "
        "Instead, generate keywords that will match the paper's actual content (e.g., ['abstract', 'introduction', 'conclusion', 'summary']).\n"
        "Translate any casual terms into academic English.\n\n"
        "Output strictly in JSON format.\n\n"
        "User Query: {query}"
    )

    chain = prompt | llm | StrOutputParser()
    try:
        result = chain.invoke({"query": query})
        content = result.strip()
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
        parsed = json.loads(content)
        
        intent = parsed.get("intent", "SPECIFIC")
        search_queries = parsed.get("search_queries", [query])
        
        # LOGGING INTENT AND QUERIES
        print(f"\n[QUERY ROUTER] Parsed user query: '{query[:100]}...'")
        print(f"[QUERY ROUTER] Detected Intent: {intent}")
        print(f"[QUERY ROUTER] Optimized Search Queries: {search_queries}\n")
        
        return {
            "intent": intent,
            "search_queries": search_queries
        }
    except Exception as e:
        print(f"[QUERY ROUTER] Error parsing user query analysis: {e}")
        return {"intent": "SPECIFIC", "search_queries": [query]}


def document_retrieve(
    query: str,
    paper_id: str,
    backend: VectorStoreBackend,
    content_collection: str = CONTENT_COLLECTION,
    k: int = 12,
) -> List[Document]:
    """
    Advanced RAG retrieval with Query Routing and Rewriting.
    """
    analysis = analyze_user_query(query)
    intent = analysis.get("intent", "SPECIFIC")
    search_queries = analysis.get("search_queries", [query])
    
    # ALWAYS ensure the exact original user query is included first to capture semantic nuances
    if query not in search_queries:
        search_queries.insert(0, query)

    results_with_scores: List[tuple] = []

    if intent == "GLOBAL":
        for search_query in search_queries:
            abstract_docs = backend.similarity_search_with_score(
                query=search_query,
                k=2,
                collection=content_collection,
                where={"$and": [{"paper_id": paper_id}, {"modality": "abstract"}]}
            )
            # Boost score for abstract chunks because they are intrinsically relevant to a GLOBAL query
            abstract_docs = [(doc, max(score, RELEVANCE_THRESHOLD + 0.1)) for doc, score in abstract_docs]
            results_with_scores.extend(abstract_docs)

            try:
                optional_docs = backend.similarity_search_with_score(
                    query=search_query,
                    k=2,
                    collection=content_collection,
                    where={"$and": [{"paper_id": paper_id}, {"section_title": {"$in": ["Introduction", "Conclusion", "Discussion"]}}]}
                )
                results_with_scores.extend(optional_docs)
            except Exception:
                pass  # Fallback gracefully if vector store indexing/querying for $in fails

    k_per_query = max(3, k // len(search_queries))
    for search_query in search_queries:
        content_docs = backend.similarity_search_with_score(
            query=search_query,
            k=k_per_query,
            collection=content_collection,
            where={"paper_id": paper_id}
        )
        results_with_scores.extend(content_docs)

    # Filter by relevance threshold
    filtered = [
        (doc, score) for doc, score in results_with_scores
        if score >= RELEVANCE_THRESHOLD
    ]

    # CRITICAL: Sort by score (descending) before deduplication
    filtered.sort(key=lambda x: x[1], reverse=True)

    # Tag each document with its relevance score for downstream grounding check
    for doc, score in filtered:
        if doc.metadata is None:
            doc.metadata = {}
        doc.metadata["relevance_score"] = score

    # Khử trùng lặp dựa trên nội dung
    seen_content = set()
    unique_docs = []
    for doc, score in filtered:
        content_key = doc.page_content[:150].strip()
        if content_key not in seen_content:
            unique_docs.append(doc)
            seen_content.add(content_key)

    return unique_docs[:k]


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