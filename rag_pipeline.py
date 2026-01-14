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
    system_instructions: Optional[str] = None
    # Biến cờ để xác định chế độ giải thích vùng ảnh
    is_visual_explanation: bool = False


# --- Default Prompts ---
DEFAULT_RAG_INSTRUCTIONS = (
    "You are the best scientific research assistant. "
    "Use ONLY the provided context to answer. Synthesize information from multiple sources when needed. "
    "If asked for a summary, combine information to create a comprehensive answer. "
    "Keep answers concise and cite sources as [S1], [S2], etc. "
    "Ground every claim in the context."
)

REGION_EXPLAIN_INSTRUCTIONS = (
    "You are an expert scientific assistant analyzing a specific region from a research paper. "
    "You are provided with a CROPPED IMAGE of the region and textual context from the paper. "
    "1. Identify if the region is a math formula, table, figure/plot, or text. "
    "2. Explain it clearly for a student audience. "
    "   - Formulas: Define symbols, explain terms/intuition. "
    "   - Tables: Describe columns, units, trends. "
    "   - Plots: Describe axes, variables, insights. "
    "3. Use the provided Textual Context to reduce hallucination. Cite it if helpful."
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
    final_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    
    chain = (
        RunnableLambda(build_mm_prompt)
        | final_llm
        | StrOutputParser()
    )
    return chain


def document_retrieve(
    query: str,
    paper_id: str,
    backend: VectorStoreBackend,
    content_collection: str = CONTENT_COLLECTION,
    k: int = 12,
) -> List[Document]:
    """
    Chiến lược retrieval tối ưu:
    1. Nếu là summary: Lấy 1-2 bản ghi Abstract + (k-2) bản ghi Content liên quan nhất.
    2. Nếu là query thường: Lấy k bản ghi tốt nhất từ toàn bộ Store (bao gồm cả Abstract).
    """
    query_lower = query.lower()
    # Mở rộng bộ từ khóa nhận diện ý định tổng quan
    is_summary = any(x in query_lower for x in ["summary", "overview", "main idea", "summarize", "abstract", "conclusion"])
    
    results: List[Document] = []

    if is_summary:
        # PATH A: Lấy Abstract để nắm ý chính (Bắt buộc)
        abstract_docs = backend.similarity_search(
            query=query, 
            k=1, 
            collection=content_collection, 
            where={"$and": [{"paper_id": paper_id}, {"modality": "abstract"}]}
        )
        results.extend(abstract_docs)
        
        # PATH B: Lấy thêm Content để có chi tiết (Không loại trừ phần nào)
        # k giảm xuống một chút để nhường chỗ cho Abstract
        search_k = max(8, k - len(results))
        content_docs = backend.similarity_search(
            query=query, 
            k=search_k, 
            collection=content_collection, 
            where={"paper_id": paper_id} # Tìm toàn bộ để không sót
        )
        results.extend(content_docs)
    else:
        # Truy vấn bình thường: Cho phép tự do tìm kiếm dựa trên độ tương đồng
        # Abstract vẫn có thể xuất hiện nếu nó thực sự liên quan đến câu hỏi
        results = backend.similarity_search(
            query=query, 
            k=k, 
            collection=content_collection, 
            where={"paper_id": paper_id}
        )

    # Khử trùng lặp dựa trên nội dung (Tránh việc Abstract bị lấy 2 lần)
    seen_content = set()
    unique_docs = []
    for doc in results:
        # Dùng hash hoặc 100 ký tự đầu làm key
        content_key = doc.page_content[:150].strip()
        if content_key not in seen_content:
            unique_docs.append(doc)
            seen_content.add(content_key)
            
    return unique_docs[:k]


def build_document_rag_chain(
    paper_id: str,
    backend: VectorStoreBackend,
    prompt_cfg: Optional[PromptConfig] = None,
):
    """
    Builds a standard RAG chain for the /query endpoint.
    Retrieval is baked in.
    """
    prompt_cfg = prompt_cfg or PromptConfig(paper_id=paper_id)
    
    # 1. Retrieval Step
    def _retrieve_step(input_dict):
        q = input_dict["question"]
        docs = document_retrieve(q, paper_id, backend)
        return split_docs(docs)

    # 2. Generation Step (Reusing the shared generative chain)
    gen_chain = build_generative_chain()

    # 3. Compose
    rag_chain = (
        {
            "context": RunnableLambda(_retrieve_step),
            "question": lambda x: x["question"],
            "prompt_cfg": lambda _: prompt_cfg,
            "focus_image_b64": lambda _: None, # No focus image in standard query
        }
        | gen_chain 
        | RunnableLambda(lambda x: {"response": x}) # Wrap for compatibility
    )
    
    # Inject context into output for API response
    final_chain = (
        RunnablePassthrough.assign(context=RunnableLambda(_retrieve_step))
        .assign(response=lambda x: gen_chain.invoke({
            "context": x["context"], 
            "question": x["question"],
            "prompt_cfg": prompt_cfg,
            "focus_image_b64": None
        }))
    )

    return final_chain


def brainstorm_questions_chain(title: str, abstract: str) -> List[str]:
    """
    Sử dụng LLM để tạo ra các câu hỏi gợi ý dựa trên Title và Abstract.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7) # Tăng temp để sáng tạo hơn
    
    prompt = (
        f"You are a research expert. Based on the following research paper metadata, "
        f"suggest 5-8 thought-provoking and diverse questions that a reader should ask "
        f"to understand the paper's contributions, methodology, and results more deeply.\n\n"
        f"Title: {title}\n"
        f"Abstract: {abstract}\n\n"
        f"Output requirements:\n"
        f"- Return ONLY a JSON list of strings.\n"
        f"- Format: [\"question 1\", \"question 2\", ...]\n"
        f"- Questions should be concise and professional."
    )
    
    # Sử dụng .with_structured_output nếu muốn chắc chắn về định dạng (hoặc parse tay)
    response = llm.invoke(prompt)
    
    try:
        # Cố gắng parse JSON từ response
        import json
        content = response.content.strip()
        # Loại bỏ markdown code blocks nếu có
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        print(f"Error parsing brainstormed questions: {e}")
        # Fallback: trả về list rỗng hoặc các dòng text
        return [line.strip("- ") for line in response.content.split("\n") if len(line) > 10][:6]