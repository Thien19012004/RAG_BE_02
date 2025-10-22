import base64
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI


def split_docs(docs: List[Any]):
    image_base64_items: List[str] = []
    text_like_items: List[Any] = []
    for item in docs:
        if isinstance(item, str):
            try:
                base64.b64decode(item)
                image_base64_items.append(item)
                continue
            except Exception:
                pass
        text_like_items.append(item)
    return {"images": image_base64_items, "texts": text_like_items}


def build_mm_prompt(kwargs: Dict[str, Any]):
    ctx, question = kwargs["context"], kwargs["question"]
    ctx_text = ""
    for t in ctx["texts"]:
        txt = getattr(t, "text", None)
        if not txt:
            meta = getattr(t, "metadata", None)
            if meta and hasattr(meta, "text_as_html"):
                txt = meta.text_as_html
        if txt:
            ctx_text += f"\n{txt}\n"
    content = [
        {
            "type": "text",
            "text": f"Use the context below to answer.\n\n{ctx_text}\n\nQuestion: {question}",
        }
    ]
    for b64 in ctx["images"]:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return [HumanMessage(content=content)]


def build_rag_chains(retrieve_fn):
    final_llm = ChatOpenAI(model="gpt-4o-mini")

    rag_chain = (
        {
            "context": RunnableLambda(lambda x: retrieve_fn(x)) | RunnableLambda(split_docs),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(build_mm_prompt)
        | final_llm
        | StrOutputParser()
    )

    rag_chain_with_ctx = (
        {
            "context": RunnableLambda(lambda x: retrieve_fn(x)) | RunnableLambda(split_docs),
            "question": RunnablePassthrough(),
        }
        | RunnablePassthrough().assign(
            response=(RunnableLambda(build_mm_prompt) | final_llm | StrOutputParser())
        )
    )

    return rag_chain, rag_chain_with_ctx


