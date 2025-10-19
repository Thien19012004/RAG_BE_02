# -*- coding: utf-8 -*-
"""
Refactor for LangChain 1.x — Multi-modal RAG
- Groq (Llama-3.1-70B-versatile) for text/table summarization (sequential with sleep)
- OpenAI gpt-4o-mini for image understanding (vision)
- OpenAI text-embedding-3-small for embeddings (Chroma)
- Reads GROQ_API_KEY, OPENAI_API_KEY from .env

Requirements:
  pip install -U "unstructured[all-docs]" pillow lxml chromadb tiktoken \
      langchain langchain-community langchain-openai langchain-groq python-dotenv

Windows OCR:
  - Tesseract (bắt buộc khi strategy="hi_res")
  - (Khuyên) Poppler

Place your PDF at ./content/attention.pdf
"""

import os
import time
import base64
import uuid
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()  # OPENAI_API_KEY, GROQ_API_KEY

# ===================== 1) PDF EXTRACT (unstructured) =====================
from unstructured.partition.pdf import partition_pdf

PDF_PATH = "./content/paper1.pdf"

# Nếu chưa có OCR, tạm đổi strategy="fast" để chạy không cần Tesseract/Poppler
chunks = partition_pdf(
    filename=PDF_PATH,
    infer_table_structure=True,
    strategy="hi_res",                   # "hi_res" cần OCR; có thể đổi "fast" nếu muốn bỏ OCR
    extract_image_block_types=["Image"], # trích ảnh ra
    extract_image_block_to_payload=True, # đưa ảnh base64 vào metadata
    chunking_strategy="by_title",
    max_characters=10000,
    combine_text_under_n_chars=2000,
    new_after_n_chars=6000,
)

tables, texts = [], []
for ch in chunks:
    tname = str(type(ch))
    if "Table" in tname:
        tables.append(ch)
    if "CompositeElement" in tname:
        texts.append(ch)

def get_images_base64(all_chunks) -> List[str]:
    images_b64 = []
    for ch in all_chunks:
        if "CompositeElement" in str(type(ch)):
            for el in ch.metadata.orig_elements:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64

images: List[str] = get_images_base64(chunks)

# ================= 2) SUMMARIZATION (Groq for text/table) ================
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# ---- 2.1 Text/Table summarization with Groq (sequential + sleep) ----
GROQ_TEXT_MODEL = "llama-3.3-70b-versatile"
TEXT_SLEEP_SECONDS = 3.0   # giảm tốc để né rate limit

text_llm = ChatGroq(
    model=GROQ_TEXT_MODEL,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

prompt_text = ChatPromptTemplate.from_template(
    """You are an assistant tasked with summarizing tables and text.
Give a concise, faithful summary of the content below (no preambles, no bullet labels).
Content:
{element}"""
)
summarize_chain = prompt_text | text_llm | StrOutputParser()

def summarize_sequential(items: List[Any], to_str=lambda x: x, sleep_s: float = 0.0) -> List[str]:
    out = []
    for i, it in enumerate(items):
        try:
            summary = summarize_chain.invoke({"element": to_str(it)})
            out.append(summary)
            print(f"✅ Summarized {i+1}/{len(items)}")
            if sleep_s > 0:
                time.sleep(sleep_s)
        except Exception as e:
            print(f"⚠️ Skip {i+1}: {e}")
            # vẫn đẩy phần tử rỗng để giữ chỉ số nếu cần
            out.append("")
    return out

text_summaries: List[str] = summarize_sequential(texts, to_str=lambda x: x.text, sleep_s=TEXT_SLEEP_SECONDS)
tables_html: List[str] = [tbl.metadata.text_as_html for tbl in tables]
table_summaries: List[str] = summarize_sequential(tables_html, to_str=lambda x: x, sleep_s=TEXT_SLEEP_SECONDS)

# ---- 2.2 Image summarization with OpenAI vision (gpt-4o-mini) ----
VISION_MODEL = "gpt-4o-mini"
VISION_SLEEP_SECONDS = 3   # nhẹ để tránh spam

vision_llm = ChatOpenAI(model=VISION_MODEL)  # needs OPENAI_API_KEY

vision_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            [
                {
                    "type": "text",
                    "text": "Describe this image in detail. The image comes from a research paper about Transformers. If a plot appears, mention axes and trends."
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image_b64}"}
                },
            ],
        ),
    ]
)
vision_chain = vision_prompt | vision_llm | StrOutputParser()

def summarize_images(images_b64: List[str], sleep_s: float = 0.0) -> List[str]:
    outs = []
    for i, b64 in enumerate(images_b64):
        try:
            s = vision_chain.invoke({"image_b64": b64})
            outs.append(s)
            print(f"🖼️ Summarized image {i+1}/{len(images_b64)}")
            if sleep_s > 0:
                time.sleep(sleep_s)
        except Exception as e:
            print(f"⚠️ Skip image {i+1}: {e}")
            outs.append("")
    return outs

image_summaries: List[str] = summarize_images(images, sleep_s=VISION_SLEEP_SECONDS)

# ================ 3) VECTORSTORE (OpenAI Embeddings + Chroma) ============
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

emb_fn = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    collection_name="multi_modal_rag",
    embedding_function=emb_fn,
    persist_directory="./chroma_store"  # để dùng lại giữa các lần chạy
)

# “Docstore cha” để thay cho MultiVectorRetriever cũ
DOCSTORE: Dict[str, Any] = {}
ID_KEY = "doc_id"

def _add_group_to_store(originals: List[Any], summaries: List[str]):
    if not originals:
        return
    ids = [str(uuid.uuid4()) for _ in originals]
    # nạp docs tóm tắt vào vectorDB (metadata trỏ id cha)
    summary_docs = [
        Document(page_content=summaries[i] or "", metadata={ID_KEY: ids[i]})
        for i in range(len(originals))
    ]
    if summary_docs:
        vectorstore.add_documents(summary_docs)
    # lưu bản gốc (cha) trong dict
    for k, v in zip(ids, originals):
        DOCSTORE[k] = v

# nạp đủ 3 nhóm
_add_group_to_store(texts, text_summaries)
_add_group_to_store(tables, table_summaries)
_add_group_to_store(images, image_summaries)

def retrieve_parents(query: str, k: int = 6) -> List[Any]:
    hits = vectorstore.similarity_search(query, k=k)
    parents = []
    for h in hits:
        pid = h.metadata.get(ID_KEY)
        if pid in DOCSTORE:
            parents.append(DOCSTORE[pid])
    return parents

# ================= 4) RAG chain (text + image) ==========================
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage

def split_docs(docs: List[Any]) -> Dict[str, List[Any]]:
    """Tách base64 images và text/table gốc từ DOCSTORE."""
    img_b64, textish = [], []
    for d in docs:
        if isinstance(d, str):
            # thử decode để chắc là base64 của ảnh
            try:
                base64.b64decode(d)
                img_b64.append(d)
                continue
            except Exception:
                pass
        textish.append(d)
    return {"images": img_b64, "texts": textish}

def build_mm_prompt(kwargs: Dict[str, Any]) -> List[HumanMessage]:
    ctx = kwargs["context"]
    question = kwargs["question"]

    # Xây context text
    context_text = ""
    for t in ctx["texts"]:
        txt = getattr(t, "text", None)
        # Table của unstructured có html
        if not txt:
            meta = getattr(t, "metadata", None)
            if meta is not None and hasattr(meta, "text_as_html"):
                txt = meta.text_as_html
        if txt:
            context_text += f"\n{txt}\n"

    content = [
        {
            "type": "text",
            "text": (
                "Answer the question strictly using the context below. "
                "Context may include text, tables (as HTML), and images.\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {question}"
            ),
        }
    ]
    for b64 in ctx["images"]:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return [HumanMessage(content=content)]

# Dùng OpenAI vision cho bước trả lời cuối cùng (vì có thể có ảnh)
final_llm = ChatOpenAI(model="gpt-4o-mini")

rag_chain = (
    {
        "context": RunnableLambda(lambda x: retrieve_parents(x)) | RunnableLambda(split_docs),
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(build_mm_prompt)
    | final_llm
    | StrOutputParser()
)

# Kèm debug context
rag_chain_with_ctx = {
    "context": RunnableLambda(lambda x: retrieve_parents(x)) | RunnableLambda(split_docs),
    "question": RunnablePassthrough(),
} | RunnablePassthrough().assign(
    response=(RunnableLambda(build_mm_prompt) | final_llm | StrOutputParser())
)

# ================= 5) DEMO ==============================================
if __name__ == "__main__":
    q1 = "What is pythagorean theorem?"
    print("\nQ:", q1)
    print("A:", rag_chain.invoke(q1))

    q2 = "What is ptolemy's generalization of Pythagorean theorem?"
    print("\nQ:", q2)
    out = rag_chain_with_ctx.invoke(q2)
    print("A:", out["response"])

    print("\n--- Context preview (first 2 text items) ---")
    for t in out["context"]["texts"][:2]:
        try:
            print((getattr(t, "text", "") or "")[:400], "\n-----")
        except Exception:
            pass
