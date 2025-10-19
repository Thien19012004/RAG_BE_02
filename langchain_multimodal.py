# -*- coding: utf-8 -*-
"""
LangChain Multi-modal RAG (with cache)
- Groq for text/table summarization
- OpenAI gpt-4o-mini for image understanding
- OpenAI embeddings + Chroma for retrieval
- Automatic caching for summaries + embeddings
"""

import os, time, base64, uuid, hashlib, json
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()  # load OPENAI_API_KEY, GROQ_API_KEY

# ===================== 1) PDF EXTRACT =====================
from unstructured.partition.pdf import partition_pdf

PDF_PATH = "./content/paper1.pdf"
CACHE_DIR = "./cache"
CHROMA_PATH = "./chroma_store"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)

def get_file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

HASH_FILE = os.path.join(CHROMA_PATH, "last_hash.txt")
pdf_hash = get_file_hash(PDF_PATH)
need_rebuild = True
if os.path.exists(HASH_FILE):
    with open(HASH_FILE) as f:
        if f.read().strip() == pdf_hash:
            need_rebuild = False
print("🔍 PDF changed:", need_rebuild)

chunks = partition_pdf(
    filename=PDF_PATH,
    infer_table_structure=True,
    strategy="hi_res",
    extract_image_block_types=["Image"],
    extract_image_block_to_payload=True,
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

import re
from collections import Counter

# --- Phát hiện & lọc header/footer xuất hiện quá nhiều lần ---
text_blocks = [el.text.strip() for el in texts if el.text.strip()]
# Đếm tần suất xuất hiện của các dòng ngắn (<= 50 ký tự)
short_lines = [t for t in text_blocks if len(t) <= 50]
freq = Counter(short_lines)

# Lấy các dòng xuất hiện ≥ 3 lần → có khả năng là header/footer
repeated_headers = {line for line, c in freq.items() if c >= 3}

cleaned_texts = []
for el in texts:
    text = el.text.strip()
    # Bỏ nếu text nằm trong danh sách header/footer
    if text and text not in repeated_headers:
        cleaned_texts.append(el)

texts = cleaned_texts


def get_images_base64(all_chunks):
    images_b64 = []
    for ch in all_chunks:
        if "CompositeElement" in str(type(ch)):
            for el in ch.metadata.orig_elements:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64

images = get_images_base64(chunks)

# ================= 2) SUMMARIZATION =====================
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

GROQ_TEXT_MODEL = "llama-3.3-70b-versatile"
TEXT_SLEEP_SECONDS = 3.0
VISION_MODEL = "gpt-4o-mini"
VISION_SLEEP_SECONDS = 3.0

text_llm = ChatGroq(model=GROQ_TEXT_MODEL, groq_api_key=os.getenv("GROQ_API_KEY"))
vision_llm = ChatOpenAI(model=VISION_MODEL)

prompt_text = ChatPromptTemplate.from_template(
    """You are an assistant tasked with summarizing tables and text.
Give a concise, faithful summary of the content below.
Content:
{element}"""
)
summarize_chain = prompt_text | text_llm | StrOutputParser()

vision_prompt = ChatPromptTemplate.from_messages([
    ("user", [
        {"type": "text", "text": "Describe this image from a research paper."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_b64}"}},
    ]),
])
vision_chain = vision_prompt | vision_llm | StrOutputParser()

def summarize_with_cache(items, cache_file, to_str=lambda x: x, sleep_s=0.0):
    if os.path.exists(cache_file) and not need_rebuild:
        print(f"📦 Loaded cache {os.path.basename(cache_file)}")
        return json.load(open(cache_file, encoding="utf-8"))
    out = []
    for i, it in enumerate(items):
        try:
            s = summarize_chain.invoke({"element": to_str(it)})
            out.append(s)
            print(f"✅ Summarized {i+1}/{len(items)}")
            time.sleep(sleep_s)
        except Exception as e:
            print(f"⚠️ Skip {i+1}: {e}")
            out.append("")
    json.dump(out, open(cache_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return out

def summarize_images_with_cache(imgs, cache_file, sleep_s=0.0):
    if os.path.exists(cache_file) and not need_rebuild:
        print(f"📦 Loaded image cache {os.path.basename(cache_file)}")
        return json.load(open(cache_file, encoding="utf-8"))
    outs = []
    for i, b64 in enumerate(imgs):
        try:
            s = vision_chain.invoke({"image_b64": b64})
            outs.append(s)
            print(f"🖼️ Summarized image {i+1}/{len(imgs)}")
            time.sleep(sleep_s)
        except Exception as e:
            print(f"⚠️ Skip image {i+1}: {e}")
            outs.append("")
    json.dump(outs, open(cache_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return outs

text_summaries = summarize_with_cache(texts, f"{CACHE_DIR}/text_summaries.json", to_str=lambda x: x.text, sleep_s=TEXT_SLEEP_SECONDS)
tables_html = [t.metadata.text_as_html for t in tables]
table_summaries = summarize_with_cache(tables_html, f"{CACHE_DIR}/table_summaries.json", to_str=lambda x: x, sleep_s=TEXT_SLEEP_SECONDS)
image_summaries = summarize_images_with_cache(images, f"{CACHE_DIR}/image_summaries.json", sleep_s=VISION_SLEEP_SECONDS)

# ================ 3) VECTORSTORE ================
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

emb_fn = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    collection_name="multi_modal_rag",
    embedding_function=emb_fn,
    persist_directory=CHROMA_PATH
)

DOCSTORE, ID_KEY = {}, "doc_id"

def _add_group_to_store(originals, summaries):
    if not originals: return
    ids = [str(uuid.uuid4()) for _ in originals]
    docs = [Document(page_content=summaries[i] or "", metadata={ID_KEY: ids[i]}) for i in range(len(originals))]
    if docs:
        vectorstore.add_documents(docs)
    for k, v in zip(ids, originals):
        DOCSTORE[k] = v

if need_rebuild:
    print("📗 Rebuilding Chroma store...")
    _add_group_to_store(texts, text_summaries)
    _add_group_to_store(tables, table_summaries)
    _add_group_to_store(images, image_summaries)
    json.dump(list(DOCSTORE.keys()), open(f"{CACHE_DIR}/docstore_index.json", "w"))
    open(HASH_FILE, "w").write(pdf_hash)
else:
    print("📗 Using existing Chroma (no rebuild)")

def retrieve_parents(query, k=6):
    hits = vectorstore.similarity_search(query, k=k)
    return [DOCSTORE[h.metadata[ID_KEY]] for h in hits if h.metadata.get(ID_KEY) in DOCSTORE]

# ================= 4) RAG =================
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage

def split_docs(docs):
    img_b64, textish = [], []
    for d in docs:
        if isinstance(d, str):
            try:
                base64.b64decode(d); img_b64.append(d); continue
            except: pass
        textish.append(d)
    return {"images": img_b64, "texts": textish}

def build_mm_prompt(kwargs):
    ctx, q = kwargs["context"], kwargs["question"]
    ctx_text = ""
    for t in ctx["texts"]:
        txt = getattr(t, "text", None)
        if not txt:
            meta = getattr(t, "metadata", None)
            if meta and hasattr(meta, "text_as_html"):
                txt = meta.text_as_html
        if txt: ctx_text += f"\n{txt}\n"
    content = [{"type": "text", "text": f"Use the context below to answer.\n\n{ctx_text}\n\nQuestion: {q}"}]
    for b64 in ctx["images"]:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return [HumanMessage(content=content)]

final_llm = ChatOpenAI(model="gpt-4o-mini")

rag_chain = (
    {"context": RunnableLambda(lambda x: retrieve_parents(x)) | RunnableLambda(split_docs),
     "question": RunnablePassthrough()}
    | RunnableLambda(build_mm_prompt)
    | final_llm
    | StrOutputParser()
)

rag_chain_with_ctx = {
    "context": RunnableLambda(lambda x: retrieve_parents(x)) | RunnableLambda(split_docs),
    "question": RunnablePassthrough(),
} | RunnablePassthrough().assign(
    response=(RunnableLambda(build_mm_prompt) | final_llm | StrOutputParser())
)

# ================= DEMO =================
if __name__ == "__main__":
    q1 = "What is pythagorean theorem?"
    print("\nQ:", q1)
    print("A:", rag_chain.invoke(q1))

    q2 = "What is Ptolemy's generalization of Pythagorean theorem?"
    print("\nQ:", q2)
    out = rag_chain_with_ctx.invoke(q2)
    print("A:", out["response"])

    print("\n--- Context preview ---")
    for t in out["context"]["texts"][:2]:
        try: print((getattr(t, "text", "") or "")[:400], "\n-----")
        except: pass
