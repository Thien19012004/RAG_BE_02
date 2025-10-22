import json, os, uuid
from typing import Any, Dict, List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config import CHROMA_PATH, CACHE_DIR


ID_KEY = "doc_id"


def build_vectorstore() -> Tuple[Chroma, Dict[str, Any]]:
    emb_fn = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        collection_name="multi_modal_rag",
        embedding_function=emb_fn,
        persist_directory=CHROMA_PATH,
    )
    return vectorstore, {}


def add_group_to_store(
    vectorstore: Chroma,
    docstore: Dict[str, Any],
    originals: List[Any],
    summaries: List[str],
):
    if not originals:
        return
    ids = [str(uuid.uuid4()) for _ in originals]
    docs = [Document(page_content=summaries[i] or "", metadata={ID_KEY: ids[i]}) for i in range(len(originals))]
    if docs:
        vectorstore.add_documents(docs)
    for k, v in zip(ids, originals):
        docstore[k] = v


def persist_docstore_index(docstore: Dict[str, Any]):
    json.dump(list(docstore.keys()), open(f"{CACHE_DIR}/docstore_index.json", "w"))


def retrieve_parents(vectorstore: Chroma, docstore: Dict[str, Any], query: str, k: int = 6):
    hits = vectorstore.similarity_search(query, k=k)
    return [docstore[h.metadata[ID_KEY]] for h in hits if h.metadata.get(ID_KEY) in docstore]


