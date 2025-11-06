import json
import uuid
from typing import Any, Dict, List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


ID_KEY = "doc_id"


def build_vectorstore(chroma_path: str) -> Tuple[Chroma, Dict[str, Any]]:
    """Build vector store with dynamic path"""
    emb_fn = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        collection_name="multi_modal_rag",
        embedding_function=emb_fn,
        persist_directory=chroma_path,
    )
    return vectorstore, {}


def add_group_to_store(
    vectorstore: Chroma,
    docstore: Dict[str, Any],
    originals: List[Any],
    summaries: List[str],
):
    """Add a group of documents to the vector store"""
    if not originals:
        print("⚠️ No originals to add to store")
        return
    
    if len(originals) != len(summaries):
        print(f"⚠️ Mismatch: {len(originals)} originals vs {len(summaries)} summaries")
        return
    
    # Filter out empty summaries and their corresponding originals
    valid_items = []
    for i, (orig, summary) in enumerate(zip(originals, summaries)):
        if summary and summary.strip():
            valid_items.append((orig, summary.strip()))
        else:
            print(f"⚠️ Skipping item {i}: empty summary")
    
    if not valid_items:
        print("⚠️ No valid items to add to vector store")
        return
    
    print(f"📚 Adding {len(valid_items)} valid items to vector store")
    
    # Create documents for valid items only
    ids = [str(uuid.uuid4()) for _ in valid_items]
    docs = [Document(page_content=summary, metadata={ID_KEY: ids[i]}) for i, (_, summary) in enumerate(valid_items)]
    
    if docs:
        vectorstore.add_documents(docs)
        print(f"✅ Added {len(docs)} documents to vector store")
    
    # Store originals in docstore
    for i, (orig, _) in enumerate(valid_items):
        docstore[ids[i]] = orig


def persist_docstore_index(docstore: Dict[str, Any], cache_file: str):
    """Persist docstore index to file"""
    json.dump(list(docstore.keys()), open(cache_file, "w"))


def retrieve_parents(vectorstore: Chroma, docstore: Dict[str, Any], query: str, k: int = 6):
    """Retrieve parent documents from vector store"""
    hits = vectorstore.similarity_search(query, k=k)
    return [docstore[h.metadata[ID_KEY]] for h in hits if h.metadata.get(ID_KEY) in docstore]


