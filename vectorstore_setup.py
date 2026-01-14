from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

DEFAULT_EMBED_MODEL = "text-embedding-3-small"
# Chỉ còn duy nhất một collection cho toàn bộ nội dung paper
CONTENT_COLLECTION = "content_store"

class VectorStoreBackend(Protocol):
    def add_documents(self, docs: List[Document], collection: str) -> None:
        ...

    def similarity_search(
        self,
        query: str,
        k: int,
        collection: str,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        ...

    def delete_where(self, collection: str, where: Dict[str, Any]) -> None:
        ...

@dataclass
class BackendInitConfig:
    persist_dir: Path
    embedding_model: str = DEFAULT_EMBED_MODEL

class LocalChromaBackend:
    def __init__(self, config: BackendInitConfig):
        self.config = config
        self.config.persist_dir.mkdir(parents=True, exist_ok=True)
        self._embedding_fn = OpenAIEmbeddings(model=config.embedding_model)
        self._stores: Dict[str, Chroma] = {}

    def _get_store(self, collection: str) -> Chroma:
        if collection not in self._stores:
            collection_path = self.config.persist_dir / collection
            collection_path.mkdir(exist_ok=True)
            self._stores[collection] = Chroma(
                collection_name=collection,
                embedding_function=self._embedding_fn,
                persist_directory=str(collection_path),
            )
        return self._stores[collection]

    def add_documents(self, docs: List[Document], collection: str) -> None:
        if not docs:
            return
        store = self._get_store(collection)
        store.add_documents(docs)

    def similarity_search(
        self,
        query: str,
        k: int,
        collection: str,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        store = self._get_store(collection)
        # ChromaDB sử dụng tham số 'filter' cho siêu dữ liệu
        return store.similarity_search(query, k=k, filter=where)

    def delete_where(self, collection: str, where: Dict[str, Any]) -> None:
        store = self._get_store(collection)
        store.delete(where=where)

def get_embedding_model(model: str = DEFAULT_EMBED_MODEL) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=model)

def build_local_chroma_backend(base_path: str, model: str = DEFAULT_EMBED_MODEL) -> LocalChromaBackend:
    return LocalChromaBackend(
        BackendInitConfig(
            persist_dir=Path(base_path),
            embedding_model=model,
        )
    )