from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


DEFAULT_EMBED_MODEL = "text-embedding-3-small"
ABSTRACT_COLLECTION = "abstract_store"
CONTENT_COLLECTION = "content_store"


class VectorStoreBackend(Protocol):
    """Minimal interface for vector backends so we can swap implementations easily."""

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
    """Configuration payload for vector backends."""

    persist_dir: Path
    embedding_model: str = DEFAULT_EMBED_MODEL


class LocalChromaBackend:
    """Disk-based Chroma backend that satisfies the VectorStoreBackend protocol."""

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
        return store.similarity_search(query, k=k, filter=where)

    def delete_where(self, collection: str, where: Dict[str, Any]) -> None:
        store = self._get_store(collection)
        store.delete(where=where)


def get_embedding_model(model: str = DEFAULT_EMBED_MODEL) -> OpenAIEmbeddings:
    """Helper to build a shared embedding model instance."""
    return OpenAIEmbeddings(model=model)


def build_local_chroma_backend(base_path: str, model: str = DEFAULT_EMBED_MODEL) -> LocalChromaBackend:
    """Factory for the default LocalChroma backend."""
    return LocalChromaBackend(
        BackendInitConfig(
            persist_dir=Path(base_path),
            embedding_model=model,
        )
    )

