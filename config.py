import os
import hashlib
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent
CONTENT_DIR = BASE_DIR / "content"
CACHE_DIR = BASE_DIR / "cache"
CHROMA_DIR = BASE_DIR / "chroma_store"

# Ensure directories exist
CONTENT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

class Settings:
    """Central settings for the RAG pipeline"""
    # Models
    GROQ_TEXT_MODEL = os.getenv("GROQ_TEXT_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
    VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # Concurrency & Speed Optimization
    MAX_PARALLEL_TEXT_SUMMARIES = int(os.getenv("MAX_PARALLEL_TEXT_SUMMARIES", 10))
    MAX_PARALLEL_IMAGE_SUMMARIES = int(os.getenv("MAX_PARALLEL_IMAGE_SUMMARIES", 5))
    API_RATE_LIMIT_DELAY = float(os.getenv("API_RATE_LIMIT_DELAY", 0.1))

    # Semantic Chunking Parameters
    CHUNK_MAX_TOKENS = 400
    CHUNK_PARA_MAX_TOKENS = 280
    CHUNK_PARA_MIN_TOKENS = 40
    CHUNK_SIMILARITY_THRESHOLD = 0.6

    # Storage Configuration
    USE_CLOUD_STORAGE = bool(os.getenv("S3_BUCKET") or os.getenv("S3_ENDPOINT_URL"))

settings = Settings()


class FileConfig:
    """
    Configuration for a specific file.

    Supports two modes:
    1. Local mode (original): PDF stored locally in CONTENT_DIR
    2. Cloud mode: PDF downloaded from cloud URL to local temp path for processing

    The pdf_path always points to a local file that extraction tools can read.
    """

    def __init__(self, file_id: str, cloud_url: Optional[str] = None):
        """
        Initialize file configuration.

        Args:
            file_id: Unique identifier for this file (used as folder names)
            cloud_url: Optional S3/cloud URL. If provided, PDF will be downloaded
                       from cloud storage before processing.
        """
        self.file_id = file_id
        self.cloud_url = cloud_url

        # Local paths (always used for processing)
        self._local_pdf_path = CONTENT_DIR / f"{file_id}.pdf"
        self.cache_dir = CACHE_DIR / file_id
        self.chroma_path = CHROMA_DIR / file_id
        self.hash_file = self.chroma_path / "last_hash.txt"

        # Ensure directories exist
        self.cache_dir.mkdir(exist_ok=True)
        self.chroma_path.mkdir(exist_ok=True)

        # Track if we need to download from cloud
        self._downloaded = False

    @property
    def pdf_path(self) -> Path:
        """
        Get the local PDF path for processing.

        If cloud_url is set and file doesn't exist locally, download it first.
        This ensures pdf_extract.py and other tools always have a local file.
        """
        if self.cloud_url and not self._local_pdf_path.exists() and not self._downloaded:
            self._download_from_cloud()
        return self._local_pdf_path

    def _download_from_cloud(self) -> None:
        """Download PDF from cloud storage to local path."""
        from storage import get_storage_backend

        storage = get_storage_backend()
        print(f"☁️ [CONFIG] Downloading PDF from cloud: {self.cloud_url}")
        storage.download_file(self.cloud_url, self._local_pdf_path)
        self._downloaded = True
        print(f"✅ [CONFIG] Downloaded to: {self._local_pdf_path}")

    def ensure_local_file(self) -> Path:
        """
        Ensure the PDF file exists locally (download if needed).
        Call this before any processing that requires the file.

        Returns:
            Path to local PDF file
        """
        return self.pdf_path

    def get_file_hash(self) -> str:
        """Get MD5 hash of the PDF file."""
        # Ensure file is downloaded first
        pdf_path = self.pdf_path
        with open(pdf_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def needs_rebuild(self) -> tuple[bool, str]:
        """
        Check if vector store needs to be rebuilt.

        Returns:
            Tuple of (needs_rebuild: bool, pdf_hash: str)
        """
        pdf_hash = self.get_file_hash()
        need_rebuild = True
        if self.hash_file.exists():
            with open(self.hash_file) as f:
                if f.read().strip() == pdf_hash:
                    need_rebuild = False
        return need_rebuild, pdf_hash

    def save_hash(self, pdf_hash: str) -> None:
        """Save the PDF hash to detect changes."""
        with open(self.hash_file, "w") as f:
            f.write(pdf_hash)

    def cleanup_local_file(self) -> None:
        """
        Remove the local PDF file after processing (optional).
        Useful for cloud mode to save disk space.
        """
        if self.cloud_url and self._local_pdf_path.exists():
            try:
                self._local_pdf_path.unlink()
                print(f"🗑️ [CONFIG] Cleaned up local file: {self._local_pdf_path}")
            except Exception as e:
                print(f"⚠️ [CONFIG] Failed to cleanup: {e}")