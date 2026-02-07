import os
import hashlib
import tempfile
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# BASE DIRECTORIES
# =============================================================================
# Only chroma_store is kept for vector index persistence
# cache/ and content/ folders are NO LONGER USED - replaced by:
# - database.py for metadata and status
# - temp files for PDF processing (cleaned up after ingestion)
# =============================================================================

BASE_DIR = Path(__file__).parent
CHROMA_DIR = BASE_DIR / "chroma_store"

# Temp directory for processing PDFs (cleaned up after ingestion)
TEMP_DIR = Path(tempfile.gettempdir()) / "rag_processing"

# Ensure directories exist
CHROMA_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

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
    Configuration for a specific file being processed.

    CLOUD-FIRST ARCHITECTURE:
    - PDFs are stored in cloud storage (S3/MinIO)
    - Local files are temporary, used only during processing
    - All metadata is stored in PostgreSQL database
    - Vector embeddings are stored in ChromaDB

    Processing Flow:
    1. Download PDF from cloud URL to temp directory
    2. Run GROBID + extraction + summarization
    3. Store embeddings in ChromaDB
    4. Store metadata in PostgreSQL
    5. Clean up temp file
    """

    def __init__(self, file_id: str, cloud_url: Optional[str] = None):
        """
        Initialize file configuration.

        Args:
            file_id: Unique identifier for this file (UUID, shared with backend)
            cloud_url: S3/cloud URL where PDF is stored
        """
        self.file_id = file_id
        self.cloud_url = cloud_url

        # Temp paths for processing (cleaned up after ingestion)
        self._temp_pdf_path = TEMP_DIR / f"{file_id}.pdf"

        # ChromaDB path (persistent - stores vector embeddings)
        self.chroma_path = CHROMA_DIR / file_id

        # Ensure chroma directory exists
        self.chroma_path.mkdir(exist_ok=True)

        # Track download status
        self._downloaded = False

    def get_temp_pdf_path(self) -> Path:
        """
        Get the temp path for storing the PDF during processing.

        Used by /upload endpoint to save the uploaded file.
        """
        TEMP_DIR.mkdir(exist_ok=True)
        return self._temp_pdf_path

    @property
    def pdf_path(self) -> Path:
        """
        Get the local PDF path for processing.

        If cloud_url is set and file doesn't exist locally, download it first.
        This ensures pdf_extract.py and other tools always have a local file.
        """
        if self.cloud_url and not self._temp_pdf_path.exists() and not self._downloaded:
            self._download_from_cloud()
        return self._temp_pdf_path

    def _download_from_cloud(self) -> None:
        """Download PDF from cloud storage to temp path."""
        from storage import get_storage_backend

        storage = get_storage_backend()
        print(f"☁️ [CONFIG] Downloading PDF from cloud: {self.cloud_url}")
        TEMP_DIR.mkdir(exist_ok=True)
        storage.download_file(self.cloud_url, self._temp_pdf_path)
        self._downloaded = True
        print(f"✅ [CONFIG] Downloaded to temp: {self._temp_pdf_path}")

    def ensure_local_file(self) -> Path:
        """
        Ensure the PDF file exists locally (download if needed).
        Call this before any processing that requires the file.

        Returns:
            Path to local PDF file (in temp directory)
        """
        return self.pdf_path

    def get_file_hash(self) -> str:
        """Get MD5 hash of the PDF file."""
        pdf_path = self.pdf_path
        with open(pdf_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def needs_rebuild(self) -> tuple[bool, str]:
        """
        Check if vector store needs to be rebuilt.

        Now uses database instead of local hash file.

        Returns:
            Tuple of (needs_rebuild: bool, pdf_hash: str)
        """
        from database import get_database

        pdf_hash = self.get_file_hash()
        db = get_database()

        return db.needs_rebuild(self.file_id, pdf_hash), pdf_hash

    def save_hash(self, pdf_hash: str) -> None:
        """
        Save the PDF hash to detect changes.

        Now uses database instead of local hash file.
        """
        from database import get_database

        db = get_database()
        db.save_file_hash(self.file_id, pdf_hash)

    def cleanup_local_file(self) -> None:
        """
        Remove the local PDF file after processing.

        This is called automatically after successful ingestion
        to ensure no local file storage is used.
        """
        if self._temp_pdf_path.exists():
            try:
                self._temp_pdf_path.unlink()
                print(f"🗑️ [CONFIG] Cleaned up temp file: {self._temp_pdf_path}")
            except Exception as e:
                print(f"⚠️ [CONFIG] Failed to cleanup: {e}")