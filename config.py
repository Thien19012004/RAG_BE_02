import os
import hashlib
from pathlib import Path
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

settings = Settings()

class FileConfig:
    """Configuration for a specific file"""
    def __init__(self, file_id: str):
        self.file_id = file_id
        self.pdf_path = CONTENT_DIR / f"{file_id}.pdf"
        self.cache_dir = CACHE_DIR / file_id
        self.chroma_path = CHROMA_DIR / file_id
        self.hash_file = self.chroma_path / "last_hash.txt"
        
        self.cache_dir.mkdir(exist_ok=True)
        self.chroma_path.mkdir(exist_ok=True)
    
    def get_file_hash(self) -> str:
        with open(self.pdf_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def needs_rebuild(self) -> tuple[bool, str]:
        pdf_hash = self.get_file_hash()
        need_rebuild = True
        if self.hash_file.exists():
            with open(self.hash_file) as f:
                if f.read().strip() == pdf_hash:
                    need_rebuild = False
        return need_rebuild, pdf_hash
    
    def save_hash(self, pdf_hash: str):
        with open(self.hash_file, "w") as f:
            f.write(pdf_hash)