import os, hashlib
from dotenv import load_dotenv

# Load environment variables (OPENAI_API_KEY, GROQ_API_KEY, etc.)
load_dotenv()

# Paths and directories
PDF_PATH = "./content/paper2.pdf"
CACHE_DIR = "./cache"
CHROMA_PATH = "./chroma_store"
HASH_FILE = os.path.join(CHROMA_PATH, "last_hash.txt")

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)


def get_file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def compute_rebuild_flag(pdf_path: str = PDF_PATH, hash_file: str = HASH_FILE) -> tuple[bool, str]:
    pdf_hash = get_file_hash(pdf_path)
    need_rebuild = True
    if os.path.exists(hash_file):
        with open(hash_file) as f:
            if f.read().strip() == pdf_hash:
                need_rebuild = False
    return need_rebuild, pdf_hash


