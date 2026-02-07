# 🔬 RAG Scientific - PDF Analysis Service

<div align="center">

![Version](https://img.shields.io/badge/version-3.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)
![License](https://img.shields.io/badge/license-MIT-orange.svg)

**Multimodal RAG (Retrieval-Augmented Generation) Service for Scientific Paper Analysis**

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [API Reference](#-api-reference) • [Configuration](#-configuration)

</div>

---

## 📋 Overview

RAG Scientific là một service xử lý và phân tích tài liệu PDF khoa học sử dụng kỹ thuật RAG đa phương thức. Service hỗ trợ:

- **Text Understanding**: Phân tích nội dung văn bản với semantic chunking
- **Table Extraction**: Trích xuất và hiểu nội dung bảng biểu
- **Image/Figure Analysis**: Phân tích hình ảnh, biểu đồ, sơ đồ
- **Cross-Paper Query**: Truy vấn so sánh đa bài báo

## ✨ Features

| Feature                    | Description                                             |
| -------------------------- | ------------------------------------------------------- |
| 🔍 **Multimodal RAG**      | Kết hợp text, table, image để trả lời câu hỏi toàn diện |
| 📊 **GROBID Integration**  | Trích xuất metadata khoa học (title, abstract, authors) |
| 🧠 **Semantic Chunking**   | Chia nhỏ văn bản theo ngữ nghĩa, không cắt giữa câu     |
| 💾 **Database Caching**    | PostgreSQL-based caching, cloud-ready architecture      |
| ☁️ **Cloud Storage**       | S3/MinIO for PDFs, no local file dependency             |
| 🔗 **arXiv Integration**   | Gợi ý papers liên quan từ arXiv                         |
| 🤖 **Question Brainstorm** | AI tự động đề xuất câu hỏi hay về paper                 |

## 🏗 Architecture

### Cloud-Native Design (v3.0)

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client (NestJS BE)                       │
│                      (owns users, papers, chats)                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                    REST API (rag_file_id linking)
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Application (RAG_BE_02)             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────┐ │
│  │   /upload   │ │   /query    │ │/query-multi │ │ /brainstorm│ │
│  │ /ingest-url │ │/explain-reg.│ │/related-pap.│ │  /status   │ │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └─────┬──────┘ │
└─────────┼───────────────┼───────────────┼──────────────┼────────┘
          │               │               │              │
          ▼               ▼               ▼              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Persistence Layer                            │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────┐│
│  │  PostgreSQL   │  │  S3/Cloud     │  │      ChromaDB         ││
│  │  (metadata,   │  │  Storage      │  │   (vector index)      ││
│  │   summaries)  │  │  (PDFs)       │  │                       ││
│  └───────────────┘  └───────────────┘  └───────────────────────┘│
│         ▲                  ▲                     ▲              │
│         │                  │                     │              │
│         └──────────────────┴─────────────────────┘              │
│                    NO LOCAL FILE STORAGE                         │
└─────────────────────────────────────────────────────────────────┘
```

### Database Tables (RAG Service Owned)

| Table                     | Purpose                                    |
| ------------------------- | ------------------------------------------ |
| `ingested_papers`         | Paper status, metadata, file hash tracking |
| `paper_content_summaries` | Cached LLM summaries for tables/images     |

### Single Source of Truth

| Data              | Owner                      | Table/Field                        |
| ----------------- | -------------------------- | ---------------------------------- |
| PDF Cloud URL     | NestJS (rag-scientific-be) | `papers.file_url`                  |
| Ingestion Status  | RAG_BE_02                  | `ingested_papers.ingestion_status` |
| Paper Metadata    | RAG_BE_02                  | `ingested_papers.*`                |
| LLM Summaries     | RAG_BE_02                  | `paper_content_summaries`          |
| Vector Embeddings | RAG_BE_02                  | ChromaDB (`chroma_store/`)         |

### Data Flow

```
PDF (S3) → Download to Temp → GROBID Parse → Extract (Text/Table/Image)
                                                    │
                                                    ▼
                                            Parallel Summarization
                                                    │
                                                    ▼
                        ┌───────────────────────────┴───────────────────────────┐
                        │                                                       │
                        ▼                                                       ▼
                PostgreSQL                                               ChromaDB
            (metadata + summaries)                                    (embeddings)
                        │                                                       │
                        └───────────────────────────┬───────────────────────────┘
                                                    ▼
                                        Cleanup Temp File
                                                    │
                                                    ▼
                                            Ready for Query ✓
```

## 🚀 Installation

### Prerequisites

- Python 3.10+
- PostgreSQL 14+
- [GROBID Server](https://github.com/kermitt2/grobid) (for metadata extraction)
- OpenAI API Key (for embeddings & vision)
- Groq API Key (for fast text generation)

### 1. Clone & Setup Environment

```bash
# Navigate to project
cd RAG_BE_02

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Required Services

```bash
# Start PostgreSQL (if not already running)
docker run -d --name postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=rag_scientific \
  -p 5432:5432 \
  postgres:14

# Start GROBID Server
docker run -t --rm --init \
  -p 8070:8070 \
  -p 8071:8071 \
  lfoppiano/grobid:0.8.2
```

### 3. Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit .env with your API keys
```

**Required Environment Variables:**

```env
# Database Connection (Required)
RAG_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag_scientific

# API Keys (Required)
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx

# Models (Optional - defaults shown)
GROQ_TEXT_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
VISION_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# Performance Tuning (Optional)
MAX_PARALLEL_TEXT_SUMMARIES=10
MAX_PARALLEL_IMAGE_SUMMARIES=5
API_RATE_LIMIT_DELAY=0.1

# Cloud Storage (Required for production)
S3_ENDPOINT_URL=https://s3.amazonaws.com
S3_BUCKET=your-bucket-name
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
```

### 4. Initialize Database

The database tables will be created automatically on first startup. If migrating from a previous version, run:

```bash
# Migrate existing JSON data to PostgreSQL
python migrate_to_database.py
```

### 5. Start Server

```bash
# Development
python api.py

# Production with Uvicorn
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

Server will be available at `http://localhost:8000`

---

## 📚 API Reference

### Base URL

```
http://localhost:8000
```

### Endpoints Overview

| Method | Endpoint                | Description                         |
| ------ | ----------------------- | ----------------------------------- |
| `POST` | `/upload`               | Upload PDF file trực tiếp           |
| `POST` | `/ingest-from-url`      | Ingest PDF từ S3/Cloud URL          |
| `POST` | `/query`                | Hỏi đáp về 1 paper                  |
| `POST` | `/query-multi`          | Hỏi đáp đa paper (so sánh)          |
| `POST` | `/explain-region`       | Giải thích vùng được chọn trong PDF |
| `POST` | `/brainstorm-questions` | Gợi ý câu hỏi hay về paper          |
| `POST` | `/related-papers`       | Tìm papers liên quan trên arXiv     |
| `GET`  | `/status/{file_id}`     | Kiểm tra trạng thái xử lý           |

---

### 📤 Upload PDF

**Direct Upload:**

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@paper.pdf" \
  -F "file_id=custom-uuid-here"
```

**From Cloud URL (S3/MinIO):**

```bash
curl -X POST "http://localhost:8000/ingest-from-url" \
  -H "Content-Type: application/json" \
  -d '{
    "file_url": "https://bucket.s3.amazonaws.com/papers/paper.pdf",
    "file_id": "custom-uuid-here"
  }'
```

**Response:**

```json
{
  "message": "Uploaded and processed successfully",
  "file_id": "8fc4b997-0165-41c4-8e5c-f2effa478855",
  "status": "completed",
  "processing_time": 45.23,
  "title": "Attention Is All You Need",
  "abstract": "The dominant sequence transduction models...",
  "node_count": 247,
  "table_count": 12,
  "image_count": 8
}
```

---

### 💬 Query Paper

**Single Paper Query:**

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "8fc4b997-0165-41c4-8e5c-f2effa478855",
    "question": "What is the main contribution of this paper?"
  }'
```

**Response:**

```json
{
  "answer": "The main contribution of this paper is the Transformer architecture [S1], which relies entirely on self-attention mechanisms...",
  "context": {
    "texts": [...],
    "tables": [...],
    "images": [...]
  }
}
```

**Multi-Paper Query (Compare/Synthesize):**

```bash
curl -X POST "http://localhost:8000/query-multi" \
  -H "Content-Type: application/json" \
  -d '{
    "file_ids": [
      "uuid-paper-1",
      "uuid-paper-2",
      "uuid-paper-3"
    ],
    "question": "Compare the attention mechanisms used in these papers"
  }'
```

---

### 🔍 Explain Region

Giải thích một vùng được crop từ PDF (figure, table, equation...):

```bash
curl -X POST "http://localhost:8000/explain-region" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "8fc4b997-0165-41c4-8e5c-f2effa478855",
    "image_b64": "data:image/png;base64,iVBORw0KGgo...",
    "page_number": 5,
    "question": "What does this figure show?"
  }'
```

---

### 🧠 Brainstorm Questions

AI tự động đề xuất các câu hỏi thông minh về paper:

```bash
curl -X POST "http://localhost:8000/brainstorm-questions" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "8fc4b997-0165-41c4-8e5c-f2effa478855"
  }'
```

**Response:**

```json
{
  "questions": [
    "How does the self-attention mechanism differ from traditional attention?",
    "What are the computational benefits of removing recurrence?",
    "How does multi-head attention improve model performance?",
    "What datasets were used to evaluate the Transformer?",
    "How does positional encoding work in this architecture?"
  ]
}
```

---

### 📖 Related Papers

Tìm papers liên quan trên arXiv:

```bash
curl -X POST "http://localhost:8000/related-papers" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "8fc4b997-0165-41c4-8e5c-f2effa478855",
    "top_k": 5,
    "max_results": 30
  }'
```

---

### 📊 Check Status

```bash
curl "http://localhost:8000/status/8fc4b997-0165-41c4-8e5c-f2effa478855"
```

**Response:**

```json
{
  "status": "completed",
  "ready": true
}
```

**Status Values:**
| Status | Meaning |
|--------|---------|
| `processing` | Đang xử lý (GROBID, chunking, embedding) |
| `completed` | Hoàn thành, sẵn sàng query |
| `error: <msg>` | Lỗi xử lý |
| `unknown` | File ID không tồn tại |

---

## ⚙️ Configuration

### Directory Structure

```
RAG_BE_02/
├── api.py                 # FastAPI application
├── config.py              # Configuration & FileConfig
├── rag_pipeline.py        # RAG chain construction
├── langchain_multimodal.py # Ingestion pipeline
├── pdf_extract.py         # PDF processing (PyMuPDF + Camelot)
├── summarization.py       # LLM summarization
├── vectorstore_setup.py   # ChromaDB setup
├── arxiv_related.py       # arXiv paper search
├── parallel_processing.py # Async parallel utilities
├── api_utils.py           # API helper functions
│
├── content/               # Uploaded PDFs (local mode)
├── cache/                 # Cached summaries & metadata
│   ├── metadata_registry.json
│   ├── status_registry.json
│   └── {file_id}/
│       ├── paper_metadata.json
│       ├── table_summaries.json
│       └── image_summaries.json
│
└── chroma_store/          # Vector embeddings
    ├── global_store/
    └── {file_id}/
```

### Model Configuration

| Setting           | Default                  | Description                      |
| ----------------- | ------------------------ | -------------------------------- |
| `GROQ_TEXT_MODEL` | `llama-4-scout-17b`      | Model cho text generation (fast) |
| `VISION_MODEL`    | `gpt-4o-mini`            | Model cho image analysis         |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model cho vector embeddings      |

### Performance Tuning

| Setting                        | Default | Description                        |
| ------------------------------ | ------- | ---------------------------------- |
| `MAX_PARALLEL_TEXT_SUMMARIES`  | 10      | Số text chunks summarize song song |
| `MAX_PARALLEL_IMAGE_SUMMARIES` | 5       | Số images summarize song song      |
| `API_RATE_LIMIT_DELAY`         | 0.1s    | Delay giữa các API calls           |
| `CHUNK_MAX_TOKENS`             | 400     | Max tokens per chunk               |

---

## 🔧 Integration Guide

### With NestJS Backend

RAG Service được thiết kế để tích hợp với NestJS backend:

```typescript
// NestJS backend uploads PDF to S3
const s3Url = await s3Service.uploadFile(file);

// Then calls RAG service to ingest
const response = await axios.post(`${RAG_URL}/ingest-from-url`, {
  file_url: s3Url,
  file_id: paper.ragFileId, // UUID generated by NestJS
});

// Save metadata to PostgreSQL
await prisma.paper.update({
  where: { id: paperId },
  data: {
    title: response.data.title,
    abstract: response.data.abstract,
    nodeCount: response.data.node_count,
    status: 'COMPLETED',
  },
});
```

### Key Integration Points

| NestJS Field       | RAG Service Field  | Description                  |
| ------------------ | ------------------ | ---------------------------- |
| `paper.ragFileId`  | `file_id`          | UUID mapping giữa 2 services |
| `paper.status`     | `/status` response | Sync processing status       |
| `paper.nodeCount`  | `node_count`       | Số text chunks               |
| `paper.tableCount` | `table_count`      | Số tables extracted          |
| `paper.imageCount` | `image_count`      | Số images extracted          |

---

## 📈 Performance

### Benchmarks (Apple M2, 16GB RAM)

| Operation                    | Time   | Notes                        |
| ---------------------------- | ------ | ---------------------------- |
| PDF Ingest (10 pages)        | ~30s   | Including GROBID + embedding |
| PDF Ingest (50 pages)        | ~90s   | Parallel summarization helps |
| Single Query                 | ~2-3s  | Retrieval + generation       |
| Multi-Paper Query (3 papers) | ~4-5s  | More context to process      |
| Brainstorm Questions         | ~3s    | Single LLM call              |
| Related Papers               | ~5-10s | arXiv search + LLM ranking   |

### Optimization Tips

1. **Enable Caching**: Summaries được cache, lần query sau sẽ nhanh hơn
2. **Use Groq for Text**: Groq LLaMA nhanh hơn OpenAI 3-5x cho text generation
3. **Parallel Processing**: Tăng `MAX_PARALLEL_*` nếu API rate limit cho phép
4. **Cloud Storage**: Dùng S3 để tránh upload file lớn qua API

---

## 🐛 Troubleshooting

### Common Issues

**1. GROBID Connection Failed**

```
Error: Connection refused to localhost:8070
```

→ Đảm bảo GROBID container đang chạy: `docker ps`

**2. OpenAI Rate Limit**

```
Error: Rate limit exceeded
```

→ Tăng `API_RATE_LIMIT_DELAY` hoặc giảm `MAX_PARALLEL_*`

**3. Out of Memory**

```
Error: Process killed (OOM)
```

→ Giảm `MAX_PARALLEL_*`, xử lý PDF nhỏ hơn (<100 pages)

**4. File Not Found After Restart**

```
Error: Metadata not found for this file
```

→ Pipeline được rebuild từ cache khi query. Chỉ mất nếu xóa `cache/` folder.

---

## 📄 License

MIT License - feel free to use in your projects.

---

<div align="center">

**[⬆ Back to Top](#-rag-scientific---pdf-analysis-service)**

Made with ❤️ for Scientific Research

</div>
