# RAG PDF API

Hệ thống RAG (Retrieval-Augmented Generation) đa phương thức để xử lý và truy vấn tài liệu PDF.

## Tính năng

- **Upload PDF**: Tải lên và xử lý tài liệu PDF
- **Đa phương thức**: Xử lý text, bảng và hình ảnh
- **Caching thông minh**: Cache summaries và embeddings để tăng tốc
- **API RESTful**: Giao diện API đơn giản và dễ sử dụng

## Cài đặt

1. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

2. Tạo file `.env` với các API keys:
```
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
```

## Sử dụng

1. Khởi động server:
```bash
python api.py
```

2. Upload PDF:
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@your_document.pdf"
```

3. Truy vấn:
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"file_id": "your_file_id", "question": "Câu hỏi của bạn"}'
```

## API Endpoints

- `GET /` - Health check
- `POST /upload` - Upload PDF file
- `POST /query` - Query uploaded PDF
- `GET /status` - System status

## Cấu trúc Project

- `api.py` - FastAPI application
- `config.py` - Configuration management
- `langchain_multimodal.py` - RAG pipeline
- `pdf_extract.py` - PDF processing
- `summarization.py` - Content summarization
- `vectorstore_setup.py` - Vector database setup
- `rag_pipeline.py` - RAG chain construction
