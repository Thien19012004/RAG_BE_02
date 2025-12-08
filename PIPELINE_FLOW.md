## RAG PDF API – Luồng hoạt động kiến trúc mới

### 1. Upload PDF (`POST /upload`)

- **Bước 1 – Nhận file & tạo `file_id`**
  - FastAPI nhận `UploadFile` và optional `file_id` từ FE.
  - Nếu không có `file_id`, backend sinh một UUID mới.
  - File được lưu vào `content/{file_id}.pdf` thông qua `FileConfig`.
  - Trạng thái trong bộ nhớ: `processing_status[file_id] = "queued"`.

- **Bước 2 – Khởi động xử lý nền**
  - Backend gọi `background_tasks.add_task(build_pipeline_sync, file_config)`.
  - Endpoint `/upload` trả về ngay: `status="processing"` kèm `file_id` để FE poll `/status` / `/status/{file_id}`.

- **Bước 3 – Ingest trong background (`build_pipeline_sync`)**
  - `build_pipeline_sync` gọi `ingest_document(file_config, VECTOR_BACKEND)` để:
    - Chạy pipeline extract + summarization + build semantic nodes.
    - Ghi abstract + content docs vào vector store.
  - Sau khi ingest xong, hàm tạo document-mode RAG chain:
    - `build_document_rag(paper_id=file_id, backend=VECTOR_BACKEND, prompt_cfg=PromptConfig(...))`.
    - Lưu vào bộ nhớ: `pipelines[file_id] = {"chains": (rag_chain, rag_chain_with_ctx), "metadata": IngestionResult}`.
  - Cập nhật trạng thái: `processing_status[file_id] = "completed"` (hoặc `"error: ..."` nếu lỗi).


### 2. Dual Extraction – GROBID + Unstructured (`pdf_extract.py`)

- **GROBID / Fallback – Text sạch & cấu trúc**
  - `run_grobid(pdf_path, paper_id)`:
    - Nếu có `GROBID_URL` → gọi HTTP đến GROBID (`/api/processFulltextDocument`) – hiện mới là stub, chưa parse TEI chi tiết.
    - Nếu không (hoặc lỗi) → fallback: PyMuPDF (`fitz`) đọc từng trang:
      - Mỗi trang tạo một `GrobidSection` với `title="Page i"`, `text=page_text`, `page_start/page_end=i`.
    - Kết quả gom thành `GrobidText`:
      - `title`, `authors`, `abstract` (cắt 500 ký tự đầu từ trang 1 nếu không có abstract chuẩn), `sections` (danh sách `GrobidSection`).  

- **Unstructured – Layout + Multimodal**
  - `partition_pdf_into_chunks(pdf_path)` dùng `unstructured.partition.pdf` với `strategy="hi_res"` và `infer_table_structure=True`:
    - Trả về list phần tử gồm `CompositeElement` (text), `Table`, `Image`.
  - `split_tables_and_texts(chunks)`:
    - Lọc ra 2 mảng: `tables`, `texts`.
  - `remove_repeated_headers(texts)`:
    - Loại bớt header lặp lại nhiều lần trên các trang (dựa trên tần suất các dòng ngắn).
  - `get_images_base64(chunks)`:
    - Duyệt `orig_elements` trong `CompositeElement`, trích `metadata.image_base64` cho các `Image` – phục vụ multimodal vision.


### 3. Semantic Chunking – Semantic Nodes (`build_semantic_nodes`)

- Input:
  - `paper_id`
  - Danh sách `GrobidSection` (text sạch, theo section/page)
  - Hàm embed `embed_fn: Callable[[List[str]], List[Sequence[float]]]` (từ OpenAI embeddings).

- Algorithm (kiểu Semantic NodeSplitter):
  - Mỗi section:
    - Chuẩn hóa text (`replace("\n", " ")`), split câu bằng regex `(?<=[.!?])\s+`.
    - Embed từng câu → `embeddings`.
    - Duy trì một node hiện tại: danh sách câu, các vector, đếm token ước lượng.
    - Nếu thêm câu mới mà:
      - cosine similarity với centroid < `similarity_threshold`, **hoặc**
      - số token vượt `max_tokens`  
      → **kết thúc node hiện tại**, bắt đầu node mới.
  - Mỗi node được lưu thành `SemanticNode`:
    - `paper_id`, `section_title`, `order_idx`
    - `text` (gộp các câu trong node)
    - `approx_page_start`, `approx_page_end` (từ section).

- Các node này là đơn vị chính cho **text retrieval** trong content store.


### 4. Multimodal Summarization & Cache (`summarization.py`, `parallel_processing.py`)

- **Text & Table Summaries**
  - Dùng Groq LLM (`build_text_summarizer`) để tóm tắt:
    - Mỗi `CompositeElement.text`
    - Mỗi bảng (đã convert sang HTML).
  - Chạy song song:
    - `process_all_content_parallel` → nội bộ gọi `summarize_texts_parallel` với batch, thread pool, và retry.
  - Cache JSON cục bộ:
    - `cache/{file_id}/text_summaries.json`
    - `cache/{file_id}/table_summaries.json`

- **Image Summaries**
  - Dùng `gpt-4o-mini` (`build_vision_summarizer`) trên base64 image.
  - Cũng chạy song song và cache:
    - `cache/{file_id}/image_summaries.json`.


### 5. Ghi vào Vector Store – Abstract & Content Stores

- **Backend trừu tượng hóa (`vectorstore_setup.py`)**
  - Định nghĩa `VectorStoreBackend` (Protocol) với:
    - `add_documents(docs, collection)`
    - `similarity_search(query, k, collection, where=None)`
    - `delete_where(collection, where)`
  - Cài đặt `LocalChromaBackend` dùng Chroma + OpenAI embeddings, mỗi collection là 1 thư mục con:
    - `abstract_store/`
    - `content_store/`

- **Sanitization metadata**
  - Hàm `_sanitize_metadata` trong `langchain_multimodal.py`:
    - Đảm bảo mọi value trong metadata là `str|int|float|bool|None`.
    - List/set/tuple → lọc các phần tử đơn giản rồi join thành string (hoặc `None` nếu rỗng).
    - Loại bỏ nguy cơ lỗi kiểu: “Expected metadata value to be a str, int, float, bool, SparseVector, or None, got list…”.

- **Content Store (`content_store`)**
  - Xóa doc cũ của paper: `delete_where(CONTENT_COLLECTION, {"paper_id": file_id})`.
  - Tạo `Document` cho:
    - **Semantic text nodes**:
      - `page_content = node.text`
      - `metadata = {"paper_id", "section_title", "modality": "text", "order_idx", "page_label"}` (đã sanitize).
    - **Tables**:
      - `page_content = summary_table`
      - `metadata = {"paper_id", "modality": "table", "section_title", "page_label", "table_html"}`.
    - **Images**:
      - `page_content = summary_image`
      - `metadata = {"paper_id", "modality": "image", "section_title": "Figure", "page_label": None, "image_b64"}`.
  - Gọi `backend.add_documents(documents, CONTENT_COLLECTION)` để upsert vào Chroma.

- **Abstract Store (`abstract_store`)**
  - Xóa doc cũ của paper: `delete_where(ABSTRACT_COLLECTION, {"paper_id": file_id})`.
  - Tạo 1 `Document` cho abstract:
    - `page_content = abstract_text` (từ GROBID hoặc từ node đầu tiên nếu thiếu).
    - `metadata = {"paper_id", "title", "authors"}` (qua `_sanitize_metadata`).
  - Gọi `backend.add_documents([abstract_doc], ABSTRACT_COLLECTION)`.


### 6. Retrieval – Document Mode vs Corpus Mode (`rag_pipeline.py`)

- **Document Mode – `document_retrieve` + `build_document_rag`**
  - `document_retrieve(query, paper_id, backend, content_collection)`:
    - Gọi `backend.similarity_search` với `where={"paper_id": paper_id}`.
    - Lấy top-k nodes (text, table, image) của đúng paper.
  - `build_document_rag(...)`:
    - Bọc `document_retrieve` thành một retrieve_fn.
    - Gọi `build_rag_chains(retrieve_fn, PromptConfig(mode="document", paper_id=...))`.

- **Corpus Mode – Abstract-first (`corpus_retrieve` + `build_corpus_rag`)**
  - `condense_query(question)` dùng `gpt-4o-mini` rút gọn câu hỏi thành ≤10 từ (academic search query).
  - Bước 1 – tìm paper-level bằng abstract:
    - `backend.similarity_search(search_query, k_abstracts, collection=abstract_store)`.
    - Lấy set `paper_id` từ các abstract top-k.
  - Bước 2 – tìm content-level trong tập paper đã lọc:
    - `where={"paper_id": {"$in": paper_ids}}` (nếu có paper_ids).
    - `backend.similarity_search(query, k_content, collection=content_store, where=where)`.
  - `build_corpus_rag(...)` dùng `corpus_retrieve` làm retrieve_fn với `PromptConfig(mode="corpus")`.


### 7. Multimodal RAG Prompt & Chains (`rag_pipeline.py`)

- **Split context theo modality**
  - `split_docs(docs)`:
    - Duyệt qua các `Document` đã retrieve, gán `source_id` dạng `[S1]`, `[S2]`…
    - Phân loại theo `metadata.modality`:
      - `"text"` → `texts`
      - `"table"` → `tables`
      - `"image"` → `images`

- **Prompt cấu trúc, có citations**
  - `build_mm_prompt({context, question, prompt_cfg})`:
    - Instructions:
      - “You are the best scientific research assistant. I will tip you 1000 dollars…”
      - “Use ONLY the provided context; if insufficient, say you do not know.”
      - “Keep answers concise (3–5 sentences) and cite sources as [S1], [S2], …”
      - “Ground every claim and highlight contradictions/uncertainty.”
    - Nếu `mode="document"`:
      - Thêm: *“This question is about a single paper (paper_id=...); restrict your answer to this paper unless told otherwise.”*
    - Text/table context:
      - Tạo các dòng dạng: `[S1] Section (p.X): ...` từ `metadata.section_title`, `metadata.page_label`.
    - Image context:
      - Thêm `image_url` (base64) + text caption `[S#] Figure description: ...`.

- **RAG chains**
  - `build_rag_chains(retrieve_fn, prompt_cfg)` tạo:
    - `rag_chain` (trả về **chỉ answer**).
    - `rag_chain_with_ctx` (trả về cả `response` lẫn context đã split) – phục vụ UI highlight.


### 8. FastAPI Endpoints & Mode Awareness (`api.py`)

- **`POST /upload` – Document ingest**
  - Lưu file → tạo `file_id` → queue background ingest → trả về `file_id` + `status="processing"`.
  - Background ingest:
    - Gọi `ingest_document` để cập nhật vector stores & cache.
    - Xây `rag_chain` / `rag_chain_with_ctx` cho Document Mode.

- **`POST /query` – Mode-aware RAG**
  - Request body (`QueryRequest`):
    - `question: str`
    - `file_id: Optional[str]`
    - `mode: Optional["document" | "corpus"]`
    - `include_context: bool`
  - Logic chọn mode:
    - Nếu `mode` NULL:
      - Có `file_id` → mặc định `"document"`.
      - Không có `file_id` → mặc định `"corpus"`.
  - **Document Mode**:
    - Check `processing_status[file_id]` (queued/processing/error).
    - Lấy pipeline từ `pipelines[file_id]["chains"]`.
    - Nếu `include_context=true` → gọi `rag_chain_with_ctx`, trả về `answer` + `context` (dùng được cho highlight SciSpace-like).
  - **Corpus Mode**:
    - Khởi tạo pipeline corpus một lần (`build_corpus_rag`) với `VECTOR_BACKEND` chung.
    - Query tương tự như trên nhưng không ràng buộc `paper_id`.

- **`GET /status` và `/status/{file_id}`**
  - Cho phép FE liệt kê các paper đã ingest, xem `status`, `title`, và kiểm tra `ready/can_query`.


### 9. Tóm tắt

- Pipeline mới kết hợp:
  - **GROBID** (hoặc fallback PyMuPDF) cho text sạch + cấu trúc section.
  - **Unstructured** cho layout + multimodal (tables/images).
  - **Semantic nodes** làm đơn vị retrieval chính cho text.
  - **Abstract/content vector stores** (Chroma cục bộ qua `VectorStoreBackend`) để hỗ trợ abstract-first retrieval trong Corpus Mode.
  - **Multimodal RAG** với prompt chặt, citations [S#], và phân biệt rõ **Document Mode** vs **Corpus Mode**.
- Mọi thứ đều thiết kế sao cho dễ dàng thay Chroma bằng Qdrant (hoặc backend khác) chỉ bằng cách thêm một lớp `VectorStoreBackend` mới, không cần sửa logic RAG. 


