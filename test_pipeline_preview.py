from typing import Any, List

from config import PDF_PATH
from pdf_extract import (
    partition_pdf_into_chunks,
    split_tables_and_texts,
    remove_repeated_headers,
    get_images_base64,
)


def _preview_text(text: str, max_len: int = 1000) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    return text[:max_len] + ("…" if len(text) > max_len else "")


def _type_name(obj: Any) -> str:
    return type(obj).__name__


def main():
    print(f"Using PDF: {PDF_PATH}")

    # Step 1: Partition into chunks
    chunks: List[Any] = partition_pdf_into_chunks(PDF_PATH)
    print(f"\nTotal chunks: {len(chunks)}")
    for i, ch in enumerate(chunks[:5]):
        txt = getattr(ch, "text", None)
        print(f"  [{i}] {_type_name(ch)} | text={bool(txt)} | preview=\"{_preview_text(txt or '')}\"")
    if len(chunks) > 5:
        print(f"  … and {len(chunks) - 5} more")

    # Step 2: Split tables and texts
    tables, texts = split_tables_and_texts(chunks)
    print(f"\nTables: {len(tables)} | Texts (raw): {len(texts)}")

    # Step 3: Remove repeated headers from texts
    texts_clean = remove_repeated_headers(texts)
    print(f"Texts (cleaned): {len(texts_clean)}")
    for i, t in enumerate(texts_clean[:5]):
        print(f"  [T{i}] {_type_name(t)} | preview=\"{_preview_text(getattr(t, 'text', '') or '')}\"")
    if len(texts_clean) > 5:
        print(f"  … and {len(texts_clean) - 5} more")

    # Step 4: Extract images as base64
    images = get_images_base64(chunks)
    print(f"\nImages (base64): {len(images)}")
    for i, b64 in enumerate(images[:5]):
        print(f"  [I{i}] len={len(b64)} | head={b64[:24]}…")
    if len(images) > 5:
        print(f"  … and {len(images) - 5} more")


if __name__ == "__main__":
    main()


