from collections import Counter
from typing import Any, List, Tuple

from unstructured.partition.pdf import partition_pdf

from config import PDF_PATH


def partition_pdf_into_chunks(pdf_path: str = PDF_PATH):
    return partition_pdf(
        filename=pdf_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )


def split_tables_and_texts(chunks: List[Any]) -> Tuple[List[Any], List[Any]]:
    tables, texts = [], []
    for ch in chunks:
        tname = str(type(ch))
        if "Table" in tname:
            tables.append(ch)
        if "CompositeElement" in tname:
            texts.append(ch)
    return tables, texts


def remove_repeated_headers(texts: List[Any]) -> List[Any]:
    text_blocks = [el.text.strip() for el in texts if el.text and el.text.strip()]
    short_lines = [t for t in text_blocks if len(t) <= 50]
    freq = Counter(short_lines)
    repeated_headers = {line for line, c in freq.items() if c >= 3}

    cleaned_texts = []
    for el in texts:
        text = (el.text or "").strip()
        if text and text not in repeated_headers:
            cleaned_texts.append(el)
    return cleaned_texts


def get_images_base64(all_chunks: List[Any]) -> List[str]:
    images_b64: List[str] = []
    for ch in all_chunks:
        if "CompositeElement" in str(type(ch)):
            for el in ch.metadata.orig_elements:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64


