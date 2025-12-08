import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Dict
from lxml import etree

from unstructured.partition.pdf import partition_pdf


@dataclass
class GrobidSection:
    title: str
    text: str
    order: int
    page_start: Optional[int] = None
    page_end: Optional[int] = None


@dataclass
class GrobidText:
    paper_id: str
    title: str
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    sections: List[GrobidSection] = field(default_factory=list)


@dataclass
class SemanticNode:
    paper_id: str
    section_title: str
    order_idx: int
    text: str
    approx_page_start: Optional[int] = None
    approx_page_end: Optional[int] = None
    # mới thêm
    page_number: Optional[int] = None
    bbox: Optional[Dict[str, float]] = None  # raw PDF coords + layout size


def partition_pdf_into_chunks(pdf_path: str):
    """Partition PDF into chunks using unstructured with optimized settings"""
    return partition_pdf(
        filename=pdf_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=8000,  # Reduced from 10000 for faster processing
        combine_text_under_n_chars=1500,  # Reduced from 2000
        new_after_n_chars=4000,  # Reduced from 6000
    )


def split_tables_and_texts(chunks: List[Any]) -> Tuple[List[Any], List[Any]]:
    """Split chunks into tables and texts"""
    tables, texts = [], []
    for ch in chunks:
        tname = str(type(ch))
        if "Table" in tname:
            tables.append(ch)
        if "CompositeElement" in tname:
            texts.append(ch)
    return tables, texts


def remove_repeated_headers(texts: List[Any]) -> List[Any]:
    """Remove repeated headers from text chunks"""
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
    """Extract base64 encoded images from chunks"""
    images_b64: List[str] = []
    for ch in all_chunks:
        if "CompositeElement" in str(type(ch)):
            for el in ch.metadata.orig_elements:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64


def parse_grobid_tei(tei_xml: str, pdf_path: str, paper_id: str) -> GrobidText:
    """
    Parse TEI XML returned by GROBID into a GrobidText object.
    This is a simplified parser: enough to get title, authors, abstract, and sections.
    """
    parser = etree.XMLParser(recover=True)
    root = etree.fromstring(tei_xml.encode("utf-8"), parser=parser)

    ns = {"tei": "http://www.tei-c.org/ns/1.0"}

    # -------- Title --------
    title_nodes = root.xpath("//tei:teiHeader//tei:titleStmt//tei:title/text()", namespaces=ns)
    title = title_nodes[0].strip() if title_nodes else Path(pdf_path).stem

    # -------- Authors --------
    author_nodes = root.xpath(
        "//tei:teiHeader//tei:titleStmt//tei:author//tei:persName//text()",
        namespaces=ns,
    )
    authors: List[str] = []
    if author_nodes:
        # Join individual name tokens & deduplicate
        raw = " ".join(a.strip() for a in author_nodes if a.strip())
        # Rough split by '  ' or ';' or ',' if needed; here we keep as single string
        authors = [raw]
    else:
        # Fallback: try simpler author extraction
        simple_authors = root.xpath("//tei:teiHeader//tei:titleStmt//tei:author//text()", namespaces=ns)
        if simple_authors:
            authors = [" ".join(a.strip() for a in simple_authors if a.strip())]

    # -------- Abstract --------
    abstract_nodes = root.xpath("//tei:profileDesc//tei:abstract//text()", namespaces=ns)
    abstract = " ".join(t.strip() for t in abstract_nodes if t.strip())

    # -------- Sections / Body --------
    sections: List[GrobidSection] = []

    divs = root.xpath("//tei:text//tei:body//tei:div", namespaces=ns)

    order_idx = 0
    for idx, div in enumerate(divs):
        head_nodes = div.xpath(".//tei:head//text()", namespaces=ns)
        sec_title = " ".join(h.strip() for h in head_nodes if h.strip()) or f"Section {idx+1}"

        # 💡 GIỮ PARAGRAPH RIÊNG LẺ
        p_elements = div.xpath(".//tei:p", namespaces=ns)
        paragraphs: List[str] = []
        for p in p_elements:
            p_text_nodes = p.xpath(".//text()", namespaces=ns)
            p_text = " ".join(t.strip() for t in p_text_nodes if t.strip())
            if p_text:
                paragraphs.append(p_text)

        if not paragraphs:
            continue

        # Lưu toàn bộ section.text, nhưng với delimiter giữa paragraphs
        sec_text = "\n\n".join(paragraphs)

        page_start = 1
        page_end = 1

        sections.append(
            GrobidSection(
                title=sec_title,
                text=sec_text,
                order=order_idx,
                page_start=page_start,
                page_end=page_end,
            )
        )
        order_idx += 1


    # Fallback: nếu không parse được section nào, vẫn dùng 1 section full text
    if not sections:
        body_nodes = root.xpath("//tei:text//tei:body//text()", namespaces=ns)
        body_text = " ".join(t.strip() for t in body_nodes if t.strip())
        if body_text:
            sections.append(
                GrobidSection(
                    title="Body",
                    text=body_text,
                    order=0,
                    page_start=1,
                    page_end=1,
                )
            )

    # Fallback abstract nếu rỗng: dùng đoạn đầu section đầu tiên
    if not abstract and sections:
        abstract = sections[0].text[:500]

    return GrobidText(
        paper_id=paper_id,
        title=title,
        authors=authors,
        abstract=abstract,
        sections=sections,
    )


def run_grobid(pdf_path: str, paper_id: str) -> GrobidText:
    """
    Run GROBID extraction if available; otherwise fall back to a lightweight PyMuPDF parser.
    """
    grobid_url = os.getenv("GROBID_URL")
    if grobid_url:
        try:
            import requests  # type: ignore

            with open(pdf_path, "rb") as f:
                files = {"input": f}
                resp = requests.post(
                    f"{grobid_url}/api/processFulltextDocument",
                    files=files,
                    timeout=60,
                )
            resp.raise_for_status()
            tei_xml = resp.text

            # ✅ Bây giờ chúng ta thực sự parse TEI
            grobid_text = parse_grobid_tei(tei_xml, pdf_path, paper_id)
            # Nếu parse ra mà sections rỗng → fallback tiếp cho an toàn
            if grobid_text.sections:
                return grobid_text
            else:
                print("⚠️ GROBID TEI parsed but no sections found; falling back to PyMuPDF parser.")
        except Exception as exc:  # pragma: no cover - network path best effort
            print(f"⚠️ GROBID call or TEI parsing failed ({exc}); falling back to lightweight parser.")

    # Fallback path: PyMuPDF-based parser
    return _fallback_parse(pdf_path, paper_id)


def _fallback_parse(pdf_path: str, paper_id: str) -> GrobidText:
    """Very lightweight PDF -> section text parser used when GROBID is unavailable."""
    try:
        import fitz  # type: ignore

        doc = fitz.open(pdf_path)
        sections: List[GrobidSection] = []
        for idx, page in enumerate(doc):
            text = page.get_text().strip()
            if not text:
                continue
            sections.append(
                GrobidSection(
                    title=f"Page {idx + 1}",
                    text=text,
                    order=idx,
                    page_start=idx + 1,
                    page_end=idx + 1,
                )
            )
        return GrobidText(
            paper_id=paper_id,
            title=Path(pdf_path).stem,
            authors=[],
            abstract=sections[0].text[:500] if sections else "",
            sections=sections,
        )
    except Exception as exc:
        print(f"⚠️ PyMuPDF fallback failed ({exc}); using empty sections.")
        return GrobidText(
            paper_id=paper_id,
            title=Path(pdf_path).stem,
            authors=[],
            abstract="",
            sections=[],
        )


def build_semantic_nodes(
    paper_id: str,
    sections: List[GrobidSection],
    embed_fn: Callable[[List[str]], List[Sequence[float]]],
    similarity_threshold: float = 0.6,
    max_tokens: int = 400,
    para_max_tokens: int = 280,        # ngưỡng cắt paragraph
    para_min_tokens: int = 40,         # paragraph quá nhỏ sẽ được merge
) -> List[SemanticNode]:
    """
    Paragraph-first, semantic-second:
    - Mỗi <p> trong TEI ~ 1 paragraph gốc.
    - Paragraph ngắn/vừa => 1 node.
    - Paragraph dài => semantic split theo câu.
    """

    def split_into_paragraphs(text: str) -> List[str]:
        # text đã có '\n\n' giữa các paragraph (từ parse_grobid_tei)
        raw_paras = re.split(r"\n\s*\n", text)
        paras = [p.strip() for p in raw_paras if p.strip()]
        return paras

    def split_into_sentences(text: str) -> List[str]:
        # giữ xuống dòng bên trong paragraph (nếu còn), chỉ normalize nhẹ
        text_norm = text.replace("\r", " ").strip()
        if not text_norm:
            return []
        sentences = re.split(r"(?<=[.!?])\s+", text_norm)
        return [s.strip() for s in sentences if s.strip()]

    def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        import math
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a)) or 1e-9
        norm_b = math.sqrt(sum(b * b for b in vec_b)) or 1e-9
        return dot / (norm_a * norm_b)

    def average_vector(vectors: List[Sequence[float]]) -> List[float]:
        if not vectors:
            return []
        dim = len(vectors[0])
        avg = [0.0] * dim
        for vec in vectors:
            for i, val in enumerate(vec):
                avg[i] += val
        return [val / len(vectors) for val in avg]

    def estimate_tokens(text: str) -> int:
        return max(1, int(len(text.split()) * 1.3))

    nodes: List[SemanticNode] = []
    node_counter = 0

    for section in sections:
        paragraphs = split_into_paragraphs(section.text)
        if not paragraphs:
            continue

        # Optional: merge very short paragraphs với paragraph trước đó
        merged_paras: List[str] = []
        for para in paragraphs:
            if not merged_paras:
                merged_paras.append(para)
                continue

            prev = merged_paras[-1]
            if estimate_tokens(prev) < para_min_tokens and estimate_tokens(para) < para_min_tokens:
                # gộp 2 paragraph rất ngắn
                merged_paras[-1] = prev + " " + para
            else:
                merged_paras.append(para)

        for para in merged_paras:
            para_tokens = estimate_tokens(para)

            # 1) Paragraph vừa/nhỏ -> 1 node = cả đoạn
            if para_tokens <= para_max_tokens:
                nodes.append(
                    SemanticNode(
                        paper_id=paper_id,
                        section_title=section.title or "Untitled Section",
                        order_idx=node_counter,
                        text=para,
                        approx_page_start=section.page_start,
                        approx_page_end=section.page_end,
                    )
                )
                node_counter += 1
                continue

            # 2) Paragraph dài -> semantic split theo câu (giống logic cũ, nhưng confined trong paragraph)
            sentences = split_into_sentences(para)
            if not sentences:
                continue

            embeddings = embed_fn(sentences)
            current_sentences: List[str] = []
            current_vecs: List[Sequence[float]] = []
            current_token_count = 0

            for sent, vec in zip(sentences, embeddings):
                sent_tokens = estimate_tokens(sent)

                if not current_sentences:
                    current_sentences = [sent]
                    current_vecs = [vec]
                    current_token_count = sent_tokens
                    continue

                centroid = average_vector(current_vecs)
                similarity = cosine_similarity(vec, centroid)
                length_exceeded = current_token_count + sent_tokens > max_tokens

                # chỉ tách khi:
                # - quá dài, hoặc
                # - similarity thấp *và* node đã đủ dài (>= 2 câu)
                should_split = False
                if length_exceeded:
                    should_split = True
                elif similarity < similarity_threshold and len(current_sentences) >= 2:
                    should_split = True

                if should_split:
                    nodes.append(
                        SemanticNode(
                            paper_id=paper_id,
                            section_title=section.title or "Untitled Section",
                            order_idx=node_counter,
                            text=" ".join(current_sentences),
                            approx_page_start=section.page_start,
                            approx_page_end=section.page_end,
                        )
                    )
                    node_counter += 1
                    current_sentences = [sent]
                    current_vecs = [vec]
                    current_token_count = sent_tokens
                else:
                    current_sentences.append(sent)
                    current_vecs.append(vec)
                    current_token_count += sent_tokens

            if current_sentences:
                nodes.append(
                    SemanticNode(
                        paper_id=paper_id,
                        section_title=section.title or "Untitled Section",
                        order_idx=node_counter,
                        text=" ".join(current_sentences),
                        approx_page_start=section.page_start,
                        approx_page_end=section.page_end,
                    )
                )
                node_counter += 1

    return nodes



def attach_layout_to_nodes(
    nodes: List[SemanticNode],
    texts: List[Any],
) -> List[SemanticNode]:
    """
    Gắn thông tin layout (page_number + bbox) từ các text chunk của unstructured
    vào từng SemanticNode.

    - page_number: số trang từ unstructured (page_number)
    - bbox: dict raw toạ độ PDF + kích thước layout để frontend tự convert.
    """

    def _norm(s: str) -> List[str]:
        s = re.sub(r"\s+", " ", s.strip().lower())
        return [w for w in s.split(" ") if w]

    def _overlap_score(a_words: List[str], b_words: List[str]) -> float:
        if not a_words or not b_words:
            return 0.0
        a_set, b_set = set(a_words), set(b_words)
        inter = len(a_set & b_set)
        return inter / max(1, len(a_set))

    def _extract_page_and_bbox(el: Any) -> Tuple[Optional[int], Optional[Dict[str, float]]]:
        meta = getattr(el, "metadata", None)
        if meta is None:
            return None, None

        page = getattr(meta, "page_number", None) or getattr(meta, "page", None)

        coords = getattr(meta, "coordinates", None)
        if not coords:
            return page, None

        bbox_dict: Dict[str, float] = {}

        try:
            # unstructured thường là dataclass có .bounding_box và layout_width/height
            if isinstance(coords, dict):
                bb = coords.get("bounding_box") or coords
                x1 = bb.get("x1")
                y1 = bb.get("y1")
                x2 = bb.get("x2")
                y2 = bb.get("y2")
                layout_w = coords.get("layout_width")
                layout_h = coords.get("layout_height")
            else:
                bb = getattr(coords, "bounding_box", None) or coords
                x1 = getattr(bb, "x1", None)
                y1 = getattr(bb, "y1", None)
                x2 = getattr(bb, "x2", None)
                y2 = getattr(bb, "y2", None)
                layout_w = getattr(coords, "layout_width", None)
                layout_h = getattr(coords, "layout_height", None)

            if None in (x1, y1, x2, y2):
                return page, None

            bbox_dict = {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            }
            if layout_w is not None and layout_h is not None:
                bbox_dict["layout_width"] = float(layout_w)
                bbox_dict["layout_height"] = float(layout_h)
        except Exception:
            return page, None

        return page, bbox_dict or None

    # Chuẩn bị candidate blocks
    blocks = []
    for el in texts:
        raw_text = getattr(el, "text", "") or ""
        raw_text = raw_text.strip()
        if not raw_text:
            continue
        words = _norm(raw_text)
        if not words:
            continue

        page, bbox = _extract_page_and_bbox(el)
        blocks.append(
            {
                "page": page,
                "words": words,
                "bbox": bbox,
            }
        )

    # Align từng node với block tốt nhất (theo word-overlap + gần page)
    for node in nodes:
        node_words = _norm(node.text)
        if not node_words or not blocks:
            continue

        best_score = 0.0
        best_block = None

        for blk in blocks:
            # ưu tiên các block gần approx_page_start
            if node.approx_page_start and blk["page"]:
                if abs(blk["page"] - node.approx_page_start) > 2:
                    continue

            score = _overlap_score(node_words, blk["words"])
            if score > best_score:
                best_score = score
                best_block = blk

        # đặt threshold nhẹ để tránh gán linh tinh
        if best_block and best_score > 0.2:
            node.page_number = best_block["page"] or node.approx_page_start
            if best_block["bbox"]:
                node.bbox = best_block["bbox"]

    return nodes
