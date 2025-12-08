import os
import re
import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Dict

from lxml import etree


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
    # enriched by attach_layout_to_nodes
    page_number: Optional[int] = None
    bbox: Optional[Dict[str, float]] = None  # raw PDF coords + layout size


# --- New light-weight layout primitives (no unstructured dependency) -----------------


@dataclass
class LayoutBlock:
    """Minimal text block with layout information extracted from PyMuPDF."""
    text: str
    page_number: int
    bbox: Dict[str, float]


@dataclass
class TableBlock:
    """Table extracted by Camelot."""
    html: str
    plaintext: str
    page_number: int
    bbox: Optional[Dict[str, float]] = None


@dataclass
class ImageBlock:
    """Raw image extracted from PyMuPDF."""
    image_b64: str
    page_number: int
    bbox: Optional[Dict[str, float]] = None


# ------------------------------------------------------------------------------------
#                                GROBID / TEI PARSING
# ------------------------------------------------------------------------------------


def parse_grobid_tei(tei_xml: str, pdf_path: str, paper_id: str) -> GrobidText:
    """
    Parse TEI XML returned by GROBID into a GrobidText object.
    This is a simplified parser: enough to get title, authors, abstract, and sections.

    NOTE: We intentionally *do not* rely on TEI page numbers here. All sections are
    created with page_start/page_end = None so that attach_layout_to_nodes is free
    to align them using real PDF layout instead of the placeholder "page 1".
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
        raw = " ".join(a.strip() for a in author_nodes if a.strip())
        authors = [raw]
    else:
        simple_authors = root.xpath(
            "//tei:teiHeader//tei:titleStmt//tei:author//text()", namespaces=ns
        )
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

        # paragraphs inside this <div>
        p_elements = div.xpath(".//tei:p", namespaces=ns)
        paragraphs: List[str] = []
        for p in p_elements:
            p_text_nodes = p.xpath(".//text()", namespaces=ns)
            p_text = " ".join(t.strip() for t in p_text_nodes if t.strip())
            if p_text:
                paragraphs.append(p_text)

        if not paragraphs:
            continue

        sec_text = "\n\n".join(paragraphs)

        sections.append(
            GrobidSection(
                title=sec_title,
                text=sec_text,
                order=order_idx,
                page_start=None,  # we let layout matching decide later
                page_end=None,
            )
        )
        order_idx += 1

    # Fallback: if no explicit sections, use full body
    if not sections:
        body_nodes = root.xpath("//tei:text//tei:body//text()", namespaces=ns)
        body_text = " ".join(t.strip() for t in body_nodes if t.strip())
        if body_text:
            sections.append(
                GrobidSection(
                    title="Body",
                    text=body_text,
                    order=0,
                    page_start=None,
                    page_end=None,
                )
            )

    # Fallback abstract if empty
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

            grobid_text = parse_grobid_tei(tei_xml, pdf_path, paper_id)
            if grobid_text.sections:
                return grobid_text
            else:
                print("⚠️ GROBID TEI parsed but no sections found; falling back to PyMuPDF parser.")
        except Exception as exc:  # pragma: no cover - network path best effort
            print(f"⚠️ GROBID call or TEI parsing failed ({exc}); falling back to lightweight parser.")

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


# ------------------------------------------------------------------------------------
#                           SEMANTIC CHUNKING FROM GROBID
# ------------------------------------------------------------------------------------


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
        raw_paras = re.split(r"\n\s*\n", text)
        paras = [p.strip() for p in raw_paras if p.strip()]
        return paras

    def split_into_sentences(text: str) -> List[str]:
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

        # merge very short paragraphs with previous one
        merged_paras: List[str] = []
        for para in paragraphs:
            if not merged_paras:
                merged_paras.append(para)
                continue

            prev = merged_paras[-1]
            if estimate_tokens(prev) < para_min_tokens and estimate_tokens(para) < para_min_tokens:
                merged_paras[-1] = prev + " " + para
            else:
                merged_paras.append(para)

        for para in merged_paras:
            para_tokens = estimate_tokens(para)

            # 1) short/medium paragraph → single node
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

            # 2) long paragraph → semantic split by sentence
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


# ------------------------------------------------------------------------------------
#                          LAYOUT EXTRACTION (PyMuPDF / Camelot)
# ------------------------------------------------------------------------------------


def extract_layout_blocks(pdf_path: str) -> List[LayoutBlock]:
    """
    Extract low-level text blocks and their bounding boxes from the PDF using PyMuPDF.

    This replaces the previous unstructured.partition.pdf based pipeline and is
    much lighter while still giving us per-block coordinates.

    Coordinates are returned in absolute PDF units together with layout_width /
    layout_height so the frontend can normalize to [0, 1] as it wishes.
    """
    try:
        import fitz  # type: ignore
    except Exception as exc:  # pragma: no cover - environment issue
        print(f"⚠️ PyMuPDF not available for layout extraction ({exc}); no layout blocks.")
        return []

    blocks: List[LayoutBlock] = []
    doc = fitz.open(pdf_path)
    for page_index, page in enumerate(doc):
        page_number = page_index + 1
        width, height = page.rect.width, page.rect.height

        for block in page.get_text("blocks"):  # (x0, y0, x1, y1, text, ...)
            if len(block) < 5:
                continue
            x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
            if not text:
                continue
            text_clean = text.strip()
            if not text_clean:
                continue

            bbox = {
                "x1": float(x0),
                "y1": float(y0),
                "x2": float(x1),
                "y2": float(y1),
                "layout_width": float(width),
                "layout_height": float(height),
            }
            # fix y2 value
            bbox["y2"] = float(y1 if False else y1)  # will be overwritten below
            bbox["y2"] = float(y1)

            blocks.append(
                LayoutBlock(
                    text=text_clean,
                    page_number=page_number,
                    bbox=bbox,
                )
            )

    # correct typo: y2 must be using original y1/y2
    for blk in blocks:
        # in case we ever need to adjust later; kept for safety
        pass

    return blocks


def extract_table_blocks(pdf_path: str) -> List[TableBlock]:
    """
    Extract tables using Camelot, if available.

    Even without LLM summarization we can still turn the table into a textual
    representation (plain text + HTML) so that it can participate in text-only
    retrieval.
    """
    try:
        import camelot  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"ℹ️ Camelot not installed or failed to import ({exc}); skipping table extraction.")
        return []

    try:
        tables = camelot.read_pdf(
            pdf_path,
            pages="all",
            flavor="lattice",   # good default when PDFs have ruling lines
            strip_text="\n",
        )
    except Exception as exc:  # pragma: no cover - parsing issues
        print(f"⚠️ Camelot failed to parse tables ({exc}); skipping.")
        return []

    blocks: List[TableBlock] = []
    for t in tables:
        try:
            page_number = int(getattr(t, "page", None) or t.parsing_report.get("page", 1))
        except Exception:
            page_number = 1

        html = t.df.to_html(index=False, border=0)
        plaintext = t.df.to_string(index=False)

        bbox: Optional[Dict[str, float]] = None
        try:
            if hasattr(t, "_bbox") and t._bbox is not None:
                x1, y1, x2, y2 = t._bbox
                bbox = {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                }
        except Exception:
            bbox = None

        blocks.append(
            TableBlock(
                html=html,
                plaintext=plaintext,
                page_number=page_number,
                bbox=bbox,
            )
        )

    return blocks


def extract_image_blocks(pdf_path: str) -> List[ImageBlock]:
    """
    Extract images as base64 + bbox using PyMuPDF.

    We still don't *summarize* these images here — they are mainly for front-end
    display and for the /explain-region multimodal endpoint.
    """
    try:
        import fitz  # type: ignore
    except Exception as exc:  # pragma: no cover
        print(f"⚠️ PyMuPDF not available for image extraction ({exc}); no images.")
        return []

    blocks: List[ImageBlock] = []
    doc = fitz.open(pdf_path)
    for page_index, page in enumerate(doc):
        page_number = page_index + 1
        width, height = page.rect.width, page.rect.height

        for img in page.get_images(full=True):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n >= 5:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_bytes = pix.tobytes("png")
                img_b64 = base64.b64encode(img_bytes).decode("ascii")

                rects = page.get_image_rects(xref)
                bbox: Optional[Dict[str, float]] = None
                if rects:
                    r = rects[0]
                    bbox = {
                        "x1": float(r.x0),
                        "y1": float(r.y0),
                        "x2": float(r.x1),
                        "y2": float(r.y1),
                        "layout_width": float(width),
                        "layout_height": float(height),
                    }

                blocks.append(
                    ImageBlock(
                        image_b64=img_b64,
                        page_number=page_number,
                        bbox=bbox,
                    )
                )
            except Exception as exc:
                print(f"⚠️ Failed to extract image on page {page_number}: {exc}")

    return blocks


# ------------------------------------------------------------------------------------
#                     ALIGN LAYOUT BLOCKS -> SEMANTIC NODES (for highlight)
# ------------------------------------------------------------------------------------


def attach_layout_to_nodes(
    nodes: List[SemanticNode],
    texts: List[Any],
) -> List[SemanticNode]:
    """
    Attach layout info (page_number + bbox) from low-level text blocks to each
    SemanticNode.

    - When texts is a list of LayoutBlock (our new PyMuPDF extractor), we use
      its page_number + bbox directly.
    - For backwards compatibility, we still support objects having a .metadata
      with page_number / coordinates (e.g. unstructured elements), but we no
      longer *depend* on unstructured being installed.
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
        if isinstance(el, LayoutBlock):
            return el.page_number, el.bbox

        meta = getattr(el, "metadata", None)
        if meta is None:
            return None, None

        page = getattr(meta, "page_number", None) or getattr(meta, "page", None)

        coords = getattr(meta, "coordinates", None)
        if not coords:
            return page, None

        bbox_dict: Dict[str, float] = {}

        try:
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

    if not blocks:
        return nodes

    for node in nodes:
        node_words = _norm(node.text)
        if not node_words:
            continue

        best_score = 0.0
        best_block: Optional[Dict[str, Any]] = None

        for blk in blocks:
            if node.approx_page_start and blk["page"]:
                if abs(blk["page"] - node.approx_page_start) > 2:
                    continue

            score = _overlap_score(node_words, blk["words"])
            if score > best_score:
                best_score = score
                best_block = blk

        if best_block and best_score > 0.2:
            node.page_number = best_block["page"] or node.approx_page_start
            if best_block["bbox"]:
                node.bbox = best_block["bbox"]

    return nodes
