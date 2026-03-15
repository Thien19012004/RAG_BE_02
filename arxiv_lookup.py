"""
ArXiv / DOI lookup for paper classification.

3-tier strategy with 10s total timeout:
1. DOI → CrossRef API (2s) → categories
2. Title → ArXiv API (5s) → categories
3. Fallback → LLM classify (handled in kb_classifier.py)
"""
import asyncio
import time
from typing import Any, Dict, List, Optional

import httpx


# ArXiv category → our system category slug mapping
ARXIV_TO_SYSTEM = {
    "cs.AI": "artificial_intelligence",
    "cs.LG": "machine_learning",
    "cs.CL": "nlp",
    "cs.CV": "computer_vision",
    "cs.SE": "software_engineering",
    "cs.DB": "databases",
    "cs.DS": "algorithms",
    "cs.CR": "security",
    "cs.NI": "networking",
    "cs.DC": "distributed_systems",
    "cs.HC": "hci",
    "stat.ML": "machine_learning",
    "cs.IR": "databases",
    "cs.NE": "deep_learning",
    "cs.RO": "artificial_intelligence",
    "cs.MA": "artificial_intelligence",
}


async def lookup_doi(doi: str, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
    """Lookup paper metadata via CrossRef API using DOI."""
    if not doi:
        return None
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            url = f"https://api.crossref.org/works/{doi}"
            resp = await client.get(url, headers={"Accept": "application/json"})
            if resp.status_code != 200:
                return None
            data = resp.json().get("message", {})
            # CrossRef doesn't have CS categories directly, but we can get subject
            subjects = data.get("subject", [])
            title_list = data.get("title", [])
            return {
                "source": "crossref",
                "title": title_list[0] if title_list else None,
                "subjects": subjects,
                "doi": doi,
            }
    except Exception as e:
        print(f"[arxiv_lookup] DOI lookup failed: {e}")
        return None


async def lookup_arxiv(title: str, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
    """Search ArXiv API by title and return categories."""
    if not title or len(title) < 5:
        return None
    try:
        # Clean title for search
        search_query = title.replace(":", " ").replace("-", " ").strip()
        async with httpx.AsyncClient(timeout=timeout) as client:
            url = "https://export.arxiv.org/api/query"

            # Try exact phrase first, then fallback to keyword search
            for query_fmt in [f'ti:"{search_query}"', f'ti:{"+".join(search_query.split()[:6])}']: 
                params = {
                    "search_query": query_fmt,
                    "max_results": 5,
                }
                print(f"[arxiv_lookup] Searching: {query_fmt}")
                resp = await client.get(url, params=params)
                if resp.status_code != 200:
                    print(f"[arxiv_lookup] HTTP {resp.status_code}")
                    continue

                result = _parse_arxiv_response(resp.text, title)
                if result:
                    return result
                print(f"[arxiv_lookup] No match with query: {query_fmt}")

            return None
    except Exception as e:
        print(f"[arxiv_lookup] ArXiv search failed: {e}")
        return None


def _parse_arxiv_response(xml_text: str, original_title: str) -> Optional[Dict[str, Any]]:
    """Parse ArXiv Atom XML and extract categories."""
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml_text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        entries = root.findall("atom:entry", ns)
        print(f"[arxiv_lookup] Found {len(entries)} entries")
        if not entries:
            return None

        # Find best matching entry by title similarity
        best_entry = None
        best_score = 0.0
        for entry in entries:
            title_el = entry.find("atom:title", ns)
            if title_el is not None and title_el.text:
                entry_title = " ".join(title_el.text.strip().split())
                score = _title_similarity(original_title, entry_title)
                print(f"[arxiv_lookup]   '{entry_title[:60]}...' score={score:.2f}")
                if score > best_score:
                    best_score = score
                    best_entry = entry

        if best_entry is None or best_score < 0.3:
            print(f"[arxiv_lookup] Best score {best_score:.2f} < 0.3, no match")
            return None

        # Extract categories — iterate ALL child elements, match by tag name
        categories = []
        for child in best_entry:
            tag_local = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if tag_local == "category":
                term = child.get("term", "")
                if term:
                    categories.append(term)

        print(f"[arxiv_lookup] Categories found: {categories}")

        title_el = best_entry.find("atom:title", ns)
        arxiv_title = " ".join(title_el.text.strip().split()) if title_el is not None and title_el.text else ""

        return {
            "source": "arxiv",
            "title": arxiv_title,
            "arxiv_categories": categories,
            "match_score": best_score,
        }
    except Exception as e:
        print(f"[arxiv_lookup] XML parse error: {e}")
        return None


def _title_similarity(a: str, b: str) -> float:
    """Simple word-overlap Jaccard similarity for title matching."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def map_arxiv_categories(arxiv_cats: List[str]) -> List[Dict[str, Any]]:
    """Map ArXiv category codes to our system category slugs with confidence."""
    mapped = {}
    for cat in arxiv_cats:
        slug = ARXIV_TO_SYSTEM.get(cat)
        if slug and slug not in mapped:
            # Primary category gets higher confidence
            confidence = 0.95 if cat == arxiv_cats[0] else 0.75
            mapped[slug] = {"slug": slug, "confidence": confidence}
    return list(mapped.values())


async def classify_with_lookup(
    title: str,
    abstract: str,
    doi: Optional[str] = None,
    total_timeout: float = 10.0,
) -> Dict[str, Any]:
    """
    3-tier classification strategy with total timeout.

    1. DOI → CrossRef (2s)
    2. Title → ArXiv API (5s)
    3. LLM fallback (remaining time)

    Returns:
        {
            "categories": [{"slug": "...", "confidence": 0.0-1.0}],
            "tags": [...],
            "source": "arxiv" | "crossref" | "llm"
        }
    """
    start = time.time()

    # Tier 1: DOI lookup
    if doi:
        result = await lookup_doi(doi, timeout=2.0)
        if result and result.get("subjects"):
            elapsed = time.time() - start
            print(f"[classify] DOI hit in {elapsed:.1f}s — subjects: {result['subjects']}")
            # CrossRef subjects aren't great for CS — continue to ArXiv

    # Tier 2: ArXiv title search
    remaining = total_timeout - (time.time() - start)
    if remaining > 1.0:
        arxiv_timeout = min(5.0, remaining - 1.0)  # Leave 1s buffer
        result = await lookup_arxiv(title, timeout=arxiv_timeout)
        if result and result.get("arxiv_categories"):
            categories = map_arxiv_categories(result["arxiv_categories"])
            if categories:
                elapsed = time.time() - start
                print(f"[classify] ArXiv hit in {elapsed:.1f}s — cats: {result['arxiv_categories']}")
                return {
                    "categories": categories,
                    "tags": [],  # ArXiv doesn't give tags, LLM can add them
                    "source": "arxiv",
                    "arxiv_categories": result["arxiv_categories"],
                }

    # Tier 3: LLM fallback
    elapsed = time.time() - start
    print(f"[classify] Lookup took {elapsed:.1f}s, falling back to LLM")
    return None  # Signal caller to use LLM
