# -*- coding: utf-8 -*-
"""
ArXiv related-paper suggestion using:
- LLM for Query Generation (Agentic/MCP style).
- 'arxiv' Python Library (Standard Wrapper) for data fetching.
- LLM for Contextual Re-ranking.
"""

from __future__ import annotations

import datetime as _dt
import json
import arxiv
from dataclasses import dataclass, asdict
from typing import List, Optional
from difflib import SequenceMatcher

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------

@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    categories: List[str]
    published: str  # ISO date string
    url: str

    @property
    def year(self) -> Optional[int]:
        try:
            return int(self.published[:4])
        except Exception:
            return None

    def to_public_dict(self) -> dict:
        d = asdict(self)
        d["title"] = " ".join(d["title"].split())
        d["abstract"] = " ".join(d["abstract"].split())
        return d


class ArxivSearchQuery(BaseModel):
    primary_query: str = Field(
        ...,
        description="The optimized search query. prioritizing BROAD CONCEPTS over specific project names."
    )
    explanation: str = Field(..., description="Reasoning.")


# -----------------------------------------------------------------------------
# 1. LLM Query Generation
# -----------------------------------------------------------------------------

def generate_search_query_with_llm(
    base_title: str,
    base_abstract: str,
    model_name: str = "gpt-4o-mini"
) -> str:
    if not base_title and not base_abstract:
        return 'all:rag OR all:"retrieval augmented"'

    llm = ChatOpenAI(model=model_name, temperature=0.2) # Tăng temp nhẹ để sáng tạo hơn
    structured_llm = llm.with_structured_output(ArxivSearchQuery)

    # --- Prompt yêu cầu tìm kiếm rộng hơn ---
    system_prompt = """You are an expert research assistant.
Generate a SEARCH QUERY to find RELATED papers for the user's input paper.

RULES:
1. REMOVE metadata (e.g., "Under review", "Accepted").
2. **CRITICAL**: If the paper has a specific project name (e.g., "PaperQA", "Llama 2"), DO NOT just search for that name. Combine it with general technical terms using OR.
   - BAD: `all:PaperQA` (Only finds the paper itself)
   - GOOD: `all:PaperQA OR (all:"Retrieval Augmented Generation" AND all:Scientific)`
3. Use ArXiv syntax: `all:"phrase"`, `all:keyword`, `AND`, `OR`.
4. Max 7 terms.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "TITLE:\n{title}\n\nABSTRACT:\n{abstract}")
    ])

    try:
        result = (prompt | structured_llm).invoke({"title": base_title, "abstract": base_abstract})
        print(f"DEBUG: Generated Query: '{result.primary_query}'")
        return result.primary_query
    except Exception as e:
        print(f"ERROR: LLM Query Gen failed: {e}")
        return f'all:{" ".join(base_title.split()[:5])}'


# -----------------------------------------------------------------------------
# 2. ArXiv Tool Execution
# -----------------------------------------------------------------------------

def search_arxiv_raw(query: str, max_results: int = 30) -> List[ArxivPaper]:
    print(f"DEBUG: Executing ArXiv Client Search with query: {query}")
    client = arxiv.Client(
        page_size=max_results,
        delay_seconds=3.0,
        num_retries=3
    )

    # Tìm kiếm
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )

    entries: List[ArxivPaper] = []
    try:
        for result in client.results(search):
            paper = ArxivPaper(
                arxiv_id=result.entry_id.split("/")[-1],
                title=result.title,
                abstract=result.summary,
                authors=[a.name for a in result.authors],
                categories=result.categories,
                published=result.published.strftime("%Y-%m-%d"),
                url=result.entry_id,
            )
            entries.append(paper)
    except Exception as e:
        print(f"ERROR: ArXiv Client failed: {e}")
        return []

    return entries


def _filter_by_year(papers: List[ArxivPaper], from_year: Optional[int]) -> List[ArxivPaper]:
    if from_year is None: return papers
    return [p for p in papers if p.year is None or p.year >= from_year]


# --- FIX: Hàm so sánh chuỗi để loại bỏ chính bài báo hiện tại ---
def _is_same_paper(title1: str, title2: str) -> bool:
    # So sánh độ tương đồng của tiêu đề (vì ID có thể khác phiên bản v1, v2)
    return SequenceMatcher(None, title1.lower(), title2.lower()).ratio() > 0.9


# -----------------------------------------------------------------------------
# 3. LLM Re-ranking
# -----------------------------------------------------------------------------

def rerank_papers_with_llm(
    base_title: str,
    base_abstract: str,
    candidates: List[ArxivPaper],
    top_k: int = 5,
    model_name: str = "gpt-4o-mini",
) -> List[dict]:

    # --- FIX: Lọc bỏ bài báo hiện tại (Self-Hit) ---
    filtered_candidates = []
    for p in candidates:
        if not _is_same_paper(base_title, p.title):
            filtered_candidates.append(p)

    if not filtered_candidates:
        return []

    llm = ChatOpenAI(model=model_name, temperature=0.1)

    prompt = ChatPromptTemplate.from_template(
        """You are a research assistant. The user is reading:
TITLE: {base_title}

Select the {top_k} most relevant *related* papers from the candidates below.
Focus on papers that use similar methods (RAG, Agents) or solve similar problems.

Return JSON: {{ "results": [ {{ "arxiv_id": "...", "reason": "..." }}, ... ] }}

CANDIDATES:
{candidates_json}
"""
    )

    cand_json = json.dumps([p.to_public_dict() for p in filtered_candidates][:30])

    try:
        raw = (prompt | llm | StrOutputParser()).invoke({
            "base_title": base_title,
            "top_k": top_k,
            "candidates_json": cand_json,
        })

        raw = raw.strip().replace("```json", "").replace("```", "")
        data = json.loads(raw)

        cleaned: List[dict] = []
        selected = data.get("results", [])

        # Build mapping arxiv_id -> (reason, rank_index)
        selected_map = {
            item["arxiv_id"]: (item.get("reason", ""), idx)
            for idx, item in enumerate(selected)
            if "arxiv_id" in item
        }

        total = max(len(selected_map), 1)

        for p in filtered_candidates:
            if p.arxiv_id in selected_map:
                reason, idx = selected_map[p.arxiv_id]
                d = p.to_public_dict()
                # Convert rank index -> score in [0,1], 1.0 = best, ~0.7 = worst (for UI % match)
                if total == 1:
                    score = 1.0
                else:
                    # Higher rank (smaller idx) → higher score
                    normalized = 1.0 - (idx / (total - 1))
                    score = 0.7 + normalized * 0.3
                d["score"] = float(f"{score:.4f}")
                d["reason"] = reason
                cleaned.append(d)

        # Ensure results are ordered by score desc
        cleaned.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return cleaned[:top_k]

    except Exception as e:
        print(f"Rerank Error: {e}")
        # Fallback
        return [
            {**p.to_public_dict(), "score": 0.0, "reason": "Fallback result"}
            for p in filtered_candidates[:top_k]
        ]


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def suggest_related_papers(
    base_title: str,
    base_abstract: str,
    categories: Optional[List[str]] = None,
    max_results: int = 30,
    top_k: int = 5,
) -> List[dict]:

    search_query = generate_search_query_with_llm(base_title, base_abstract)

    if categories:
        cat_part = " OR ".join(f"cat:{c}" for c in categories)
        search_query = f"({search_query}) AND ({cat_part})"

    # Tăng max_results lên để bù cho việc lọc trùng
    raw_papers = search_arxiv_raw(search_query, max_results=max_results + 5)

    year_now = _dt.datetime.utcnow().year
    from_year = year_now - 4 if base_title else None

    filtered = _filter_by_year(raw_papers, from_year)

    ranked = rerank_papers_with_llm(
        base_title=base_title,
        base_abstract=base_abstract,
        candidates=filtered,
        top_k=top_k,
    )

    return ranked