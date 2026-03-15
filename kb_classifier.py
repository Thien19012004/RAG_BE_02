"""
KB Classifier — Auto-classify papers into Knowledge Base categories.

Uses LLM to analyze paper title + abstract and suggest category matches
from a predefined CS taxonomy.
"""
import json
from typing import Any, Dict, List
from model_factory import get_llm


# CS category taxonomy — matches backend kb_categories
CS_CATEGORIES = [
    {"slug": "artificial_intelligence", "name": "Artificial Intelligence"},
    {"slug": "machine_learning", "name": "Machine Learning"},
    {"slug": "deep_learning", "name": "Deep Learning"},
    {"slug": "nlp", "name": "Natural Language Processing"},
    {"slug": "computer_vision", "name": "Computer Vision"},
    {"slug": "software_engineering", "name": "Software Engineering"},
    {"slug": "databases", "name": "Databases"},
    {"slug": "algorithms", "name": "Data Structures & Algorithms"},
    {"slug": "security", "name": "Cryptography & Security"},
    {"slug": "networking", "name": "Networking"},
    {"slug": "distributed_systems", "name": "Distributed Systems"},
    {"slug": "hci", "name": "Human-Computer Interaction"},
]

CATEGORY_SLUGS = [c["slug"] for c in CS_CATEGORIES]
CATEGORY_LIST_STR = "\n".join(
    f"- {c['slug']}: {c['name']}" for c in CS_CATEGORIES
)


def classify_paper(
    title: str,
    abstract: str,
) -> Dict[str, Any]:
    """
    Classify a scientific paper into KB categories using LLM.

    Args:
        title: Paper title
        abstract: Paper abstract text

    Returns:
        {
            "categories": [{"slug": "machine_learning", "confidence": 0.95}, ...],
            "tags": ["transformer", "attention", "nlp"]
        }
    """
    if not title and not abstract:
        return {"categories": [], "tags": []}

    try:
        llm = get_llm("classification")

        prompt = (
            "You are a computer science paper classifier. Given a paper's title and abstract, "
            "classify it into one or more categories from the list below. "
            "Also extract 3-5 relevant tags (lowercase, specific terms).\n\n"
            f"Available categories:\n{CATEGORY_LIST_STR}\n\n"
            f"Paper Title: {title}\n"
            f"Paper Abstract: {abstract[:2000]}\n\n"
            "Respond in VALID JSON only, no extra text:\n"
            '{"categories": [{"slug": "category_slug", "confidence": 0.0-1.0}], '
            '"tags": ["tag1", "tag2"]}\n\n'
            "Rules:\n"
            "- Pick 1-3 most relevant categories\n"
            "- Confidence should reflect how well the paper matches\n"
            "- Tags should be specific technical terms from the paper\n"
            "- Use ONLY slugs from the list above\n"
            "JSON:"
        )

        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        # Parse JSON from response
        result = _parse_classification_response(content)
        return result

    except Exception as e:
        print(f"[kb_classifier] Error: {e}")
        return {"categories": [], "tags": []}


def _parse_classification_response(content: str) -> Dict[str, Any]:
    """Parse and validate LLM classification response."""
    try:
        # Try to extract JSON from response
        content = content.strip()
        if content.startswith("```"):
            # Remove markdown code block
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        result = json.loads(content)

        # Validate structure
        categories = result.get("categories", [])
        tags = result.get("tags", [])

        # Filter only valid category slugs and clamp confidence
        valid_categories = []
        for cat in categories:
            if isinstance(cat, dict) and cat.get("slug") in CATEGORY_SLUGS:
                confidence = float(cat.get("confidence", 0.5))
                valid_categories.append({
                    "slug": cat["slug"],
                    "confidence": max(0.0, min(1.0, confidence)),
                })

        # Ensure tags are strings
        valid_tags = [str(t).lower().strip() for t in tags if isinstance(t, str)]

        return {
            "categories": valid_categories,
            "tags": valid_tags[:10],
        }

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"[kb_classifier] JSON parse error: {e}")
        return {"categories": [], "tags": []}
