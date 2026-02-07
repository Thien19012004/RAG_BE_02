# -*- coding: utf-8 -*-
"""
Parallel processing utilities for summarization.

ARCHITECTURE NOTE:
- Caching is now done via PostgreSQL database (database.py)
- No local file storage is used for summaries
"""
import asyncio
import hashlib
from typing import Any, Callable, List

from database import get_database


def _compute_content_hash(content: str) -> str:
    """Compute MD5 hash of content for cache validation."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()


async def summarize_texts_parallel(
    texts: List[Any],
    rag_paper_id: str,
    content_type: str,  # 'table' or 'image'
    summarize_chain,
    to_str: Callable[[Any], str] = lambda x: x,
    use_cache: bool = True,
    batch_size: int = 20,
) -> List[str]:
    """
    Summarize texts in parallel using asyncio.

    Args:
        texts: List of text items to summarize
        rag_paper_id: Paper ID for database caching
        content_type: 'table' or 'image' for cache key
        summarize_chain: LangChain chain for summarization
        to_str: Function to convert item to string
        use_cache: Whether to use database cache
        batch_size: Max concurrent requests

    Returns:
        List of summary strings
    """
    db = get_database()

    # Try to load from database cache
    if use_cache:
        cached = db.get_cached_summaries(rag_paper_id, content_type)
        if cached is not None and len(cached) == len(texts):
            print(f"✅ [CACHE] Loaded {len(cached)} {content_type} summaries from database")
            return cached

    print(f"⚡ [ASYNC] Summarizing {len(texts)} {content_type}s...")
    semaphore = asyncio.Semaphore(batch_size)

    async def _sum_task(item):
        async with semaphore:
            text = to_str(item)
            if not text or len(text.strip()) < 20:
                return ""
            # Retry logic with rate limit handling
            for _ in range(2):
                try:
                    res = await summarize_chain.ainvoke({"element": text[:6000]})
                    return res.strip() if isinstance(res, str) else ""
                except Exception:
                    await asyncio.sleep(0.5)
            return ""

    tasks = [_sum_task(t) for t in texts]
    summaries = await asyncio.gather(*tasks)

    # Compute hashes for cache invalidation
    content_hashes = [_compute_content_hash(to_str(t)) for t in texts]

    # Save to database cache
    db.save_cached_summaries(
        rag_paper_id=rag_paper_id,
        content_type=content_type,
        summaries=summaries,
        content_hashes=content_hashes,
    )
    print(f"💾 [CACHE] Saved {len(summaries)} {content_type} summaries to database")

    return summaries


async def summarize_images_parallel(
    images: List[str],
    rag_paper_id: str,
    content_type: str,  # Should be 'image'
    vision_chain,
    use_cache: bool = True,
    batch_size: int = 5,
) -> List[str]:
    """
    Summarize images in parallel using asyncio.

    Args:
        images: List of base64-encoded images
        rag_paper_id: Paper ID for database caching
        content_type: Should be 'image' for cache key
        vision_chain: LangChain chain for vision summarization
        use_cache: Whether to use database cache
        batch_size: Max concurrent requests

    Returns:
        List of summary strings
    """
    db = get_database()

    # Try to load from database cache
    if use_cache:
        cached = db.get_cached_summaries(rag_paper_id, content_type)
        if cached is not None and len(cached) == len(images):
            print(f"✅ [CACHE] Loaded {len(cached)} {content_type} summaries from database")
            return cached

    print(f"⚡ [ASYNC] Summarizing {len(images)} images...")
    semaphore = asyncio.Semaphore(batch_size)

    async def _vis_task(img_b64):
        async with semaphore:
            if not img_b64 or len(img_b64) < 100:
                return ""
            try:
                res = await vision_chain.ainvoke({"image_b64": img_b64})
                return res.strip() if res else ""
            except Exception:
                return ""

    tasks = [_vis_task(img) for img in images]
    summaries = await asyncio.gather(*tasks)

    # Compute hashes for cache invalidation (use first 100 chars of base64)
    content_hashes = [_compute_content_hash(img[:100] if img else "") for img in images]

    # Save to database cache
    db.save_cached_summaries(
        rag_paper_id=rag_paper_id,
        content_type=content_type,
        summaries=summaries,
        content_hashes=content_hashes,
    )
    print(f"💾 [CACHE] Saved {len(summaries)} image summaries to database")

    return summaries