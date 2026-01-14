# -*- coding: utf-8 -*-
import asyncio
import json
import os
import time
from typing import Any, Callable, List

async def summarize_texts_parallel(
    texts: List[Any],
    cache_file: str,
    summarize_chain,
    to_str: Callable[[Any], str] = lambda x: x,
    use_cache: bool = True,
    batch_size: int = 20, # Tăng số lượng concurrent request lên 20
) -> List[str]:
    """Tóm tắt văn bản hàng loạt bằng Asyncio để giảm I/O wait."""
    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except: pass

    print(f"⚡ [ASYNC] Summarizing {len(texts)} texts...")
    semaphore = asyncio.Semaphore(batch_size)

    async def _sum_task(item):
        async with semaphore:
            text = to_str(item)
            if not text or len(text.strip()) < 20: return ""
            # Giới hạn token đầu vào để tránh overload
            for _ in range(2):
                try:
                    res = await summarize_chain.ainvoke({"element": text[:6000]})
                    return res.strip() if isinstance(res, str) else ""
                except Exception:
                    await asyncio.sleep(0.5)
            return ""

    tasks = [_sum_task(t) for t in texts]
    summaries = await asyncio.gather(*tasks)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    return summaries

async def summarize_images_parallel(
    images: List[str],
    cache_file: str,
    vision_chain,
    use_cache: bool = True,
    batch_size: int = 5,
) -> List[str]:
    """Tóm tắt hình ảnh hàng loạt bằng Asyncio."""
    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except: pass

    semaphore = asyncio.Semaphore(batch_size)

    async def _vis_task(img_b64):
        async with semaphore:
            if not img_b64 or len(img_b64) < 100: return ""
            try:
                res = await vision_chain.ainvoke({"image_b64": img_b64})
                return res.strip() if res else ""
            except Exception: return ""

    tasks = [_vis_task(img) for img in images]
    summaries = await asyncio.gather(*tasks)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    return summaries