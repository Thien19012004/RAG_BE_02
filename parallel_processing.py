# -*- coding: utf-8 -*-
"""
Parallel processing utilities for RAG pipeline optimization
- Concurrent summarization of texts and images
- Batch processing with rate limiting
- Async operations for better performance
"""

import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, List, Dict, Tuple
import threading
from queue import Queue


class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, max_calls_per_second: float = 2.0):
        self.max_calls_per_second = max_calls_per_second
        self.min_interval = 1.0 / max_calls_per_second
        self.last_call_time = 0
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limit"""
        with self.lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time
            
            if time_since_last_call < self.min_interval:
                sleep_time = self.min_interval - time_since_last_call
                time.sleep(sleep_time)
            
            self.last_call_time = time.time()


def process_batch_with_rate_limit(
    items: List[Any],
    process_func: Callable,
    batch_size: int = 5,
    max_workers: int = 3,
    rate_limit_per_second: float = 2.0
) -> List[Any]:
    """Process items in parallel batches with rate limiting"""
    
    rate_limiter = RateLimiter(rate_limit_per_second)
    results = [None] * len(items)
    
    def process_item_with_limit(index_and_item):
        """Process single item with rate limiting"""
        index, item = index_and_item
        rate_limiter.wait_if_needed()
        try:
            result = process_func(item)
            return index, result, None
        except Exception as e:
            return index, None, str(e)
    
    # Process in batches to avoid overwhelming APIs
    for batch_start in range(0, len(items), batch_size):
        batch_end = min(batch_start + batch_size, len(items))
        batch_items = [(i, items[i]) for i in range(batch_start, batch_end)]
        
        print(f"🔄 Processing batch {batch_start//batch_size + 1}: items {batch_start+1}-{batch_end}")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(process_item_with_limit, item): item[0] 
                for item in batch_items
            }
            
            for future in as_completed(future_to_index):
                index, result, error = future.result()
                if error:
                    print(f"❌ Error processing item {index+1}: {error}")
                    results[index] = ""
                else:
                    results[index] = result
                    print(f"✅ Processed item {index+1}")
    
    return results


def summarize_texts_parallel(
    texts: List[Any],
    cache_file: str,
    summarize_chain,
    to_str: Callable[[Any], str] = lambda x: x,
    use_cache: bool = True,
    batch_size: int = 5,
    max_workers: int = 3
) -> List[str]:
    """Summarize texts in parallel with caching"""
    
    if use_cache and os.path.exists(cache_file):
        cached_summaries = json.load(open(cache_file, encoding="utf-8"))
        print(f"📁 Loaded {len(cached_summaries)} cached text summaries from {cache_file}")
        return cached_summaries
    
    print(f"🔄 Parallel summarizing {len(texts)} texts...")
    
    def summarize_single_text(text_item):
        """Summarize a single text item"""
        try:
            item_text = to_str(text_item)
            
            # Skip empty or very short items
            if not item_text or not isinstance(item_text, str) or len(item_text.strip()) < 20:
                return ""
            
            # Skip items that are too long
            if len(item_text) > 6000:  # Reduced from 8000
                return ""
            
            # Add retry logic
            max_retries = 2  # Reduced from 3
            for retry in range(max_retries):
                try:
                    result = summarize_chain.invoke({"element": item_text})
                    if result and isinstance(result, str) and len(result.strip()) > 10:
                        return result.strip()
                    return ""
                except Exception as retry_e:
                    if retry == max_retries - 1:
                        print(f"⚠️ Failed after {max_retries} retries: {str(retry_e)}")
                        return ""
                    time.sleep(0.5)  # Shorter retry delay
            
        except Exception as e:
            print(f"❌ Error in text summarization: {str(e)}")
            return ""
    
    # Process texts in parallel
    summaries = process_batch_with_rate_limit(
        texts,
        summarize_single_text,
        batch_size=batch_size,
        max_workers=max_workers,
        rate_limit_per_second=2.0
    )
    
    print(f"📊 Text summary complete: {len(summaries)} total")
    
    # Save to cache
    json.dump(summaries, open(cache_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"💾 Saved text summaries to {cache_file}")
    
    return summaries


def summarize_images_parallel(
    images: List[str],
    cache_file: str,
    vision_chain,
    use_cache: bool = True,
    batch_size: int = 3,  # Smaller batch for images
    max_workers: int = 2  # Fewer workers for images
) -> List[str]:
    """Summarize images in parallel with caching"""
    
    if use_cache and os.path.exists(cache_file):
        cached_summaries = json.load(open(cache_file, encoding="utf-8"))
        print(f"📁 Loaded {len(cached_summaries)} cached image summaries from {cache_file}")
        return cached_summaries
    
    print(f"🔄 Parallel summarizing {len(images)} images...")
    
    def summarize_single_image(image_b64):
        """Summarize a single image"""
        try:
            # Validate base64 image
            if not image_b64 or len(image_b64) < 100:
                return ""
            
            # Add retry logic
            max_retries = 2
            for retry in range(max_retries):
                try:
                    result = vision_chain.invoke({"image_b64": image_b64})
                    if result and len(result.strip()) > 0:
                        return result.strip()
                    return ""
                except Exception as retry_e:
                    if retry == max_retries - 1:
                        print(f"⚠️ Failed after {max_retries} retries: {str(retry_e)}")
                        return ""
                    time.sleep(0.5)
            
        except Exception as e:
            print(f"❌ Error in image summarization: {str(e)}")
            return ""
    
    # Process images in parallel
    summaries = process_batch_with_rate_limit(
        images,
        summarize_single_image,
        batch_size=batch_size,
        max_workers=max_workers,
        rate_limit_per_second=1.5  # Slower rate for images
    )
    
    print(f"📊 Image summary complete: {len(summaries)} total")
    
    # Save to cache
    json.dump(summaries, open(cache_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"💾 Saved image summaries to {cache_file}")
    
    return summaries


def process_all_content_parallel(
    texts: List[Any],
    tables_html: List[str],
    images: List[str],
    cache_files: Dict[str, str],
    text_summarizer,
    vision_summarizer,
    use_cache: bool = True
) -> Tuple[List[str], List[str], List[str]]:
    """Process all content types in parallel"""
    
    print("🚀 Starting parallel content processing...")
    start_time = time.time()
    
    # Create tasks for parallel execution
    tasks = []
    
    # Text summarization task
    if texts:
        tasks.append((
            "texts",
            lambda: summarize_texts_parallel(
                texts,
                cache_files["text_summaries"],
                text_summarizer,
                to_str=lambda x: x.text,
                use_cache=use_cache,
                batch_size=6,  # Larger batch for texts
                max_workers=4
            )
        ))
    
    # Table summarization task
    if tables_html:
        tasks.append((
            "tables",
            lambda: summarize_texts_parallel(
                tables_html,
                cache_files["table_summaries"],
                text_summarizer,
                to_str=lambda x: x,
                use_cache=use_cache,
                batch_size=6,
                max_workers=4
            )
        ))
    
    # Image summarization task
    if images:
        tasks.append((
            "images",
            lambda: summarize_images_parallel(
                images,
                cache_files["image_summaries"],
                vision_summarizer,
                use_cache=use_cache,
                batch_size=3,
                max_workers=2
            )
        ))
    
    # Execute tasks in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_task = {
            executor.submit(task_func): task_name 
            for task_name, task_func in tasks
        }
        
        for future in as_completed(future_to_task):
            task_name = future_to_task[future]
            try:
                results[task_name] = future.result()
                print(f"✅ Completed {task_name} processing")
            except Exception as e:
                print(f"❌ Error in {task_name} processing: {str(e)}")
                results[task_name] = []
    
    end_time = time.time()
    print(f"🎉 Parallel processing completed in {end_time - start_time:.2f} seconds")
    
    return (
        results.get("texts", []),
        results.get("tables", []),
        results.get("images", [])
    )
