import json
import os
import time
from typing import Any, Callable, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


GROQ_TEXT_MODEL = "llama-3.3-70b-versatile"
TEXT_SLEEP_SECONDS = 0.5  # Reduced from 3.0 for faster processing
VISION_MODEL = "gpt-4o-mini"
VISION_SLEEP_SECONDS = 0.5  # Reduced from 3.0 for faster processing


def build_text_summarizer():
    """Build text summarizer using Groq"""
    text_llm = ChatGroq(model=GROQ_TEXT_MODEL)
    prompt_text = ChatPromptTemplate.from_template(
        """You are an assistant tasked with summarizing tables and text.
        Give a concise summary of the table or text.

        Respond only with the summary, no additionnal comment.
        Do not start your message by saying "Here is a summary" or anything like that.
        Just give the summary as it is.

        Table or text chunk: {element}"""
    )
    return prompt_text | text_llm | StrOutputParser()


def build_vision_summarizer():
    """Build vision summarizer using OpenAI"""
    vision_llm = ChatOpenAI(model=VISION_MODEL)
    vision_prompt = ChatPromptTemplate.from_messages([
        (
            "user",
            [
                {"type": "text", "text": "Describe the image in detail. For context, the image is part of a research paper explaining the transformers architecture. Be specific about graphs, such as bar plots."},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_b64}"}},
            ],
        )
    ])
    return vision_prompt | vision_llm | StrOutputParser()


def build_region_explainer():
    """Build a region explainer specialized for math formulas, tables, and figures."""
    vision_llm = ChatOpenAI(model=VISION_MODEL)
    prompt_text = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert scientific assistant. Analyze a cropped region from a research paper. "
            "Determine if it is a math formula, a table, a figure/plot, or text. "
            "Then explain it clearly and concisely for a student audience. "
            "If it is a formula: explain the meaning of symbols, each term, and how the expression is used; if possible, outline steps or intuition. "
            "If it is a table: describe columns/rows, units, key trends, and notable comparisons. "
            "If it is a plot/figure: describe axes, units, variables, and the main relationship/insight. "
            "If it is text: summarize the key point. "
            "Avoid hallucination; if uncertain, say what is unclear."
        ),
        (
            "user",
            [
                {"type": "text", "text": "Please analyze and explain this cropped region."},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_b64}"}},
            ],
        )
    ])
    return prompt_text | vision_llm | StrOutputParser()


def build_region_explainer_hybrid():
    """Build hybrid explainer that uses both the cropped image and optional textual context from the paper."""
    vision_llm = ChatOpenAI(model=VISION_MODEL)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert scientific assistant. Analyze a cropped region from a research paper. "
            "You are also given textual context (summaries of related parts of the same paper). "
            "First, identify whether the region is a math formula, table, figure/plot, or text. "
            "Then produce a clear explanation for a student: "
            "- For formulas: define symbols, explain terms, and provide intuition/derivation outline if possible.\n"
            "- For tables: describe columns/rows, units, trends, comparisons.\n"
            "- For figures/plots: describe axes, variables, units, and the main relationships/insights.\n"
            "- For text: summarize the key point.\n"
            "Use the provided textual context to reduce hallucination; if context contradicts the image, state the uncertainty. "
            "Cite relevant pieces of the context in-line by short quotes if helpful."
        ),
        (
            "user",
            [
                {"type": "text", "text": "Context from the same paper (summarized):\n{context_text}\n\nNow explain the cropped region below."},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_b64}"}},
            ],
        ),
    ])
    return prompt | vision_llm | StrOutputParser()


def summarize_with_cache(
    items: List[Any],
    cache_file: str,
    summarize_chain,
    to_str: Callable[[Any], str] = lambda x: x,
    sleep_s: float = 0.0,
    use_cache: bool = True,
):
    """Summarize items with caching and better error handling"""
    if use_cache and os.path.exists(cache_file):
        cached_summaries = json.load(open(cache_file, encoding="utf-8"))
        print(f"📁 Loaded {len(cached_summaries)} cached summaries from {cache_file}")
        return cached_summaries
    
    print(f"🔄 Summarizing {len(items)} items...")
    out: List[str] = []
    failed_count = 0
    skipped_count = 0
    
    for i, it in enumerate(items):
        try:
            # Convert item to string for summarization
            item_text = to_str(it)
            
            # Skip empty, None, or very short items
            if not item_text or not isinstance(item_text, str) or len(item_text.strip()) < 20:
                print(f"⚠️ Skipping item {i}: too short or empty ({len(item_text) if item_text else 0} chars)")
                out.append("")
                skipped_count += 1
                continue
            
            # Skip items that are too long (might cause API issues)
            if len(item_text) > 8000:
                print(f"⚠️ Skipping item {i}: too long ({len(item_text)} chars)")
                out.append("")
                skipped_count += 1
                continue
            
            print(f"📝 Summarizing item {i+1}/{len(items)} (length: {len(item_text)})")
            
            # Add retry logic for API calls
            max_retries = 3
            for retry in range(max_retries):
                try:
                    s = summarize_chain.invoke({"element": item_text})
                    break
                except Exception as retry_e:
                    if retry == max_retries - 1:
                        raise retry_e
                    print(f"⚠️ Retry {retry + 1}/{max_retries} for item {i+1}: {str(retry_e)}")
                    time.sleep(sleep_s * 2)  # Longer sleep on retry
            
            # Check if summary is valid
            if s and isinstance(s, str) and len(s.strip()) > 10:
                out.append(s.strip())
                print(f"✅ Item {i+1} summarized successfully ({len(s.strip())} chars)")
            else:
                print(f"⚠️ Item {i+1} produced invalid summary: '{s}'")
                out.append("")
                failed_count += 1
                
            time.sleep(sleep_s)
            
        except Exception as e:
            print(f"❌ Failed to summarize item {i+1}: {str(e)}")
            out.append("")
            failed_count += 1
    
    print(f"📊 Summary complete: {len(out)} total, {failed_count} failed, {skipped_count} skipped")
    
    # Save to cache
    json.dump(out, open(cache_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"💾 Saved summaries to {cache_file}")
    
    return out


def summarize_images_with_cache(
    imgs: List[str],
    cache_file: str,
    vision_chain,
    sleep_s: float = 0.0,
    use_cache: bool = True,
):
    """Summarize images with caching"""
    if use_cache and os.path.exists(cache_file):
        cached_summaries = json.load(open(cache_file, encoding="utf-8"))
        print(f"📁 Loaded {len(cached_summaries)} cached image summaries from {cache_file}")
        return cached_summaries
    
    print(f"🔄 Summarizing {len(imgs)} images...")
    outs: List[str] = []
    failed_count = 0
    
    for i, b64 in enumerate(imgs):
        try:
            # Validate base64 image
            if not b64 or len(b64) < 100:
                print(f"⚠️ Skipping image {i}: invalid base64")
                outs.append("")
                continue
            
            print(f"🖼️ Summarizing image {i+1}/{len(imgs)} (size: {len(b64)} chars)")
            s = vision_chain.invoke({"image_b64": b64})
            
            # Check if summary is valid
            if s and len(s.strip()) > 0:
                outs.append(s.strip())
                print(f"✅ Image {i+1} summarized successfully")
            else:
                print(f"⚠️ Image {i+1} produced empty summary")
                outs.append("")
                failed_count += 1
                
            time.sleep(sleep_s)
            
        except Exception as e:
            print(f"❌ Failed to summarize image {i+1}: {str(e)}")
            outs.append("")
            failed_count += 1
    
    print(f"📊 Image summary complete: {len(outs)} total, {failed_count} failed")
    
    # Save to cache
    json.dump(outs, open(cache_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"💾 Saved image summaries to {cache_file}")
    
    return outs


