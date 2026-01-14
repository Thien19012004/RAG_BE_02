import json
import os
import time
from typing import Any, Callable, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


GROQ_TEXT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct" # Updated to new Llama model or keep yours
TEXT_SLEEP_SECONDS = 0.5
VISION_MODEL = "gpt-4o-mini"


def build_text_summarizer():
    """Build text summarizer using Groq"""
    text_llm = ChatGroq(model=GROQ_TEXT_MODEL) # Or your config
    prompt_text = ChatPromptTemplate.from_template(
        """Summarize the following scientific text/table concisely.
        No preamble.
        Content: {element}"""
    )
    return prompt_text | text_llm | StrOutputParser()


def build_vision_summarizer():
    """Build vision summarizer using OpenAI for Ingestion"""
    vision_llm = ChatOpenAI(model=VISION_MODEL)
    vision_prompt = ChatPromptTemplate.from_messages([
        (
            "user",
            [
                {"type": "text", "text": "Describe this scientific image (figure/chart) in detail for retrieval purposes."},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_b64}"}},
            ],
        )
    ])
    return vision_prompt | vision_llm | StrOutputParser()


    # Keep existing implementation logic
    if use_cache and os.path.exists(cache_file):
        return json.load(open(cache_file, encoding="utf-8"))
    
    out = []
    for it in items:
        try:
            txt = to_str(it)
            if not txt or len(txt) < 10: 
                out.append("")
                continue
            res = summarize_chain.invoke({"element": txt})
            out.append(res)
            time.sleep(sleep_s)
        except Exception as e:
            print(f"Error summarizing: {e}")
            out.append("")
            
    json.dump(out, open(cache_file, "w"), indent=2)
    return out

    # Keep existing implementation logic
    if use_cache and os.path.exists(cache_file):
        return json.load(open(cache_file, encoding="utf-8"))
        
    out = []
    for b64 in imgs:
        try:
            if not b64:
                out.append("")
                continue
            res = vision_chain.invoke({"image_b64": b64})
            out.append(res)
            time.sleep(sleep_s)
        except Exception as e:
            print(f"Error vision sum: {e}")
            out.append("")
            
    json.dump(out, open(cache_file, "w"), indent=2)
    return out