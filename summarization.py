import json, os, time
from typing import Any, Callable, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from config import CACHE_DIR


GROQ_TEXT_MODEL = "llama-3.3-70b-versatile"
TEXT_SLEEP_SECONDS = 3.0
VISION_MODEL = "gpt-4o-mini"
VISION_SLEEP_SECONDS = 3.0


def build_text_summarizer():
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


def summarize_with_cache(
    items: List[Any],
    cache_file: str,
    summarize_chain,
    to_str: Callable[[Any], str] = lambda x: x,
    sleep_s: float = 0.0,
    use_cache: bool = True,
):
    if use_cache and os.path.exists(cache_file):
        return json.load(open(cache_file, encoding="utf-8"))
    out: List[str] = []
    for i, it in enumerate(items):
        try:
            s = summarize_chain.invoke({"element": to_str(it)})
            out.append(s)
            time.sleep(sleep_s)
        except Exception:
            out.append("")
    json.dump(out, open(cache_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return out


def summarize_images_with_cache(
    imgs: List[str],
    cache_file: str,
    vision_chain,
    sleep_s: float = 0.0,
    use_cache: bool = True,
):
    if use_cache and os.path.exists(cache_file):
        return json.load(open(cache_file, encoding="utf-8"))
    outs: List[str] = []
    for b64 in imgs:
        try:
            s = vision_chain.invoke({"image_b64": b64})
            outs.append(s)
            time.sleep(sleep_s)
        except Exception:
            outs.append("")
    json.dump(outs, open(cache_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return outs


