from __future__ import annotations

import json
import base64
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import settings
from app.constants import CATEGORY_OPTIONS, DOC_TYPE_OPTIONS
from app.services.tokens import count_tokens


def _openai_kwargs(provider: Optional[str] = None, inference: Optional[dict[str, str]] = None) -> dict:
    active_provider = provider or settings.llm_provider
    if inference and inference.get("api_key"):
        base_url = inference.get("base_url")
        inferred_provider = inference.get("provider") or active_provider
        if inferred_provider == "openrouter" or base_url:
            return {
                "api_key": inference["api_key"],
                "base_url": base_url or settings.openrouter_base_url,
                "default_headers": {
                    "HTTP-Referer": settings.app_base_url or "http://localhost",
                    "X-Title": inference.get("title") or "simplr",
                },
            }
        return {"api_key": inference["api_key"], **({"base_url": base_url} if base_url else {})}
    if active_provider == "openrouter":
        if not settings.openrouter_api_key:
            return {}
        return {
            "api_key": settings.openrouter_api_key,
            "base_url": settings.openrouter_base_url,
            "default_headers": {
                "HTTP-Referer": settings.app_base_url or "http://localhost",
                "X-Title": "simplr",
            },
        }
    if not settings.openai_api_key:
        return {}
    return {"api_key": settings.openai_api_key}


def get_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    provider: Optional[str] = None,
    inference: Optional[dict[str, str]] = None,
) -> Optional[ChatOpenAI]:
    kwargs = _openai_kwargs(provider, inference)
    if not kwargs:
        return None
    model_name = model or settings.llm_model
    temp_value = temperature if temperature is not None else 0.2
    extra = {}
    if max_tokens is not None:
        extra["max_tokens"] = max_tokens
    return ChatOpenAI(model=model_name, temperature=temp_value, **kwargs, **extra)


def _safe_json_parse(raw: str) -> Optional[dict]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


SUMMARY_SYSTEM_PROMPT = "You summarize incoming knowledge into a concise knowledge-base article summary."
SUMMARY_USER_PROMPT = "Summarize the following content in 4-7 bullet points.\n\n{content}"

RAG_SYSTEM_PROMPT = (
    "You answer questions using the provided knowledge base context. "
    "Always respond in clean Markdown with proper headings, paragraphs, and lists. "
    "Leave a blank line between paragraphs and before lists. "
    "If the answer is not in the context, provide a best-effort response "
    "based on the closest relevant context and start with the disclaimer: "
    "'Closest match in the knowledge base:'. If no context is provided, say you don't know."
)
RAG_USER_PROMPT = (
    "Context:\n{context}\n\nQuestion:\n{question}\n\n"
    "Answer in a concise, helpful way and cite sources using [1], [2], etc "
    "based on the Sources list in the context."
)

FREE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer directly and clearly. "
    "If you are uncertain, say so briefly and suggest how to verify."
)


def summarize_text(text: str, inference: Optional[dict[str, str]] = None) -> str:
    summary, _usage = summarize_text_with_usage(text, inference=inference)
    return summary


def summarize_text_with_usage(text: str, inference: Optional[dict[str, str]] = None) -> tuple[str, dict[str, int]]:
    llm = get_llm(inference=inference)
    if llm is None:
        summary = text[:700].strip()
        prompt_tokens = count_tokens(text)
        completion_tokens = count_tokens(summary)
        return summary, {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SUMMARY_SYSTEM_PROMPT),
            ("user", SUMMARY_USER_PROMPT),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({"content": text}).strip()
    prompt_text = f"{SUMMARY_SYSTEM_PROMPT}\n{SUMMARY_USER_PROMPT.format(content=text)}"
    prompt_tokens = count_tokens(prompt_text, llm.model_name)
    completion_tokens = count_tokens(summary, llm.model_name)
    return summary, {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _normalize_categories(value: Optional[list[str]]) -> list[str]:
    if not value:
        return []
    normalized = []
    allowed = set(CATEGORY_OPTIONS)
    for item in value:
        if not isinstance(item, str):
            continue
        cleaned = item.strip().lower().replace(" ", "_")
        if cleaned in allowed and cleaned not in normalized:
            normalized.append(cleaned)
    return normalized


def _fallback_classification() -> dict:
    return {
        "category": "uncategorized",
        "categories": ["uncategorized"],
        "doc_type": "other",
        "document_date": None,
        "language": "en",
        "tags": [],
        "entities": {"people": [], "orgs": [], "locations": []},
        "date_mentions": [],
        "confidence": None,
    }


def categorize_and_extract(text: str, inference: Optional[dict[str, str]] = None) -> dict:
    llm = get_llm(inference=inference)
    if llm is None:
        return _fallback_classification()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a classifier. Return JSON only with keys: "
                "category, categories, doc_type, document_date, language, tags, entities, date_mentions, confidence.",
            ),
            (
                "user",
                "Classify the content into one or more categories from: "
                f"{', '.join(CATEGORY_OPTIONS)}. "
                "Set category to the single best primary category from that list. "
                f"Set doc_type to one of: {', '.join(DOC_TYPE_OPTIONS)}. "
                "Set document_date to the single most relevant date (YYYY-MM-DD) if present. "
                "Set language (e.g., en). "
                "Extract entities with keys: people (array), orgs (array), locations (array). "
                "Extract tags (array) and date_mentions (array of YYYY-MM or YYYY-MM-DD). "
                "Set confidence from 0 to 1. "
                "Return JSON only.\n\nContent:\n{content}",
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"content": text}).strip()
    parsed = _safe_json_parse(raw)
    if not parsed:
        return _fallback_classification()

    categories = _normalize_categories(parsed.get("categories"))
    primary = parsed.get("category")
    if isinstance(primary, str):
        primary = primary.strip().lower().replace(" ", "_")
    else:
        primary = None
    if primary and primary in CATEGORY_OPTIONS and primary not in categories:
        categories = [primary, *categories]
    if not categories:
        categories = ["uncategorized"]
    if not primary or primary not in CATEGORY_OPTIONS:
        primary = categories[0]

    return {
        "category": primary,
        "categories": categories,
        "doc_type": parsed.get("doc_type") or "other",
        "document_date": parsed.get("document_date"),
        "language": parsed.get("language") or "en",
        "tags": parsed.get("tags") or [],
        "entities": parsed.get("entities") or {"people": [], "orgs": [], "locations": []},
        "date_mentions": parsed.get("date_mentions") or [],
        "confidence": parsed.get("confidence"),
    }


def extract_insights(text: str, inference: Optional[dict[str, str]] = None) -> dict:
    llm = get_llm(inference=inference)
    if llm is None:
        return {}

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You extract decision-ready insights from knowledge base content. "
                "Return JSON only with keys: key_points, decisions, action_items, risks, metrics, stakeholders, deadlines. "
                "Each key should be an array of short strings. Keep arrays empty if not present.",
            ),
            (
                "user",
                "Extract insights from the content. Prefer concrete takeaways, numbers, and dates. "
                "Return JSON only.\n\nContent:\n{content}",
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"content": text}).strip()
    parsed = _safe_json_parse(raw)
    if not parsed:
        return {}

    def _as_list(value: Optional[list[str]]) -> list[str]:
        if not value or not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, str) and item.strip()]

    return {
        "key_points": _as_list(parsed.get("key_points")),
        "decisions": _as_list(parsed.get("decisions")),
        "action_items": _as_list(parsed.get("action_items")),
        "risks": _as_list(parsed.get("risks")),
        "metrics": _as_list(parsed.get("metrics")),
        "stakeholders": _as_list(parsed.get("stakeholders")),
        "deadlines": _as_list(parsed.get("deadlines")),
    }


def answer_with_context(
    question: str,
    context: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    provider: Optional[str] = None,
    inference: Optional[dict[str, str]] = None,
) -> str:
    llm = get_llm(model=model, temperature=temperature, max_tokens=max_tokens, provider=provider, inference=inference)
    if llm is None:
        return "LLM is not configured. Set your API key to enable answers."

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            ("user", RAG_USER_PROMPT),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question}).strip()


def answer_freely(
    question: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    provider: Optional[str] = None,
    inference: Optional[dict[str, str]] = None,
) -> str:
    llm = get_llm(model=model, temperature=temperature, max_tokens=max_tokens, provider=provider, inference=inference)
    if llm is None:
        return "LLM is not configured. Set your API key to enable answers."

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", FREE_SYSTEM_PROMPT),
            ("user", "{question}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question}).strip()


def stream_answer_with_context(
    question: str,
    context: str,
    history_text: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    provider: Optional[str] = None,
    inference: Optional[dict[str, str]] = None,
):
    llm = get_llm(model=model, temperature=temperature, max_tokens=max_tokens, provider=provider, inference=inference)
    if llm is None:
        return ["LLM is not configured. Set your API key to enable answers."]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            ("system", "Conversation history:\n{history}"),
            ("user", RAG_USER_PROMPT),
        ]
    )
    chain = prompt | llm
    for chunk in chain.stream({"context": context, "question": question, "history": history_text}):
        content = getattr(chunk, "content", "") or ""
        if content:
            yield content


def stream_answer_freely(
    question: str,
    history_text: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    provider: Optional[str] = None,
    inference: Optional[dict[str, str]] = None,
):
    llm = get_llm(model=model, temperature=temperature, max_tokens=max_tokens, provider=provider, inference=inference)
    if llm is None:
        return ["LLM is not configured. Set your API key to enable answers."]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", FREE_SYSTEM_PROMPT),
            ("system", "Conversation history:\n{history}"),
            ("user", "{question}"),
        ]
    )
    chain = prompt | llm
    for chunk in chain.stream({"question": question, "history": history_text}):
        content = getattr(chunk, "content", "") or ""
        if content:
            yield content


def extract_text_from_images(image_bytes: list[bytes], inference: Optional[dict[str, str]] = None) -> str:
    if not image_bytes:
        return ""
    llm = get_llm(model=settings.vision_model, inference=inference)
    if llm is None:
        return ""
    content_parts = [{"type": "text", "text": "Extract the text from these images. Return plain text."}]
    for data in image_bytes[:3]:
        b64 = base64.b64encode(data).decode("utf-8")
        content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    prompt = ChatPromptTemplate.from_messages([("user", content_parts)])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({}).strip()


def estimate_rag_usage(
    question: str,
    context: str,
    history_text: str,
    answer: str,
    model: Optional[str] = None,
) -> dict[str, int]:
    prompt_text = "\n".join(
        [
            RAG_SYSTEM_PROMPT,
            f"Conversation history:\n{history_text}",
            RAG_USER_PROMPT.format(context=context, question=question),
        ]
    )
    prompt_tokens = count_tokens(prompt_text, model)
    completion_tokens = count_tokens(answer, model)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def estimate_free_usage(
    question: str,
    history_text: str,
    answer: str,
    model: Optional[str] = None,
) -> dict[str, int]:
    prompt_text = "\n".join(
        [
            FREE_SYSTEM_PROMPT,
            f"Conversation history:\n{history_text}",
            question,
        ]
    )
    prompt_tokens = count_tokens(prompt_text, model)
    completion_tokens = count_tokens(answer, model)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
