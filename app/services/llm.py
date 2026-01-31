from __future__ import annotations

import json
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import settings


def _openai_kwargs() -> dict:
    if settings.llm_provider == "openrouter":
        if not settings.openrouter_api_key:
            return {}
        return {
            "api_key": settings.openrouter_api_key,
            "base_url": settings.openrouter_base_url,
            "default_headers": {
                "HTTP-Referer": "http://localhost",
                "X-Title": "rag-email-mvp",
            },
        }
    if not settings.openai_api_key:
        return {}
    return {"api_key": settings.openai_api_key}


def get_llm() -> Optional[ChatOpenAI]:
    kwargs = _openai_kwargs()
    if not kwargs:
        return None
    return ChatOpenAI(model=settings.llm_model, temperature=0.2, **kwargs)


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


def summarize_text(text: str) -> str:
    llm = get_llm()
    if llm is None:
        return text[:700].strip()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You summarize incoming knowledge into a concise knowledge-base article summary.",
            ),
            (
                "user",
                "Summarize the following content in 4-7 bullet points.\n\n{content}",
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"content": text}).strip()


def categorize_and_extract(text: str) -> dict:
    llm = get_llm()
    if llm is None:
        return {
            "category": "uncategorized",
            "doc_type": "other",
            "document_date": None,
            "language": "en",
            "tags": [],
            "entities": {"people": [], "orgs": [], "locations": []},
            "date_mentions": [],
            "confidence": None,
        }

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a classifier. Return JSON only with keys: "
                "category, doc_type, document_date, language, tags, entities, date_mentions, confidence.",
            ),
            (
                "user",
                "Classify the content into one of: support_tickets, policies, "
                "documentation, projects, other. "
                "Set doc_type to one of: report, memo, policy, ticket, meeting_notes, other. "
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
        return {
            "category": "uncategorized",
            "doc_type": "other",
            "document_date": None,
            "language": "en",
            "tags": [],
            "entities": {"people": [], "orgs": [], "locations": []},
            "date_mentions": [],
            "confidence": None,
        }

    return {
        "category": parsed.get("category") or "uncategorized",
        "doc_type": parsed.get("doc_type") or "other",
        "document_date": parsed.get("document_date"),
        "language": parsed.get("language") or "en",
        "tags": parsed.get("tags") or [],
        "entities": parsed.get("entities") or {"people": [], "orgs": [], "locations": []},
        "date_mentions": parsed.get("date_mentions") or [],
        "confidence": parsed.get("confidence"),
    }


def answer_with_context(question: str, context: str) -> str:
    llm = get_llm()
    if llm is None:
        return "LLM is not configured. Set your API key to enable answers."

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You answer questions using the provided knowledge base context. "
                "If the answer is not in the context, provide a best-effort response "
                "based on the closest relevant context and start with the disclaimer: "
                "'Closest match in the knowledge base:'. If no context is provided, say you don't know.",
            ),
            (
                "user",
                "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer in a concise, helpful way.",
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question}).strip()


def stream_answer_with_context(question: str, context: str, history_text: str):
    llm = get_llm()
    if llm is None:
        return ["LLM is not configured. Set your API key to enable answers."]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You answer questions using the provided knowledge base context. "
                "If the answer is not in the context, provide a best-effort response "
                "based on the closest relevant context and start with the disclaimer: "
                "'Closest match in the knowledge base:'. If no context is provided, say you don't know.",
            ),
            ("system", "Conversation history:\n{history}"),
            (
                "user",
                "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer in a concise, helpful way.",
            ),
        ]
    )
    chain = prompt | llm
    for chunk in chain.stream({"context": context, "question": question, "history": history_text}):
        content = getattr(chunk, "content", "") or ""
        if content:
            yield content
