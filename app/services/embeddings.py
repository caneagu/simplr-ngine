from __future__ import annotations

from typing import Optional

from langchain_openai import OpenAIEmbeddings

from app.config import settings


def _embedding_kwargs() -> dict:
    if settings.llm_provider == "openrouter":
        if not settings.openrouter_api_key:
            return {}
        return {
            "api_key": settings.openrouter_api_key,
            "base_url": settings.openrouter_base_url,
        }
    if not settings.openai_api_key:
        return {}
    return {"api_key": settings.openai_api_key}


def get_embeddings() -> Optional[OpenAIEmbeddings]:
    kwargs = _embedding_kwargs()
    if not kwargs:
        return None
    return OpenAIEmbeddings(model=settings.embedding_model, **kwargs)


def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = get_embeddings()
    if embeddings is None:
        raise RuntimeError("Embedding provider is not configured.")
    return embeddings.embed_documents(texts)


def embed_query(text: str) -> list[float]:
    embeddings = get_embeddings()
    if embeddings is None:
        raise RuntimeError("Embedding provider is not configured.")
    return embeddings.embed_query(text)
