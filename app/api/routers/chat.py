from __future__ import annotations

import base64
import json
import uuid
from typing import Any, Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.deps import (
    _build_cited_context,
    _get_current_user,
    _get_session_and_user,
    _get_session_id,
    _get_user_inference_config,
    _maybe_stats_answer,
    _provider_for_inference,
    _render_history,
    _resolve_scope,
    _template_context,
    _user_groups,
)
from app.models import Article, Chunk, Source, UserInferenceConfig
from app.services.embeddings import embed_query
from app.services.llm import (
    answer_freely,
    answer_with_context,
    estimate_free_usage,
    estimate_rag_usage,
    stream_answer_freely,
    stream_answer_with_context,
)
from app.web import templates


router = APIRouter()

CHAT_MEMORY: dict[str, list[dict[str, str]]] = {}
CHAT_MEMORY_LIMIT = 6
CHAT_MIN_SEMANTIC = 0.78
CHAT_MIN_LEXICAL = 0.08
CHAT_ARTICLE_LIMIT = 3


@router.get("/chat", response_class=HTMLResponse)
def chat_ui(request: Request, db: Session = Depends(get_db)):
    session, current_user = _get_session_and_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)

    scope_name, selected_group = _resolve_scope(
        db,
        current_user,
        request.query_params.get("scope"),
        request.query_params.get("group_id"),
    )
    groups = _user_groups(db, current_user)
    inference_configs = (
        db.query(UserInferenceConfig)
        .filter(UserInferenceConfig.user_id == current_user.id)
        .order_by(UserInferenceConfig.created_at.desc())
        .all()
    )
    session_id = _get_session_id(request)
    response = templates.TemplateResponse(
        "chat.html",
        _template_context(
            request,
            current_user,
            session,
            question="",
            answer=None,
            references=[],
            error=None,
            scope=scope_name,
            selected_group=selected_group,
            groups=groups,
            inference_configs=inference_configs,
        ),
    )
    response.set_cookie("rag_session_id", session_id, max_age=60 * 60 * 6)
    return response


@router.post("/chat", response_class=HTMLResponse)
def chat_submit(
    request: Request,
    question: str = Form(...),
    ground: bool = Form(True),
    scope: Optional[str] = Form(None),
    group_id: Optional[str] = Form(None),
    inference_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    session, current_user = _get_session_and_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    error = None
    answer = None
    references = []

    scope_name, selected_group = _resolve_scope(db, current_user, scope, group_id)
    groups = _user_groups(db, current_user)
    inference_configs = (
        db.query(UserInferenceConfig)
        .filter(UserInferenceConfig.user_id == current_user.id)
        .order_by(UserInferenceConfig.created_at.desc())
        .all()
    )

    stats_answer = _maybe_stats_answer(question, current_user, db)
    if stats_answer:
        return templates.TemplateResponse(
            "chat.html",
            _template_context(
                request,
                current_user,
                session,
                question=question,
                answer=stats_answer,
                references=[],
                error=None,
                scope=scope_name,
                selected_group=selected_group,
                groups=groups,
            ),
        )

    if not inference_id:
        inference_id = request.query_params.get("inference_id")
    if not ground:
        inference = _get_user_inference_config(db, current_user)
        provider = _provider_for_inference(inference)
        if inference_id:
            override = db.get(UserInferenceConfig, inference_id)
            if override and override.user_id == current_user.id:
                inference = {
                    "provider": override.provider,
                    "api_key": override.api_key,
                    "base_url": override.base_url or "",
                    "title": "simplr",
                }
                provider = _provider_for_inference(inference)
        answer = answer_freely(question, provider=provider, inference=inference)
        return templates.TemplateResponse(
            "chat.html",
            _template_context(
                request,
                current_user,
                session,
                question=question,
                answer=answer,
                references=[],
                error=None,
                scope=scope_name,
                selected_group=selected_group,
                groups=groups,
                inference_configs=inference_configs,
            ),
        )

    try:
        query_embedding = embed_query(question)
    except RuntimeError as exc:
        error = str(exc)
        return templates.TemplateResponse(
            "chat.html",
            _template_context(
                request,
                current_user,
                session,
                question=question,
                answer=None,
                references=[],
                error=error,
            ),
        )

    distance = Chunk.embedding.cosine_distance(query_embedding).label("distance")
    matches = (
        db.query(Chunk, Article, distance)
        .join(Article, Article.id == Chunk.article_id)
        .filter(
            Article.group_id == selected_group.id if scope_name == "group" and selected_group else
            and_(Article.owner_id == current_user.id, Article.group_id.is_(None))
        )
        .order_by(distance)
        .limit(30)
        .all()
    )

    sources: list[dict[str, Any]] = []
    seen_articles: set[uuid.UUID] = set()
    for chunk, article, _dist in matches:
        if article.id in seen_articles:
            continue
        if len(seen_articles) >= CHAT_ARTICLE_LIMIT:
            continue
        seen_articles.add(article.id)
        folder_path = article.metadata_.get("folder_path") or "Root"
        sources.append(
            {
                "article_id": str(article.id),
                "title": article.title,
                "summary": article.summary or "",
                "excerpt": chunk.content[:600],
                "folder_path": folder_path,
            }
        )
        if len(sources) >= 8:
            break

    if not sources and matches:
        for chunk, article, _dist in matches[:CHAT_ARTICLE_LIMIT]:
            if article.id in seen_articles:
                continue
            seen_articles.add(article.id)
            folder_path = article.metadata_.get("folder_path") or "Root"
            sources.append(
                {
                    "article_id": str(article.id),
                    "title": article.title,
                    "summary": article.summary or "",
                    "excerpt": chunk.content[:600],
                    "folder_path": folder_path,
                }
            )
        sources = sources[: min(8, len(sources))]

    for index, source in enumerate(sources, start=1):
        source["index"] = index

    context = _build_cited_context(sources) if sources else ""
    inference = _get_user_inference_config(db, current_user)
    if inference_id:
        override = db.get(UserInferenceConfig, inference_id)
        if override and override.user_id == current_user.id:
            inference = {
                "provider": override.provider,
                "api_key": override.api_key,
                "base_url": override.base_url or "",
                "title": "simplr",
            }
    provider = _provider_for_inference(inference)
    answer = answer_with_context(question, context, provider=provider, inference=inference)

    for source in sources:
        references.append(
            {
                "id": source["article_id"],
                "title": source["title"],
                "summary": source.get("summary") or "",
                "excerpt": source["excerpt"],
                "index": source["index"],
            }
        )

    return templates.TemplateResponse(
        "chat.html",
        _template_context(
            request,
            current_user,
            session,
            question=question,
            answer=answer,
            references=references,
            error=error,
            scope=scope_name,
            selected_group=selected_group,
            groups=groups,
            inference_configs=inference_configs,
        ),
    )


@router.post("/chat/stream")
async def chat_stream(request: Request, db: Session = Depends(get_db)):
    current_user = _get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Login required")

    payload = await request.json()
    question = (payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    scope = payload.get("scope")
    group_id = payload.get("group_id")
    inference_id = payload.get("inference_id")
    model = payload.get("model")
    provider = payload.get("provider")
    temperature = payload.get("temperature")
    max_tokens = payload.get("max_tokens")
    ground = payload.get("ground", True)
    try:
        temperature = float(temperature) if temperature is not None else None
    except (TypeError, ValueError):
        temperature = None
    try:
        max_tokens = int(max_tokens) if max_tokens is not None else None
    except (TypeError, ValueError):
        max_tokens = None
    inference = _get_user_inference_config(db, current_user)
    if inference_id:
        override = db.get(UserInferenceConfig, inference_id)
        if override and override.user_id == current_user.id:
            inference = {
                "provider": override.provider,
                "api_key": override.api_key,
                "base_url": override.base_url or "",
                "title": "simplr",
            }
    provider = _provider_for_inference(inference) if provider is None else provider
    if provider == "openrouter" and not settings.openrouter_api_key and not inference:
        raise HTTPException(status_code=400, detail="OpenRouter is not configured")

    scope_name, selected_group = _resolve_scope(db, current_user, scope, group_id)

    stats_answer = _maybe_stats_answer(question, current_user, db)
    stream_headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    if stats_answer:
        async def event_stream():
            context_b64 = base64.b64encode("".encode("utf-8")).decode("utf-8")
            yield f"event: context\ndata: {json.dumps({'b64': context_b64})}\n\n"
            refs_b64 = base64.b64encode(json.dumps([]).encode("utf-8")).decode("utf-8")
            yield f"event: refs\ndata: {json.dumps({'b64': refs_b64})}\n\n"
            yield f"event: answer\ndata: {json.dumps(stats_answer)}\n\n"
            yield f"event: usage\ndata: {json.dumps({'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0})}\n\n"
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream", headers=stream_headers)

    session_id = _get_session_id(request)
    history = CHAT_MEMORY.get(session_id, [])
    history_text = _render_history(history)

    if not ground:
        async def event_stream():
            context_b64 = base64.b64encode("".encode("utf-8")).decode("utf-8")
            yield f"event: context\ndata: {json.dumps({'b64': context_b64})}\n\n"
            refs_b64 = base64.b64encode(json.dumps([]).encode("utf-8")).decode("utf-8")
            yield f"event: refs\ndata: {json.dumps({'b64': refs_b64})}\n\n"
            answer_parts: list[str] = []
            try:
                for token in stream_answer_freely(
                    question,
                    history_text,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    provider=provider,
                    inference=inference,
                ):
                    answer_parts.append(token)
                    yield f"event: answer\ndata: {json.dumps(token)}\n\n"
            except Exception as exc:
                yield f"event: error\ndata: {json.dumps(str(exc))}\n\n"
                yield "event: done\ndata: {}\n\n"
                return

            answer_text = "".join(answer_parts).strip()
            usage = estimate_free_usage(
                question,
                history_text,
                answer_text,
                model or settings.llm_model,
            )
            yield f"event: usage\ndata: {json.dumps(usage)}\n\n"
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer_text})
            CHAT_MEMORY[session_id] = history[-CHAT_MEMORY_LIMIT * 2 :]
            yield "event: done\ndata: {}\n\n"

        response = StreamingResponse(event_stream(), media_type="text/event-stream", headers=stream_headers)
        response.set_cookie("rag_session_id", session_id, max_age=60 * 60 * 6)
        return response

    try:
        query_embedding = embed_query(question)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    distance = Chunk.embedding.cosine_distance(query_embedding).label("distance")
    ts_query = func.plainto_tsquery("english", question)
    ts_rank = func.ts_rank(
        func.to_tsvector("english", Article.title + " " + Article.summary + " " + Article.content_text),
        ts_query,
    ).label("ts_rank")
    matches = (
        db.query(Chunk, Article, distance, ts_rank)
        .join(Article, Article.id == Chunk.article_id)
        .filter(
            Article.group_id == selected_group.id if scope_name == "group" and selected_group else
            and_(Article.owner_id == current_user.id, Article.group_id.is_(None))
        )
        .order_by(distance)
        .limit(18)
        .all()
    )

    references: list[dict[str, str]] = []
    sources: list[dict[str, Any]] = []
    seen_articles: set[uuid.UUID] = set()
    for chunk, article, dist, lex_rank in matches:
        semantic = 0.0
        if dist is not None:
            semantic = max(0.0, 1.0 - float(dist))
        lexical = float(lex_rank or 0.0)

        if semantic < CHAT_MIN_SEMANTIC and lexical < CHAT_MIN_LEXICAL:
            continue

        if article.id in seen_articles:
            continue
        if len(seen_articles) >= CHAT_ARTICLE_LIMIT:
            continue
        seen_articles.add(article.id)
        folder_path = article.metadata_.get("folder_path") or "Root"
        sources.append(
            {
                "article_id": str(article.id),
                "title": article.title,
                "summary": article.summary or "",
                "excerpt": chunk.content[:600],
                "folder_path": folder_path,
            }
        )
        if len(sources) >= 8:
            break

    if not sources and matches:
        for chunk, article, _dist, _lex_rank in matches[:CHAT_ARTICLE_LIMIT]:
            if article.id in seen_articles:
                continue
            seen_articles.add(article.id)
            folder_path = article.metadata_.get("folder_path") or "Root"
            sources.append(
                {
                    "article_id": str(article.id),
                    "title": article.title,
                    "summary": article.summary or "",
                    "excerpt": chunk.content[:600],
                    "folder_path": folder_path,
                }
            )
        sources = sources[: min(8, len(sources))]

    for index, source in enumerate(sources, start=1):
        source["index"] = index
        references.append(
            {
                "id": source["article_id"],
                "title": source["title"],
                "summary": source.get("summary") or "",
                "excerpt": source["excerpt"],
                "index": source["index"],
            }
        )

    context = _build_cited_context(sources) if sources else ""

    async def event_stream():
        context_b64 = base64.b64encode(context.encode("utf-8")).decode("utf-8")
        yield f"event: context\ndata: {json.dumps({'b64': context_b64})}\n\n"
        refs_b64 = base64.b64encode(json.dumps(references).encode("utf-8")).decode("utf-8")
        yield f"event: refs\ndata: {json.dumps({'b64': refs_b64})}\n\n"
        answer_parts: list[str] = []
        try:
            for token in stream_answer_with_context(
                question,
                context,
                history_text,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                provider=provider,
                inference=inference,
            ):
                answer_parts.append(token)
                yield f"event: answer\ndata: {json.dumps(token)}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps(str(exc))}\n\n"
            yield "event: done\ndata: {}\n\n"
            return

        answer_text = "".join(answer_parts).strip()
        usage = estimate_rag_usage(
            question,
            context,
            history_text,
            answer_text,
            model or settings.llm_model,
        )
        yield f"event: usage\ndata: {json.dumps(usage)}\n\n"
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer_text})
        CHAT_MEMORY[session_id] = history[-CHAT_MEMORY_LIMIT * 2 :]
        yield "event: done\ndata: {}\n\n"

    response = StreamingResponse(event_stream(), media_type="text/event-stream", headers=stream_headers)
    response.set_cookie("rag_session_id", session_id, max_age=60 * 60 * 6)
    return response
