from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import and_, func, or_
from sqlalchemy.orm import Session

from app.config import settings
from app.constants import CATEGORIES
from app.db import get_db
from app.deps import (
    _get_session_and_user,
    _group_ids_for_user,
    _human_bytes,
    _keyword_boost,
    _keyword_terms,
    _require_user_api,
    _resolve_scope,
    _template_context,
    _user_groups,
)
from app.models import Article, Chunk, Source
from app.services.embeddings import embed_query
from app.web import templates


router = APIRouter()


@router.get("/api/search")
def search_articles(request: Request, query: str, limit: int = 10, db: Session = Depends(get_db)):
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    user = _require_user_api(request, db)

    try:
        query_embedding = embed_query(query)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    terms = _keyword_terms(query)
    distance = Chunk.embedding.cosine_distance(query_embedding).label("distance")
    ts_query = func.plainto_tsquery("english", query)
    ts_rank = func.ts_rank(
        func.to_tsvector("english", Article.title + " " + Article.summary + " " + Article.content_text),
        ts_query,
    ).label("ts_rank")
    group_ids = _group_ids_for_user(db, user)
    access_filter = or_(
        and_(Article.owner_id == user.id, Article.group_id.is_(None)),
        Article.group_id.in_(list(group_ids) or [uuid.UUID(int=0)]),
    )
    results = (
        db.query(Chunk, Article, distance, ts_rank)
        .join(Article, Article.id == Chunk.article_id)
        .filter(access_filter)
        .order_by(distance)
        .limit(limit * 3)
        .all()
    )

    scored: dict[str, dict[str, Any]] = {}
    for chunk, article, dist, lex_rank in results:
        base_similarity = 0.0
        if dist is not None:
            base_similarity = max(0.0, 1.0 - float(dist))
        boost = _keyword_boost(terms, f"{article.title} {article.summary} {chunk.content}")
        lexical = float(lex_rank or 0.0)
        hybrid = min(1.0, base_similarity + boost + min(0.25, lexical))

        existing = scored.get(str(article.id))
        if not existing or hybrid > existing["similarity"]:
            scored[str(article.id)] = {
                "article_id": str(article.id),
                "title": article.title,
                "summary": article.summary,
                "chunk": chunk.content,
                "category": article.metadata_.get("category", "uncategorized"),
                "similarity": hybrid,
                "semantic_score": base_similarity,
                "lexical_score": lexical,
            }

    payload = sorted(scored.values(), key=lambda item: item["similarity"], reverse=True)[:limit]
    return payload


@router.get("/insights", response_class=HTMLResponse)
def insights_ui(
    request: Request,
    db: Session = Depends(get_db),
    query: Optional[str] = None,
    category: Optional[str] = None,
    scope: Optional[str] = None,
    group_id: Optional[str] = None,
):
    session, current_user = _get_session_and_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)

    error = None
    results = []
    articles = []

    scope_name, selected_group = _resolve_scope(db, current_user, scope, group_id)
    groups = _user_groups(db, current_user)
    if scope_name == "group" and selected_group:
        article_scope_filter = Article.group_id == selected_group.id
    else:
        article_scope_filter = and_(Article.owner_id == current_user.id, Article.group_id.is_(None))

    selected_category = category if category in CATEGORIES else "all"

    if query:
        try:
            query_embedding = embed_query(query)
        except RuntimeError as exc:
            error = str(exc)
            like_query = f"%{query.strip()}%"
            fallback_query = db.query(Article).filter(article_scope_filter).filter(
                or_(
                    Article.title.ilike(like_query),
                    Article.summary.ilike(like_query),
                    Article.content_text.ilike(like_query),
                )
            )
            if selected_category != "all":
                fallback_query = fallback_query.filter(
                    or_(
                        Article.metadata_["category"].astext == selected_category,
                        Article.metadata_["categories"].contains([selected_category]),
                    )
                )
            fallback_articles = fallback_query.order_by(Article.updated_at.desc()).limit(20).all()
            results = [
                {
                    "article": article,
                    "snippet": (article.summary or article.content_text or "")[:240],
                    "similarity": None,
                    "semantic_score": None,
                    "lexical_score": None,
                }
                for article in fallback_articles
            ]
        else:
            terms = _keyword_terms(query)
            distance = Chunk.embedding.cosine_distance(query_embedding).label("distance")
            ts_query = func.plainto_tsquery("english", query)
            ts_rank = func.ts_rank(
                func.to_tsvector("english", Article.title + " " + Article.summary + " " + Article.content_text),
                ts_query,
            ).label("ts_rank")
            match_query = (
                db.query(Chunk, Article, distance, ts_rank)
                .join(Article, Article.id == Chunk.article_id)
                .filter(article_scope_filter)
            )
            if selected_category != "all":
                match_query = match_query.filter(
                    or_(
                        Article.metadata_["category"].astext == selected_category,
                        Article.metadata_["categories"].contains([selected_category]),
                    )
                )
            matches = (
                match_query
                .order_by(distance)
                .limit(60)
                .all()
            )
            scored: dict[str, dict[str, Any]] = {}
            for chunk, article, dist, lex_rank in matches:
                base_similarity = 0.0
                if dist is not None:
                    base_similarity = max(0.0, 1.0 - float(dist))
                boost = _keyword_boost(terms, f"{article.title} {article.summary} {chunk.content}")
                lexical = float(lex_rank or 0.0)
                similarity = min(1.0, base_similarity + boost + min(0.25, lexical))

                existing = scored.get(str(article.id))
                if not existing or similarity > existing["similarity"]:
                    scored[str(article.id)] = {
                        "article": article,
                        "snippet": chunk.content[:240],
                        "similarity": similarity,
                        "semantic_score": base_similarity,
                        "lexical_score": lexical,
                    }
            results = sorted(scored.values(), key=lambda item: item["similarity"], reverse=True)[:20]

            if not results:
                like_query = f"%{query.strip()}%"
                fallback_query = db.query(Article).filter(article_scope_filter).filter(
                    or_(
                        Article.title.ilike(like_query),
                        Article.summary.ilike(like_query),
                        Article.content_text.ilike(like_query),
                    )
                )
                if selected_category != "all":
                    fallback_query = fallback_query.filter(
                        or_(
                            Article.metadata_["category"].astext == selected_category,
                            Article.metadata_["categories"].contains([selected_category]),
                        )
                    )
                fallback_articles = fallback_query.order_by(Article.updated_at.desc()).limit(20).all()
                results = [
                    {
                        "article": article,
                        "snippet": (article.summary or article.content_text or "")[:240],
                        "similarity": None,
                        "semantic_score": None,
                        "lexical_score": None,
                    }
                    for article in fallback_articles
                ]
    else:
        list_query = db.query(Article).filter(article_scope_filter)
        if selected_category != "all":
            list_query = list_query.filter(
                or_(
                    Article.metadata_["category"].astext == selected_category,
                    Article.metadata_["categories"].contains([selected_category]),
                )
            )
        articles = list_query.order_by(Article.created_at.desc()).all()

    article_count = db.query(func.count(Article.id)).filter(article_scope_filter).scalar() or 0
    today = datetime.utcnow().date()
    start_date = today - timedelta(days=6)
    daily_counts = (
        db.query(func.date(Article.created_at).label("day"), func.count(Article.id))
        .filter(article_scope_filter, Article.created_at >= start_date)
        .group_by("day")
        .order_by("day")
        .all()
    )
    daily_counts_map = {day: count for day, count in daily_counts}
    article_activity = []
    for offset in range(7):
        day = start_date + timedelta(days=offset)
        article_activity.append(
            {
                "label": day.strftime("%b %d"),
                "count": daily_counts_map.get(day, 0),
                "date": day.isoformat(),
            }
        )
    activity_max = max([item["count"] for item in article_activity], default=0)
    text_bytes = (
        db.query(func.coalesce(func.sum(func.length(Article.content_text)), 0))
        .filter(article_scope_filter)
        .scalar()
        or 0
    )
    chunk_bytes = (
        db.query(func.coalesce(func.sum(func.length(Chunk.content)), 0))
        .join(Article, Article.id == Chunk.article_id)
        .filter(article_scope_filter)
        .scalar()
        or 0
    )
    attachment_bytes = 0
    sources = (
        db.query(Source.source_uri)
        .join(Article, Article.id == Source.article_id)
        .filter(article_scope_filter, Source.source_type == "attachment")
        .all()
    )
    for (uri,) in sources:
        if not uri:
            continue
        try:
            attachment_bytes += os.path.getsize(uri)
        except OSError:
            continue
    user_data_size = _human_bytes(int(text_bytes + chunk_bytes + attachment_bytes))

    return templates.TemplateResponse(
        "articles.html",
        _template_context(
            request,
            current_user,
            session,
            article_count=article_count,
            user_data_size=user_data_size,
            llm_model=settings.llm_model,
            embedding_model=settings.embedding_model,
            article_activity=article_activity,
            article_activity_max=max(1, activity_max),
            scope=scope_name,
            selected_group=selected_group,
            groups=groups,
            articles=articles,
            query=query or "",
            category=selected_category,
            categories=CATEGORIES,
            results=results,
            error=error,
        ),
    )
