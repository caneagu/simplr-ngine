from __future__ import annotations

import base64
import json
import logging
import math
import re
import uuid
import urllib.request
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import markdown
from fastapi import BackgroundTasks, Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup
from sqlalchemy import func, or_, text
from sqlalchemy.orm import Session

from app.config import settings
from app.constants import CATEGORIES
from app.db import get_db
from app.models import Article, ArticleVersion, Chunk, Folder, MagicLinkToken, SessionToken, Source, User
from app.schemas import ArticleRead, ArticleUpdate
from app.services.auth import (
    generate_token,
    hash_token,
    magiclink_expiry,
    normalize_email,
    now_utc,
    session_expiry,
)
from app.services.embeddings import embed_query
from app.services.emailer import send_magic_link
from app.services.ingest import ingest_email, ingest_email_job
from app.services.llm import (
    answer_freely,
    answer_with_context,
    estimate_free_usage,
    estimate_rag_usage,
    stream_answer_freely,
    stream_answer_with_context,
)
from app.services.mailersend import parse_mailersend_payload

app = FastAPI(title="RAG Email MVP")

logger = logging.getLogger("rag-email-mvp")

CHAT_MEMORY: dict[str, list[dict[str, str]]] = {}
CHAT_MEMORY_LIMIT = 6
CHAT_MIN_SEMANTIC = 0.78
CHAT_MIN_LEXICAL = 0.08
SESSION_COOKIE = "rag_session"
CHAT_ARTICLE_LIMIT = 3

app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")


def render_markdown(text: Optional[str]) -> Markup:
    if not text:
        return Markup("")
    html = markdown.markdown(text, extensions=["extra", "nl2br", "sane_lists"])
    return Markup(html)


templates.env.filters["markdown"] = render_markdown


def format_dt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        candidate = value.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(candidate)
        except ValueError:
            return value
    else:
        return str(value)
    return dt.strftime("%d:%m:%Y %H:%M:%S")


templates.env.filters["format_dt"] = format_dt


def _get_current_session(request: Request, db: Session) -> Optional[SessionToken]:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    token_hash = hash_token(token)
    session = db.query(SessionToken).filter(SessionToken.token_hash == token_hash).first()
    if not session:
        return None
    if session.expires_at < now_utc():
        db.delete(session)
        db.commit()
        return None
    session.last_seen_at = now_utc()
    db.add(session)
    db.commit()
    return session


def _get_session_and_user(request: Request, db: Session) -> tuple[Optional[SessionToken], Optional[User]]:
    session = _get_current_session(request, db)
    return session, session.user if session else None


def _get_current_user(request: Request, db: Session) -> Optional[User]:
    session = _get_current_session(request, db)
    return session.user if session else None


def _require_user_api(request: Request, db: Session) -> User:
    user = _get_current_user(request, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Login required")
    return user


def _ensure_user_ui(request: Request, db: Session) -> Optional[User]:
    return _get_current_user(request, db)


def _session_expires_in_days(session: Optional[SessionToken]) -> Optional[int]:
    if not session:
        return None
    delta = session.expires_at - now_utc()
    seconds = max(0, int(delta.total_seconds()))
    return max(0, math.ceil(seconds / 86400))


def _template_context(
    request: Request,
    current_user: Optional[User],
    session: Optional[SessionToken],
    **kwargs: Any,
) -> dict[str, Any]:
    context = {
        "request": request,
        "current_user": current_user,
        "session_expires_in_days": _session_expires_in_days(session),
    }
    context.update(kwargs)
    return context


def _detect_category_from_question(question: str) -> Optional[str]:
    normalized = question.lower()
    for category in CATEGORIES:
        if category == "all":
            continue
        label = category.replace("_", " ")
        if label in normalized or category in normalized:
            return category
    return None


def _maybe_stats_answer(question: str, current_user: User, db: Session) -> Optional[str]:
    normalized = question.lower()
    if not any(term in normalized for term in ["how many", "count", "number of", "total"]):
        return None

    if "pdf" in normalized:
        pdf_count = (
            db.query(func.count(Source.id))
            .join(Article, Article.id == Source.article_id)
            .filter(
                Article.owner_id == current_user.id,
                Source.source_type == "attachment",
                func.lower(Source.metadata_["content_type"].astext).like("%pdf%"),
            )
            .scalar()
            or 0
        )
        return f"You have {pdf_count} PDF attachment(s) in the knowledge base."

    if "report" in normalized:
        report_count = (
            db.query(func.count(Article.id))
            .filter(Article.owner_id == current_user.id, Article.metadata_["doc_type"].astext == "report")
            .scalar()
            or 0
        )
        return f"You have {report_count} report article(s) in the knowledge base."

    category = _detect_category_from_question(normalized)
    if category:
        category_count = (
            db.query(func.count(Article.id))
            .filter(
                Article.owner_id == current_user.id,
                or_(
                    Article.metadata_["category"].astext == category,
                    Article.metadata_["categories"].contains([category]),
                ),
            )
            .scalar()
            or 0
        )
        label = category.replace("_", " ")
        return f"You have {category_count} article(s) in the “{label}” category."

    if "article" in normalized or "document" in normalized:
        article_count = (
            db.query(func.count(Article.id)).filter(Article.owner_id == current_user.id).scalar() or 0
        )
        return f"You have {article_count} article(s) in the knowledge base."

    if "attachment" in normalized or "file" in normalized:
        attachment_count = (
            db.query(func.count(Source.id))
            .join(Article, Article.id == Source.article_id)
            .filter(Article.owner_id == current_user.id, Source.source_type == "attachment")
            .scalar()
            or 0
        )
        return f"You have {attachment_count} attachment(s) stored in the knowledge base."

    return None

def _keyword_terms(query: str) -> list[str]:
    terms = [t for t in re.split(r"\\W+", query.lower()) if len(t) >= 3]
    seen = set()
    deduped = []
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        deduped.append(term)
    return deduped


def _keyword_boost(terms: list[str], text: str) -> float:
    if not terms:
        return 0.0
    haystack = text.lower()
    matches = sum(1 for term in terms if term in haystack)
    return min(0.15, 0.03 * matches)


def _order_articles(article_ids: list[uuid.UUID], articles: list[Article]) -> list[Article]:
    by_id = {article.id: article for article in articles}
    return [by_id[article_id] for article_id in article_ids if article_id in by_id]


def _build_context(articles: list[Article]) -> str:
    parts = []
    for article in articles:
        summary = article.summary or ""
        content = article.content_text or ""
        folder_path = article.metadata_.get("folder_path") or "Root"
        parts.append(
            "\n".join(
                [
                    f"Title: {article.title}",
                    f"Folder: {folder_path}",
                    "Summary:",
                    summary,
                    "Full content:",
                    content,
                ]
            )
        )
    return "\n\n".join(parts)


def _build_cited_context(sources: list[dict[str, Any]]) -> str:
    parts = ["Sources:"]
    for source in sources:
        parts.append(
            "\n".join(
                [
                    f"[{source['index']}] Title: {source['title']}",
                    f"Folder: {source.get('folder_path') or 'Root'}",
                    "Summary:",
                    source.get("summary") or "",
                    "Excerpt:",
                    source["excerpt"],
                ]
            )
        )
    return "\n\n".join(parts)


def _human_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def _build_folder_tree(folders: list[Folder]) -> list[dict[str, Any]]:
    by_parent: dict[Optional[uuid.UUID], list[Folder]] = {}
    for folder in folders:
        by_parent.setdefault(folder.parent_id, []).append(folder)
    for items in by_parent.values():
        items.sort(key=lambda f: f.name.lower())

    def build(parent_id: Optional[uuid.UUID]) -> list[dict[str, Any]]:
        nodes = []
        for folder in by_parent.get(parent_id, []):
            nodes.append(
                {
                    "id": folder.id,
                    "name": folder.name,
                    "children": build(folder.id),
                }
            )
        return nodes

    return build(None)


def _folder_paths_map(folders: list[Folder]) -> dict[uuid.UUID, str]:
    by_id = {folder.id: folder for folder in folders}
    cache: dict[uuid.UUID, str] = {}

    def build_path(folder: Folder, seen: set[uuid.UUID]) -> str:
        if folder.id in cache:
            return cache[folder.id]
        if folder.id in seen:
            return folder.name
        seen.add(folder.id)
        if folder.parent_id and folder.parent_id in by_id:
            parent_path = build_path(by_id[folder.parent_id], seen)
            path = f"{parent_path}/{folder.name}"
        else:
            path = folder.name
        cache[folder.id] = path
        return path

    for folder in folders:
        build_path(folder, set())
    return cache


def _resolve_folder_path(db: Session, folder_id: Optional[uuid.UUID]) -> Optional[str]:
    if not folder_id:
        return None
    folder = db.get(Folder, folder_id)
    if not folder:
        return None
    folders = db.query(Folder).filter(Folder.owner_id == folder.owner_id).all()
    paths = _folder_paths_map(folders)
    return paths.get(folder_id)


def _sync_article_folder_metadata(db: Session, owner_id: uuid.UUID) -> None:
    folders = db.query(Folder).filter(Folder.owner_id == owner_id).all()
    folder_paths = _folder_paths_map(folders)
    articles = db.query(Article).filter(Article.owner_id == owner_id).all()
    for article in articles:
        if article.metadata_ is None:
            article.metadata_ = {}
        if article.folder_id and article.folder_id in folder_paths:
            article.metadata_["folder_id"] = str(article.folder_id)
            article.metadata_["folder_path"] = folder_paths[article.folder_id]
        else:
            article.metadata_["folder_id"] = None
            article.metadata_["folder_path"] = "Root"
        db.add(article)


def _get_session_id(request: Request) -> str:
    existing = request.cookies.get("rag_session_id")
    if existing:
        return existing
    return str(uuid.uuid4())


def _render_history(messages: list[dict[str, str]]) -> str:
    lines = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)


@app.get("/", response_class=HTMLResponse)
def root() -> RedirectResponse:
    return RedirectResponse(url="/insights")


@app.get("/login", response_class=HTMLResponse)
def login_ui(request: Request, db: Session = Depends(get_db)):
    session, current_user = _get_session_and_user(request, db)
    if current_user:
        return RedirectResponse(url="/insights", status_code=303)
    return templates.TemplateResponse(
        "login.html",
        _template_context(request, None, session, message=None, error=None, resend_email=None),
    )


@app.post("/login", response_class=HTMLResponse)
def login_request(request: Request, email: str = Form(...), db: Session = Depends(get_db)):
    normalized = normalize_email(email)
    user = db.query(User).filter(User.email == normalized).first()
    if user:
        token = generate_token()
        token_hash = hash_token(token)
        db.add(
            MagicLinkToken(
                user_id=user.id,
                token_hash=token_hash,
                expires_at=magiclink_expiry(),
            )
        )
        db.commit()
        magic_link = f"{settings.app_base_url.rstrip('/')}/auth/callback?token={token}"
        try:
            send_magic_link(normalized, magic_link)
        except RuntimeError as exc:
            return templates.TemplateResponse(
                "login.html",
                _template_context(request, None, None, message=None, error=str(exc), resend_email=normalized),
            )

    return templates.TemplateResponse(
        "login.html",
        _template_context(
            request,
            None,
            None,
            message="If that email has articles, a sign-in link is on the way.",
            error=None,
            resend_email=normalized,
        ),
    )


@app.get("/auth/callback", response_class=HTMLResponse)
def auth_callback(request: Request, token: str, db: Session = Depends(get_db)):
    token_hash = hash_token(token)
    magic_link = (
        db.query(MagicLinkToken)
        .filter(MagicLinkToken.token_hash == token_hash)
        .first()
    )
    if not magic_link or magic_link.expires_at < now_utc() or magic_link.used_at:
        return templates.TemplateResponse(
            "login.html",
            _template_context(
                request,
                None,
                None,
                message=None,
                error="This sign-in link is invalid or expired.",
                resend_email=None,
            ),
        )

    magic_link.used_at = now_utc()
    db.add(magic_link)
    session_token = generate_token()
    db.add(
        SessionToken(
            user_id=magic_link.user_id,
            token_hash=hash_token(session_token),
            expires_at=session_expiry(),
        )
    )
    db.commit()

    response = RedirectResponse(url="/insights", status_code=303)
    response.set_cookie(
        SESSION_COOKIE,
        session_token,
        httponly=True,
        secure=settings.cookie_secure,
        max_age=60 * 60 * 24 * settings.session_days,
        samesite="lax",
    )
    return response


@app.post("/logout")
def logout(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get(SESSION_COOKIE)
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE)
    if not token:
        return response
    token_hash = hash_token(token)
    session = db.query(SessionToken).filter(SessionToken.token_hash == token_hash).first()
    if session:
        db.delete(session)
        db.commit()
    return response


@app.get("/api/openrouter/models")
def list_openrouter_models():
    if not settings.openrouter_api_key:
        raise HTTPException(status_code=400, detail="OpenRouter is not configured")
    req = urllib.request.Request(
        f"{settings.openrouter_base_url}/models",
        headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    models = [item.get("id") for item in payload.get("data", []) if item.get("id")]
    return {"models": sorted(models)}


@app.post("/webhooks/mailersend", status_code=202)
def mailersend_webhook(
    payload: dict[str, Any],
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    inbound = parse_mailersend_payload(payload)
    if not inbound.text and not inbound.attachments:
        data = payload.get("data") or payload.get("message") or {}
        payload_keys = sorted(payload.keys())
        data_keys = sorted(data.keys()) if isinstance(data, dict) else []
        logger.warning("MailerSend payload missing content. keys=%s data_keys=%s", payload_keys, data_keys)
        return {
            "status": "ignored",
            "detail": {"error": "No content in payload", "payload_keys": payload_keys, "data_keys": data_keys},
        }

    if inbound.message_id:
        exists = (
            db.query(Article)
            .filter(Article.metadata_["message_id"].astext == inbound.message_id)
            .first()
        )
        if exists:
            return {"status": "duplicate", "article_id": str(exists.id)}

    if inbound.inbound_id:
        exists = (
            db.query(Article)
            .filter(Article.metadata_["inbound_id"].astext == inbound.inbound_id)
            .first()
        )
        if exists:
            return {"status": "duplicate", "article_id": str(exists.id)}

    background_tasks.add_task(ingest_email_job, inbound)
    return {"status": "accepted"}


@app.get("/api/articles", response_model=list[ArticleRead])
def list_articles(request: Request, db: Session = Depends(get_db)):
    user = _require_user_api(request, db)
    articles = (
        db.query(Article)
        .filter(Article.owner_id == user.id)
        .order_by(Article.created_at.desc())
        .all()
    )
    return articles


@app.get("/api/articles/{article_id}", response_model=ArticleRead)
def get_article(article_id: str, request: Request, db: Session = Depends(get_db)):
    user = _require_user_api(request, db)
    article = db.get(Article, article_id)
    if not article or article.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Article not found")
    return article


@app.put("/api/articles/{article_id}", response_model=ArticleRead)
def update_article(article_id: str, payload: ArticleUpdate, request: Request, db: Session = Depends(get_db)):
    user = _require_user_api(request, db)
    article = db.get(Article, article_id)
    if not article or article.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Article not found")

    version_number = len(article.versions) + 1
    db.add(
        ArticleVersion(
            article_id=article.id,
            version=version_number,
            title=article.title,
            summary=article.summary,
            content_text=article.content_text,
            metadata_=article.metadata_,
        )
    )

    if payload.title is not None:
        article.title = payload.title
    if payload.summary is not None:
        article.summary = payload.summary
    if payload.metadata is not None:
        article.metadata_ = payload.metadata

    db.add(article)
    db.commit()
    db.refresh(article)
    return article


@app.delete("/api/articles/{article_id}")
def delete_article(article_id: str, request: Request, db: Session = Depends(get_db)):
    user = _require_user_api(request, db)
    article = db.get(Article, article_id)
    if not article or article.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Article not found")
    db.delete(article)
    db.commit()
    return {"status": "deleted"}


@app.get("/api/search")
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
    results = (
        db.query(Chunk, Article, distance, ts_rank)
        .join(Article, Article.id == Chunk.article_id)
        .filter(Article.owner_id == user.id)
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


@app.get("/insights", response_class=HTMLResponse)
def insights_ui(request: Request, db: Session = Depends(get_db), query: Optional[str] = None, category: Optional[str] = None):
    session, current_user = _get_session_and_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)

    error = None
    results = []
    articles = []

    selected_category = category if category in CATEGORIES else "all"

    if query:
        try:
            query_embedding = embed_query(query)
        except RuntimeError as exc:
            error = str(exc)
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
                .filter(Article.owner_id == current_user.id)
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
    else:
        list_query = db.query(Article).filter(Article.owner_id == current_user.id)
        if selected_category != "all":
            list_query = list_query.filter(
                or_(
                    Article.metadata_["category"].astext == selected_category,
                    Article.metadata_["categories"].contains([selected_category]),
                )
            )
        articles = list_query.order_by(Article.created_at.desc()).all()

    article_count = db.query(func.count(Article.id)).filter(Article.owner_id == current_user.id).scalar() or 0
    text_bytes = (
        db.query(func.coalesce(func.sum(func.length(Article.content_text)), 0))
        .filter(Article.owner_id == current_user.id)
        .scalar()
        or 0
    )
    chunk_bytes = (
        db.query(func.coalesce(func.sum(func.length(Chunk.content)), 0))
        .join(Article, Article.id == Chunk.article_id)
        .filter(Article.owner_id == current_user.id)
        .scalar()
        or 0
    )
    attachment_bytes = 0
    sources = (
        db.query(Source.source_uri)
        .join(Article, Article.id == Source.article_id)
        .filter(Article.owner_id == current_user.id, Source.source_type == "attachment")
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
            articles=articles,
            query=query or "",
            category=selected_category,
            categories=CATEGORIES,
            results=results,
            error=error,
        ),
    )


@app.get("/chat", response_class=HTMLResponse)
def chat_ui(request: Request, db: Session = Depends(get_db)):
    session, current_user = _get_session_and_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)

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
        ),
    )
    response.set_cookie("rag_session_id", session_id, max_age=60 * 60 * 6)
    return response


@app.post("/chat", response_class=HTMLResponse)
def chat_submit(
    request: Request,
    question: str = Form(...),
    ground: bool = Form(True),
    db: Session = Depends(get_db),
):
    session, current_user = _get_session_and_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    error = None
    answer = None
    references = []

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
            ),
        )

    if not ground:
        answer = answer_freely(question)
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
        .filter(Article.owner_id == current_user.id)
        .order_by(distance)
        .limit(30)
        .all()
    )

    sources: list[dict[str, Any]] = []
    seen_articles: set[uuid.UUID] = set()
    for chunk, article, _dist in matches:
        if article.id not in seen_articles and len(seen_articles) >= CHAT_ARTICLE_LIMIT:
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
    answer = answer_with_context(question, context)

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
        ),
    )


@app.post("/chat/stream")
async def chat_stream(request: Request, db: Session = Depends(get_db)):
    current_user = _get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Login required")

    payload = await request.json()
    question = (payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
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
    if provider == "openrouter" and not settings.openrouter_api_key:
        raise HTTPException(status_code=400, detail="OpenRouter is not configured")

    stats_answer = _maybe_stats_answer(question, current_user, db)
    if stats_answer:
        async def event_stream():
            context_b64 = base64.b64encode("".encode("utf-8")).decode("utf-8")
            yield f"event: context\\ndata: {json.dumps({'b64': context_b64})}\\n\\n"
            refs_b64 = base64.b64encode(json.dumps([]).encode("utf-8")).decode("utf-8")
            yield f"event: refs\\ndata: {json.dumps({'b64': refs_b64})}\\n\\n"
            yield f"event: answer\\ndata: {json.dumps(stats_answer)}\\n\\n"
            yield f"event: usage\\ndata: {json.dumps({'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0})}\\n\\n"
            yield "event: done\\ndata: {}\\n\\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    session_id = _get_session_id(request)
    history = CHAT_MEMORY.get(session_id, [])
    history_text = _render_history(history)

    if not ground:
        async def event_stream():
            context_b64 = base64.b64encode("".encode("utf-8")).decode("utf-8")
            yield f"event: context\\ndata: {json.dumps({'b64': context_b64})}\\n\\n"
            refs_b64 = base64.b64encode(json.dumps([]).encode("utf-8")).decode("utf-8")
            yield f"event: refs\\ndata: {json.dumps({'b64': refs_b64})}\\n\\n"
            answer_parts: list[str] = []
            try:
                for token in stream_answer_freely(
                    question,
                    history_text,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    provider=provider,
                ):
                    answer_parts.append(token)
                    yield f"event: answer\\ndata: {json.dumps(token)}\\n\\n"
            except Exception as exc:
                yield f"event: error\\ndata: {json.dumps(str(exc))}\\n\\n"
                yield "event: done\\ndata: {}\\n\\n"
                return

            answer_text = "".join(answer_parts).strip()
            usage = estimate_free_usage(
                question,
                history_text,
                answer_text,
                model or settings.llm_model,
            )
            yield f"event: usage\\ndata: {json.dumps(usage)}\\n\\n"
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer_text})
            CHAT_MEMORY[session_id] = history[-CHAT_MEMORY_LIMIT * 2 :]
            yield "event: done\ndata: {}\n\n"

        response = StreamingResponse(event_stream(), media_type="text/event-stream")
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
        .filter(Article.owner_id == current_user.id)
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

        if article.id not in seen_articles and len(seen_articles) >= CHAT_ARTICLE_LIMIT:
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
        yield f"event: context\\ndata: {json.dumps({'b64': context_b64})}\\n\\n"
        refs_b64 = base64.b64encode(json.dumps(references).encode("utf-8")).decode("utf-8")
        yield f"event: refs\\ndata: {json.dumps({'b64': refs_b64})}\\n\\n"
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
            ):
                answer_parts.append(token)
                yield f"event: answer\\ndata: {json.dumps(token)}\\n\\n"
        except Exception as exc:
            yield f"event: error\\ndata: {json.dumps(str(exc))}\\n\\n"
            yield "event: done\\ndata: {}\\n\\n"
            return

        answer_text = "".join(answer_parts).strip()
        usage = estimate_rag_usage(
            question,
            context,
            history_text,
            answer_text,
            model or settings.llm_model,
        )
        yield f"event: usage\\ndata: {json.dumps(usage)}\\n\\n"
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer_text})
        CHAT_MEMORY[session_id] = history[-CHAT_MEMORY_LIMIT * 2 :]
        yield "event: done\ndata: {}\n\n"

    response = StreamingResponse(event_stream(), media_type="text/event-stream")
    response.set_cookie("rag_session_id", session_id, max_age=60 * 60 * 6)
    return response


@app.get("/articles/{article_id}", response_class=HTMLResponse)
def article_detail(article_id: str, request: Request, db: Session = Depends(get_db)):
    session, current_user = _get_session_and_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    article = db.get(Article, article_id)
    if not article or article.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Article not found")
    sources = (
        db.query(Source)
        .filter(Source.article_id == article.id, Source.source_type == "attachment")
        .all()
    )
    source_links = []
    for source in sources:
        name = (source.metadata_ or {}).get("original_filename") or source.source_name
        if not name and source.source_uri:
            name = Path(source.source_uri).name
        if not name:
            name = (source.metadata_ or {}).get("original_filename")
        if not name:
            name = "Attachment"
        content_type = (source.metadata_ or {}).get("content_type", "")
        if "." not in name and "pdf" in content_type.lower():
            name = f"{name}.pdf"
        source_links.append({"id": source.id, "name": name})
    folder_path = article.metadata_.get("folder_path") or "Root"
    folder_id = article.folder_id
    return templates.TemplateResponse(
        "article_detail.html",
        _template_context(
            request,
            current_user,
            session,
            article=article,
            sources=source_links,
            folder_path=folder_path,
            folder_id=str(folder_id) if folder_id else None,
        ),
    )


@app.get("/articles/{article_id}/edit", response_class=HTMLResponse)
def article_edit(article_id: str, request: Request, db: Session = Depends(get_db)):
    session, current_user = _get_session_and_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    article = db.get(Article, article_id)
    if not article or article.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Article not found")
    return templates.TemplateResponse(
        "article_edit.html", _template_context(request, current_user, session, article=article)
    )


@app.post("/articles/{article_id}/edit")
def article_edit_submit(
    article_id: str,
    request: Request,
    title: str = Form(...),
    summary: str = Form(...),
    metadata_json: str = Form("{}"),
    db: Session = Depends(get_db),
):
    session, current_user = _get_session_and_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    article = db.get(Article, article_id)
    if not article or article.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Article not found")

    version_number = len(article.versions) + 1
    db.add(
        ArticleVersion(
            article_id=article.id,
            version=version_number,
            title=article.title,
            summary=article.summary,
            content_text=article.content_text,
            metadata_=article.metadata_,
        )
    )

    try:
        metadata = json.loads(metadata_json) if metadata_json else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid metadata JSON")

    article.title = title
    article.summary = summary
    article.metadata_ = metadata
    db.add(article)
    db.commit()

    return RedirectResponse(url=f"/articles/{article_id}", status_code=303)


@app.post("/articles/{article_id}/delete")
def article_delete(article_id: str, request: Request, db: Session = Depends(get_db)):
    session, current_user = _get_session_and_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    article = db.get(Article, article_id)
    if not article or article.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Article not found")
    db.delete(article)
    db.commit()
    return RedirectResponse(url="/insights", status_code=303)


@app.get("/articles/{article_id}/sources/{source_id}")
def download_source(article_id: str, source_id: str, request: Request, db: Session = Depends(get_db)):
    _session, current_user = _get_session_and_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)

    article = db.get(Article, article_id)
    if not article or article.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Article not found")

    source = db.get(Source, source_id)
    if not source or source.article_id != article.id or source.source_type != "attachment":
        raise HTTPException(status_code=404, detail="Source not found")

    if not source.source_uri:
        raise HTTPException(status_code=404, detail="Source file missing")

    storage_root = Path(settings.storage_dir).resolve()
    attachments_root = (storage_root / "attachments").resolve()
    source_path = Path(source.source_uri).resolve()

    if attachments_root not in source_path.parents and source_path != attachments_root:
        raise HTTPException(status_code=404, detail="Source file invalid")
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Source file missing")

    filename = (source.metadata_ or {}).get("original_filename") or source.source_name or source_path.name
    if "." not in filename:
        content_type = (source.metadata_ or {}).get("content_type", "")
        if "pdf" in content_type or source_path.suffix.lower() == "":
            filename = f"{filename}.pdf"
    inline = request.query_params.get("inline") == "1"
    disposition = "inline" if inline else "attachment"
    headers = {"Content-Disposition": f'{disposition}; filename="{filename}"'}
    return FileResponse(source_path, filename=filename, headers=headers)


@app.get("/articles", response_class=HTMLResponse)
def files_ui(request: Request, db: Session = Depends(get_db), folder_id: Optional[str] = None):
    session, current_user = _get_session_and_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)

    folders = db.query(Folder).filter(Folder.owner_id == current_user.id).all()
    folder_tree = _build_folder_tree(folders)
    folder_path_by_id = _folder_paths_map(folders)

    active_folder_id: Optional[uuid.UUID] = None
    if folder_id:
        try:
            candidate = uuid.UUID(folder_id)
            folder = db.get(Folder, candidate)
            if folder and folder.owner_id == current_user.id:
                active_folder_id = candidate
        except ValueError:
            active_folder_id = None

    articles_query = db.query(Article).filter(Article.owner_id == current_user.id)
    if active_folder_id:
        articles_query = articles_query.filter(Article.folder_id == active_folder_id)
    else:
        articles_query = articles_query.filter(Article.folder_id.is_(None))
    articles = articles_query.order_by(Article.updated_at.desc()).all()
    version_counts: dict[uuid.UUID, int] = {}
    article_folder_paths: dict[uuid.UUID, str] = {}
    if articles:
        counts = (
            db.query(ArticleVersion.article_id, func.count(ArticleVersion.id))
            .filter(ArticleVersion.article_id.in_([article.id for article in articles]))
            .group_by(ArticleVersion.article_id)
            .all()
        )
        version_counts = {article_id: count for article_id, count in counts}
        for article in articles:
            if article.folder_id and article.folder_id in folder_path_by_id:
                article_folder_paths[article.id] = folder_path_by_id[article.folder_id]
            else:
                article_folder_paths[article.id] = "Root"

    return templates.TemplateResponse(
        "files.html",
        _template_context(
            request,
            current_user,
            session,
            folders=folder_tree,
            active_folder_id=str(active_folder_id) if active_folder_id else None,
            articles=articles,
            version_counts=version_counts,
            folder_paths=article_folder_paths,
        ),
    )


@app.post("/articles/folders")
def create_folder(
    request: Request,
    name: str = Form(...),
    parent_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    _session, current_user = _get_session_and_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)

    parent_uuid: Optional[uuid.UUID] = None
    if parent_id:
        try:
            parent_uuid = uuid.UUID(parent_id)
        except ValueError:
            parent_uuid = None
    if parent_uuid:
        parent = db.get(Folder, parent_uuid)
        if not parent or parent.owner_id != current_user.id:
            raise HTTPException(status_code=404, detail="Parent folder not found")

    folder_name = name.strip()
    if not folder_name:
        raise HTTPException(status_code=400, detail="Folder name is required")
    db.add(Folder(owner_id=current_user.id, parent_id=parent_uuid, name=folder_name))
    db.commit()
    _sync_article_folder_metadata(db, current_user.id)
    db.commit()
    return RedirectResponse(url="/articles", status_code=303)


@app.post("/articles/folders/{folder_id}/rename")
def rename_folder(folder_id: str, request: Request, name: str = Form(...), db: Session = Depends(get_db)):
    _session, current_user = _get_session_and_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    folder = db.get(Folder, folder_id)
    if not folder or folder.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Folder not found")
    folder.name = name.strip()
    db.add(folder)
    db.commit()
    _sync_article_folder_metadata(db, current_user.id)
    db.commit()
    return RedirectResponse(url="/articles", status_code=303)


@app.post("/articles/folders/{folder_id}/delete")
def delete_folder(folder_id: str, request: Request, db: Session = Depends(get_db)):
    _session, current_user = _get_session_and_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    folder = db.get(Folder, folder_id)
    if not folder or folder.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Folder not found")
    db.delete(folder)
    db.commit()
    _sync_article_folder_metadata(db, current_user.id)
    db.commit()
    return RedirectResponse(url="/articles", status_code=303)


@app.post("/articles/folders/{folder_id}/move")
async def move_folder(folder_id: str, request: Request, db: Session = Depends(get_db)):
    _session, current_user = _get_session_and_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Login required")

    payload = {}
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            payload = await request.json()
        except Exception:
            payload = {}
    else:
        try:
            form = await request.form()
            payload = dict(form)
        except Exception:
            payload = {}

    target_parent = payload.get("parent_id")
    folder = db.get(Folder, folder_id)
    if not folder or folder.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Folder not found")

    parent_uuid: Optional[uuid.UUID] = None
    if target_parent:
        try:
            parent_uuid = uuid.UUID(target_parent)
        except ValueError:
            parent_uuid = None

    if parent_uuid:
        parent = db.get(Folder, parent_uuid)
        if not parent or parent.owner_id != current_user.id:
            raise HTTPException(status_code=404, detail="Parent folder not found")
        if parent.id == folder.id:
            raise HTTPException(status_code=400, detail="Cannot move into itself")

    folder.parent_id = parent_uuid
    db.add(folder)
    db.commit()
    _sync_article_folder_metadata(db, current_user.id)
    db.commit()
    return {"status": "ok"}


@app.post("/articles/{article_id}/move")
async def move_article(article_id: str, request: Request, db: Session = Depends(get_db)):
    _session, current_user = _get_session_and_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Login required")

    folder_id = None
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            payload = await request.json()
            folder_id = payload.get("folder_id")
        except Exception:
            folder_id = None
    else:
        try:
            form = await request.form()
            folder_id = form.get("folder_id")
        except Exception:
            folder_id = None
    article = db.get(Article, article_id)
    if not article or article.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Article not found")

    folder_uuid: Optional[uuid.UUID] = None
    if folder_id:
        try:
            folder_uuid = uuid.UUID(folder_id)
        except ValueError:
            folder_uuid = None

    if folder_uuid:
        folder = db.get(Folder, folder_uuid)
        if not folder or folder.owner_id != current_user.id:
            raise HTTPException(status_code=404, detail="Folder not found")

    article.folder_id = folder_uuid
    if article.metadata_ is None:
        article.metadata_ = {}
    article.metadata_["folder_id"] = str(folder_uuid) if folder_uuid else None
    article.metadata_["folder_path"] = _resolve_folder_path(db, folder_uuid) or "Root"
    db.add(article)
    db.commit()
    return {"status": "ok"}


@app.get("/files")
def files_redirect() -> RedirectResponse:
    return RedirectResponse(url="/articles", status_code=302)
