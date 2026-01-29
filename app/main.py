from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Optional

import markdown
from fastapi import BackgroundTasks, Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Article, Chunk
from app.schemas import ArticleRead, ArticleUpdate
from app.services.embeddings import embed_query
from app.services.ingest import ingest_email, ingest_email_job
from app.services.llm import answer_with_context, stream_answer_with_context
from app.services.mailersend import parse_mailersend_payload

app = FastAPI(title="RAG Email MVP")

logger = logging.getLogger("rag-email-mvp")

CATEGORIES = ["all", "support_tickets", "policies", "documentation", "projects", "other", "uncategorized"]

CHAT_MEMORY: dict[str, list[dict[str, str]]] = {}
CHAT_MEMORY_LIMIT = 6
CHAT_MIN_SEMANTIC = 0.78
CHAT_MIN_LEXICAL = 0.08

app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")


def render_markdown(text: Optional[str]) -> Markup:
    if not text:
        return Markup("")
    html = markdown.markdown(text, extensions=["extra", "nl2br", "sane_lists"])
    return Markup(html)


templates.env.filters["markdown"] = render_markdown

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


def _build_context(chunks: list[tuple[Chunk, Article]]) -> str:
    parts = []
    for chunk, article in chunks:
        parts.append(f"Title: {article.title}\nExcerpt: {chunk.content}")
    return "\n\n".join(parts)


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
    return RedirectResponse(url="/articles")


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
def list_articles(db: Session = Depends(get_db)):
    articles = db.query(Article).order_by(Article.created_at.desc()).all()
    return articles


@app.get("/api/articles/{article_id}", response_model=ArticleRead)
def get_article(article_id: str, db: Session = Depends(get_db)):
    article = db.get(Article, article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    return article


@app.put("/api/articles/{article_id}", response_model=ArticleRead)
def update_article(article_id: str, payload: ArticleUpdate, db: Session = Depends(get_db)):
    article = db.get(Article, article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")

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
def delete_article(article_id: str, db: Session = Depends(get_db)):
    article = db.get(Article, article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    db.delete(article)
    db.commit()
    return {"status": "deleted"}


@app.get("/api/search")
def search_articles(query: str, limit: int = 10, db: Session = Depends(get_db)):
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

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


@app.get("/articles", response_class=HTMLResponse)
def articles_ui(request: Request, db: Session = Depends(get_db), query: Optional[str] = None, category: Optional[str] = None):
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
            match_query = db.query(Chunk, Article, distance, ts_rank).join(Article, Article.id == Chunk.article_id)
            if selected_category != "all":
                match_query = match_query.filter(Article.metadata_["category"].astext == selected_category)
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
        list_query = db.query(Article)
        if selected_category != "all":
            list_query = list_query.filter(Article.metadata_["category"].astext == selected_category)
        articles = list_query.order_by(Article.created_at.desc()).all()

    return templates.TemplateResponse(
        "articles.html",
        {
            "request": request,
            "articles": articles,
            "query": query or "",
            "category": selected_category,
            "categories": CATEGORIES,
            "results": results,
            "error": error,
        },
    )


@app.get("/chat", response_class=HTMLResponse)
def chat_ui(request: Request):
    session_id = _get_session_id(request)
    response = templates.TemplateResponse(
        "chat.html",
        {"request": request, "question": "", "answer": None, "references": [], "error": None},
    )
    response.set_cookie("rag_session_id", session_id, max_age=60 * 60 * 6)
    return response


@app.post("/chat", response_class=HTMLResponse)
def chat_submit(request: Request, question: str = Form(...), db: Session = Depends(get_db)):
    error = None
    answer = None
    references = []

    try:
        query_embedding = embed_query(question)
    except RuntimeError as exc:
        error = str(exc)
        return templates.TemplateResponse(
            "chat.html",
            {"request": request, "question": question, "answer": None, "references": [], "error": error},
        )

    distance = Chunk.embedding.cosine_distance(query_embedding).label("distance")
    matches = (
        db.query(Chunk, Article, distance)
        .join(Article, Article.id == Chunk.article_id)
        .order_by(distance)
        .limit(8)
        .all()
    )

    context_chunks = [(chunk, article) for chunk, article, _dist in matches]
    context = _build_context(context_chunks)
    answer = answer_with_context(question, context)

    seen = set()
    for chunk, article in context_chunks:
        if article.id in seen:
            continue
        seen.add(article.id)
        references.append({"id": str(article.id), "title": article.title})

    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "question": question,
            "answer": answer,
            "references": references,
            "error": error,
        },
    )


@app.post("/chat/stream")
async def chat_stream(request: Request, db: Session = Depends(get_db)):
    payload = await request.json()
    question = (payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    session_id = _get_session_id(request)
    history = CHAT_MEMORY.get(session_id, [])
    history_text = _render_history(history)

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
        .order_by(distance)
        .limit(18)
        .all()
    )

    references = []
    context_chunks: list[tuple[Chunk, Article]] = []
    per_article_counts: dict[str, int] = {}
    for chunk, article, dist, lex_rank in matches:
        semantic = 0.0
        if dist is not None:
            semantic = max(0.0, 1.0 - float(dist))
        lexical = float(lex_rank or 0.0)

        if semantic < CHAT_MIN_SEMANTIC and lexical < CHAT_MIN_LEXICAL:
            continue

        article_key = str(article.id)
        if article_key not in per_article_counts:
            if len(references) >= 3:
                continue
            references.append({"id": article_key, "title": article.title})
            per_article_counts[article_key] = 0

        if per_article_counts[article_key] >= 2:
            continue
        per_article_counts[article_key] += 1
        context_chunks.append((chunk, article))

    context = _build_context(context_chunks) if context_chunks else ""

    async def event_stream():
        yield f"event: refs\\ndata: {json.dumps(references)}\\n\\n"
        answer_parts: list[str] = []
        for token in stream_answer_with_context(question, context, history_text):
            answer_parts.append(token)
            yield f"event: answer\\ndata: {json.dumps(token)}\\n\\n"

        answer_text = "".join(answer_parts).strip()
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer_text})
        CHAT_MEMORY[session_id] = history[-CHAT_MEMORY_LIMIT * 2 :]
        yield "event: done\ndata: {}\n\n"

    response = StreamingResponse(event_stream(), media_type="text/event-stream")
    response.set_cookie("rag_session_id", session_id, max_age=60 * 60 * 6)
    return response


@app.get("/articles/{article_id}", response_class=HTMLResponse)
def article_detail(article_id: str, request: Request, db: Session = Depends(get_db)):
    article = db.get(Article, article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    return templates.TemplateResponse(
        "article_detail.html", {"request": request, "article": article}
    )


@app.get("/articles/{article_id}/edit", response_class=HTMLResponse)
def article_edit(article_id: str, request: Request, db: Session = Depends(get_db)):
    article = db.get(Article, article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    return templates.TemplateResponse(
        "article_edit.html", {"request": request, "article": article}
    )


@app.post("/articles/{article_id}/edit")
def article_edit_submit(
    article_id: str,
    title: str = Form(...),
    summary: str = Form(...),
    metadata_json: str = Form("{}"),
    db: Session = Depends(get_db),
):
    article = db.get(Article, article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")

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
def article_delete(article_id: str, db: Session = Depends(get_db)):
    article = db.get(Article, article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    db.delete(article)
    db.commit()
    return RedirectResponse(url="/articles", status_code=303)
