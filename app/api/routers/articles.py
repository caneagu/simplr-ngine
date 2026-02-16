from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.deps import (
    _article_accessible,
    _effective_article_view,
    _get_session_and_user,
    _group_ids_for_user,
    _require_user_api,
    _template_context,
)
from app.models import Article, ArticleVersion, Source
from app.schemas import ArticleRead, ArticleUpdate
from app.services.auth import now_utc
from app.web import templates


router = APIRouter()


@router.get("/api/articles", response_model=list[ArticleRead])
def list_articles(request: Request, db: Session = Depends(get_db)):
    user = _require_user_api(request, db)
    articles = (
        db.query(Article)
        .filter(Article.owner_id == user.id)
        .order_by(Article.created_at.desc())
        .all()
    )
    return articles


@router.get("/api/articles/{article_id}", response_model=ArticleRead)
def get_article(article_id: str, request: Request, db: Session = Depends(get_db)):
    user = _require_user_api(request, db)
    article = db.get(Article, article_id)
    group_ids = _group_ids_for_user(db, user)
    if not article or not _article_accessible(article, user, group_ids):
        raise HTTPException(status_code=404, detail="Article not found")
    return article


@router.put("/api/articles/{article_id}", response_model=ArticleRead)
def update_article(article_id: str, payload: ArticleUpdate, request: Request, db: Session = Depends(get_db)):
    user = _require_user_api(request, db)
    article = db.get(Article, article_id)
    group_ids = _group_ids_for_user(db, user)
    if not article or not _article_accessible(article, user, group_ids):
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


@router.delete("/api/articles/{article_id}")
def delete_article(article_id: str, request: Request, db: Session = Depends(get_db)):
    user = _require_user_api(request, db)
    article = db.get(Article, article_id)
    group_ids = _group_ids_for_user(db, user)
    if not article or not _article_accessible(article, user, group_ids):
        raise HTTPException(status_code=404, detail="Article not found")
    db.delete(article)
    db.commit()
    return {"status": "deleted"}


@router.get("/articles/{article_id}", response_class=HTMLResponse)
def article_detail(article_id: str, request: Request, db: Session = Depends(get_db)):
    session, current_user = _get_session_and_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    article = db.get(Article, article_id)
    group_ids = _group_ids_for_user(db, current_user)
    if not article or not _article_accessible(article, current_user, group_ids):
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
    selected_group = article.group if article.group_id else None
    effective_view = _effective_article_view(article)
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
            scope="group" if selected_group else "personal",
            selected_group=selected_group,
            comments=effective_view["comments"],
            effective_summary=effective_view["summary"],
            effective_content=effective_view["content_text"],
            has_effective_override=effective_view["has_override"],
        ),
    )


@router.get("/articles/{article_id}/edit", response_class=HTMLResponse)
def article_edit(article_id: str, request: Request, db: Session = Depends(get_db)):
    _session, current_user = _get_session_and_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    return RedirectResponse(url=f"/articles/{article_id}", status_code=303)


@router.post("/articles/{article_id}/edit")
def article_edit_submit(
    article_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    return RedirectResponse(url=f"/articles/{article_id}", status_code=303)


@router.post("/articles/{article_id}/comments")
def article_comment_submit(
    article_id: str,
    request: Request,
    mode: str = Form(...),
    comment_text: str = Form(""),
    replacement_summary: str = Form(""),
    replacement_content: str = Form(""),
    db: Session = Depends(get_db),
):
    _session, current_user = _get_session_and_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    article = db.get(Article, article_id)
    group_ids = _group_ids_for_user(db, current_user)
    if not article or not _article_accessible(article, current_user, group_ids):
        raise HTTPException(status_code=404, detail="Article not found")

    selected_mode = mode.strip().lower()
    if selected_mode not in {"augment", "supersede"}:
        raise HTTPException(status_code=400, detail="Invalid comment mode")

    comment_text = comment_text.strip()
    replacement_summary = replacement_summary.strip()
    replacement_content = replacement_content.strip()
    if selected_mode == "augment" and not comment_text:
        raise HTTPException(status_code=400, detail="Comment text is required for augment mode")
    if selected_mode == "supersede" and not (replacement_summary or replacement_content or comment_text):
        raise HTTPException(
            status_code=400, detail="Provide replacement summary/content or context when superseding"
        )

    metadata = dict(article.metadata_ or {})
    comments = list(metadata.get("comments") or [])
    comments.append(
        {
            "id": str(uuid.uuid4()),
            "mode": selected_mode,
            "author_email": current_user.email,
            "created_at": now_utc().isoformat(),
            "comment_text": comment_text,
            "replacement_summary": replacement_summary,
            "replacement_content": replacement_content,
        }
    )
    metadata["comments"] = comments
    article.metadata_ = metadata
    db.add(article)
    db.commit()
    return RedirectResponse(url=f"/articles/{article_id}#comments", status_code=303)


@router.post("/articles/{article_id}/delete")
def article_delete(article_id: str, request: Request, db: Session = Depends(get_db)):
    _session, current_user = _get_session_and_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    article = db.get(Article, article_id)
    group_ids = _group_ids_for_user(db, current_user)
    if not article or not _article_accessible(article, current_user, group_ids):
        raise HTTPException(status_code=404, detail="Article not found")
    db.delete(article)
    db.commit()
    return RedirectResponse(url="/insights", status_code=303)


@router.get("/articles/{article_id}/sources/{source_id}")
def download_source(article_id: str, source_id: str, request: Request, db: Session = Depends(get_db)):
    _session, current_user = _get_session_and_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)

    article = db.get(Article, article_id)
    group_ids = _group_ids_for_user(db, current_user)
    if not article or not _article_accessible(article, current_user, group_ids):
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
