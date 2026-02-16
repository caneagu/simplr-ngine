from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import (
    _article_accessible,
    _build_folder_tree,
    _folder_accessible,
    _folder_paths_map,
    _get_session_and_user,
    _group_ids_for_user,
    _resolve_folder_path,
    _resolve_scope,
    _sync_article_folder_metadata,
    _template_context,
    _user_groups,
)
from app.models import Article, ArticleVersion, Folder, Group
from app.web import templates


router = APIRouter()


@router.get("/articles", response_class=HTMLResponse)
def files_ui(
    request: Request,
    db: Session = Depends(get_db),
    folder_id: Optional[str] = None,
    day: Optional[str] = None,
    scope: Optional[str] = None,
    group_id: Optional[str] = None,
):
    session, current_user = _get_session_and_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)

    scope_name, selected_group = _resolve_scope(db, current_user, scope, group_id)
    groups = _user_groups(db, current_user)

    if scope_name == "group" and selected_group:
        folders = db.query(Folder).filter(Folder.group_id == selected_group.id).all()
    else:
        folders = (
            db.query(Folder)
            .filter(Folder.owner_id == current_user.id, Folder.group_id.is_(None))
            .all()
        )
    folder_tree = _build_folder_tree(folders)
    folder_path_by_id = _folder_paths_map(folders)

    active_folder_id: Optional[uuid.UUID] = None
    if folder_id:
        try:
            candidate = uuid.UUID(folder_id)
            folder = db.get(Folder, candidate)
            if folder:
                if scope_name == "group" and selected_group:
                    if folder.group_id == selected_group.id:
                        active_folder_id = candidate
                elif folder.owner_id == current_user.id and folder.group_id is None:
                    active_folder_id = candidate
        except ValueError:
            active_folder_id = None

    if scope_name == "group" and selected_group:
        articles_query = db.query(Article).filter(Article.group_id == selected_group.id)
    else:
        articles_query = db.query(Article).filter(
            Article.owner_id == current_user.id, Article.group_id.is_(None)
        )
    selected_day = None
    if day:
        try:
            selected_day = datetime.fromisoformat(day).date()
        except ValueError:
            selected_day = None
    if selected_day:
        articles_query = articles_query.filter(func.date(Article.created_at) == selected_day)
    if not selected_day:
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
            selected_day=selected_day.isoformat() if selected_day else None,
            scope=scope_name,
            selected_group=selected_group,
            groups=groups,
            articles=articles,
            version_counts=version_counts,
            folder_paths=article_folder_paths,
        ),
    )


@router.post("/articles/folders")
def create_folder(
    request: Request,
    name: str = Form(...),
    parent_id: Optional[str] = Form(None),
    group_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    _session, current_user = _get_session_and_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)

    selected_group: Optional[Group] = None
    if group_id:
        scope_name, group = _resolve_scope(db, current_user, "group", group_id)
        if scope_name != "group" or not group:
            raise HTTPException(status_code=404, detail="Group not found")
        selected_group = group

    parent_uuid: Optional[uuid.UUID] = None
    if parent_id:
        try:
            parent_uuid = uuid.UUID(parent_id)
        except ValueError:
            parent_uuid = None
    if parent_uuid:
        parent = db.get(Folder, parent_uuid)
        if not parent or not _folder_accessible(parent, current_user, _group_ids_for_user(db, current_user)):
            raise HTTPException(status_code=404, detail="Parent folder not found")
        if selected_group and parent.group_id != selected_group.id:
            raise HTTPException(status_code=400, detail="Parent folder is in a different group")
        if not selected_group and parent.group_id is not None:
            raise HTTPException(status_code=400, detail="Parent folder is in a group")

    folder_name = name.strip()
    if not folder_name:
        raise HTTPException(status_code=400, detail="Folder name is required")
    db.add(
        Folder(
            owner_id=current_user.id,
            group_id=selected_group.id if selected_group else None,
            parent_id=parent_uuid,
            name=folder_name,
        )
    )
    db.commit()
    _sync_article_folder_metadata(db, current_user.id, selected_group.id if selected_group else None)
    db.commit()
    return RedirectResponse(url="/articles", status_code=303)


@router.post("/articles/folders/{folder_id}/rename")
def rename_folder(folder_id: str, request: Request, name: str = Form(...), db: Session = Depends(get_db)):
    _session, current_user = _get_session_and_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    folder = db.get(Folder, folder_id)
    group_ids = _group_ids_for_user(db, current_user)
    if not folder or not _folder_accessible(folder, current_user, group_ids):
        raise HTTPException(status_code=404, detail="Folder not found")
    folder.name = name.strip()
    db.add(folder)
    db.commit()
    _sync_article_folder_metadata(db, current_user.id, folder.group_id)
    db.commit()
    return RedirectResponse(url="/articles", status_code=303)


@router.post("/articles/folders/{folder_id}/delete")
def delete_folder(folder_id: str, request: Request, db: Session = Depends(get_db)):
    _session, current_user = _get_session_and_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    folder = db.get(Folder, folder_id)
    group_ids = _group_ids_for_user(db, current_user)
    if not folder or not _folder_accessible(folder, current_user, group_ids):
        raise HTTPException(status_code=404, detail="Folder not found")
    db.delete(folder)
    db.commit()
    _sync_article_folder_metadata(db, current_user.id, folder.group_id)
    db.commit()
    return RedirectResponse(url="/articles", status_code=303)


@router.post("/articles/folders/{folder_id}/move")
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
    group_ids = _group_ids_for_user(db, current_user)
    if not folder or not _folder_accessible(folder, current_user, group_ids):
        raise HTTPException(status_code=404, detail="Folder not found")

    parent_uuid: Optional[uuid.UUID] = None
    if target_parent:
        try:
            parent_uuid = uuid.UUID(target_parent)
        except ValueError:
            parent_uuid = None

    if parent_uuid:
        parent = db.get(Folder, parent_uuid)
        if not parent or not _folder_accessible(parent, current_user, group_ids):
            raise HTTPException(status_code=404, detail="Parent folder not found")
        if parent.id == folder.id:
            raise HTTPException(status_code=400, detail="Cannot move into itself")
        if parent.group_id != folder.group_id:
            raise HTTPException(status_code=400, detail="Parent folder is in a different group")

    folder.parent_id = parent_uuid
    db.add(folder)
    db.commit()
    _sync_article_folder_metadata(db, current_user.id, folder.group_id)
    db.commit()
    return {"status": "ok"}


@router.post("/articles/{article_id}/move")
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
    group_ids = _group_ids_for_user(db, current_user)
    if not article or not _article_accessible(article, current_user, group_ids):
        raise HTTPException(status_code=404, detail="Article not found")

    folder_uuid: Optional[uuid.UUID] = None
    if folder_id:
        try:
            folder_uuid = uuid.UUID(folder_id)
        except ValueError:
            folder_uuid = None

    if folder_uuid:
        folder = db.get(Folder, folder_uuid)
        if not folder or not _folder_accessible(folder, current_user, group_ids):
            raise HTTPException(status_code=404, detail="Folder not found")
        if folder.group_id != article.group_id:
            raise HTTPException(status_code=400, detail="Folder is in a different group")

    article.folder_id = folder_uuid
    if article.metadata_ is None:
        article.metadata_ = {}
    article.metadata_["folder_id"] = str(folder_uuid) if folder_uuid else None
    article.metadata_["folder_path"] = _resolve_folder_path(db, folder_uuid) or "Root"
    db.add(article)
    db.commit()
    return {"status": "ok"}


@router.get("/files")
def files_redirect() -> RedirectResponse:
    return RedirectResponse(url="/articles", status_code=302)
