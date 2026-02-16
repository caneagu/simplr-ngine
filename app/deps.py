from __future__ import annotations

import math
import re
import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import HTTPException, Request, status
from sqlalchemy import Integer, and_, cast, func, or_
from sqlalchemy.orm import Session

from app.config import settings
from app.constants import CATEGORIES
from app.models import (
    Article,
    Chunk,
    Folder,
    Group,
    GroupMember,
    SessionToken,
    Source,
    UserInferenceConfig,
    User,
)
from app.services.auth import hash_token, now_utc

SESSION_COOKIE = "rag_session"


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


def _normalize_group_slug(value: str) -> str:
    return value.strip().lower()


def _valid_group_slug(value: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9]+", value))


def _user_groups(db: Session, user: User) -> list[Group]:
    return (
        db.query(Group)
        .outerjoin(GroupMember, GroupMember.group_id == Group.id)
        .filter(
            or_(
                Group.owner_id == user.id,
                GroupMember.user_id == user.id,
            )
        )
        .distinct()
        .order_by(Group.created_at.desc())
        .all()
    )


def _group_ids_for_user(db: Session, user: User) -> set[uuid.UUID]:
    return {group.id for group in _user_groups(db, user)}


def _resolve_scope(
    db: Session,
    user: User,
    scope: Optional[str],
    group_id: Optional[str],
) -> tuple[str, Optional[Group]]:
    if scope != "group":
        return "personal", None
    if not group_id:
        return "personal", None
    try:
        group_uuid = uuid.UUID(group_id)
    except ValueError:
        return "personal", None
    group = db.get(Group, group_uuid)
    if not group:
        return "personal", None
    if group.owner_id != user.id:
        is_member = (
            db.query(GroupMember)
            .filter(GroupMember.group_id == group.id, GroupMember.user_id == user.id)
            .first()
        )
        if not is_member:
            return "personal", None
    return "group", group


def _article_accessible(article: Article, user: User, group_ids: set[uuid.UUID]) -> bool:
    if article.group_id:
        return article.group_id in group_ids
    return article.owner_id == user.id


def _folder_accessible(folder: Folder, user: User, group_ids: set[uuid.UUID]) -> bool:
    if folder.group_id:
        return folder.group_id in group_ids
    return folder.owner_id == user.id


def _get_user_inference_config(db: Session, user: User) -> Optional[dict[str, str]]:
    config: Optional[UserInferenceConfig] = None
    if user.default_inference_id:
        config = db.get(UserInferenceConfig, user.default_inference_id)
    if not config:
        config = (
            db.query(UserInferenceConfig)
            .filter(UserInferenceConfig.user_id == user.id)
            .order_by(UserInferenceConfig.created_at.desc())
            .first()
        )
    if config:
        return {
            "provider": config.provider,
            "api_key": config.api_key,
            "base_url": config.base_url or "",
            "title": "simplr",
        }
    if user.inference_api_key:
        return {
            "provider": "openrouter",
            "api_key": user.inference_api_key,
            "base_url": user.inference_endpoint or settings.openrouter_base_url,
            "title": "simplr",
        }
    return None


def _provider_for_inference(inference: Optional[dict[str, str]]) -> str:
    if inference and inference.get("provider"):
        return inference["provider"]
    return settings.llm_provider


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
        article_count = db.query(func.count(Article.id)).filter(Article.owner_id == current_user.id).scalar() or 0
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


def _article_comments(article: Article) -> list[dict[str, Any]]:
    metadata = article.metadata_ or {}
    comments = metadata.get("comments")
    if not isinstance(comments, list):
        return []
    normalized: list[dict[str, Any]] = []
    for entry in comments:
        if not isinstance(entry, dict):
            continue
        mode = entry.get("mode")
        if mode not in {"augment", "supersede"}:
            mode = "augment"
        normalized.append(
            {
                "id": entry.get("id"),
                "mode": mode,
                "author_email": entry.get("author_email") or "unknown",
                "created_at": entry.get("created_at"),
                "comment_text": (entry.get("comment_text") or "").strip(),
                "replacement_summary": (entry.get("replacement_summary") or "").strip(),
                "replacement_content": (entry.get("replacement_content") or "").strip(),
            }
        )
    return normalized


def _effective_article_view(article: Article) -> dict[str, Any]:
    effective_summary = article.summary or ""
    effective_content = article.content_text or ""
    comments = _article_comments(article)
    has_override = False
    for comment in comments:
        if comment["mode"] == "supersede":
            if comment["replacement_summary"]:
                effective_summary = comment["replacement_summary"]
                has_override = True
            if comment["replacement_content"]:
                effective_content = comment["replacement_content"]
                has_override = True
            continue
        if comment["mode"] == "augment":
            if comment["comment_text"]:
                effective_content = (
                    effective_content
                    + "\n\n"
                    + f"[User comment by {comment['author_email']} at {comment['created_at']}]"
                    + "\n"
                    + comment["comment_text"]
                ).strip()
                has_override = True
    return {
        "summary": effective_summary,
        "content_text": effective_content,
        "has_override": has_override,
        "comments": comments,
    }


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
    if folder.group_id:
        folders = db.query(Folder).filter(Folder.group_id == folder.group_id).all()
    else:
        folders = db.query(Folder).filter(Folder.owner_id == folder.owner_id).all()
    paths = _folder_paths_map(folders)
    return paths.get(folder_id)


def _sync_article_folder_metadata(
    db: Session,
    owner_id: uuid.UUID,
    group_id: Optional[uuid.UUID] = None,
) -> None:
    if group_id:
        folders = db.query(Folder).filter(Folder.group_id == group_id).all()
        articles = db.query(Article).filter(Article.group_id == group_id).all()
    else:
        folders = db.query(Folder).filter(Folder.owner_id == owner_id, Folder.group_id.is_(None)).all()
        articles = (
            db.query(Article)
            .filter(Article.owner_id == owner_id, Article.group_id.is_(None))
            .all()
        )
    folder_paths = _folder_paths_map(folders)
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
