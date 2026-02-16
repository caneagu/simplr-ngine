from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import Integer, cast, func
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import (
    _get_session_and_user,
    _normalize_group_slug,
    _template_context,
    _user_groups,
    _valid_group_slug,
)
from app.models import Article, Group, GroupMember, User, UserInferenceConfig
from app.services.auth import normalize_email
from app.web import templates


router = APIRouter()


@router.get("/profile", response_class=HTMLResponse)
def profile_ui(request: Request, db: Session = Depends(get_db)):
    session, current_user = _get_session_and_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)

    total_tokens = (
        db.query(
            func.coalesce(
                func.sum(cast(Article.metadata_["summary_tokens"]["total_tokens"].astext, Integer)),
                0,
            )
        )
        .filter(Article.owner_id == current_user.id)
        .scalar()
        or 0
    )

    owned_groups = (
        db.query(Group)
        .filter(Group.owner_id == current_user.id)
        .order_by(Group.created_at.desc())
        .all()
    )
    inference_configs = (
        db.query(UserInferenceConfig)
        .filter(UserInferenceConfig.user_id == current_user.id)
        .order_by(UserInferenceConfig.created_at.desc())
        .all()
    )
    group_ids = [group.id for group in owned_groups]
    members_by_group: dict[uuid.UUID, list[User]] = {}
    if group_ids:
        members = (
            db.query(GroupMember, User)
            .join(User, User.id == GroupMember.user_id)
            .filter(GroupMember.group_id.in_(group_ids))
            .order_by(User.email.asc())
            .all()
        )
        for member, user in members:
            members_by_group.setdefault(member.group_id, []).append(user)

    member_groups = [group for group in _user_groups(db, current_user) if group.owner_id != current_user.id]

    return templates.TemplateResponse(
        "profile.html",
        _template_context(
            request,
            current_user,
            session,
            total_tokens=total_tokens,
            owned_groups=owned_groups,
            members_by_group=members_by_group,
            member_groups=member_groups,
            inference_configs=inference_configs,
        ),
    )


@router.post("/profile")
def profile_update(
    request: Request,
    mobile_phone: str = Form(""),
    db: Session = Depends(get_db),
):
    _session, current_user = _get_session_and_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)

    current_user.mobile_phone = mobile_phone.strip() or None
    db.add(current_user)
    db.commit()

    return RedirectResponse(url="/profile", status_code=303)


@router.post("/profile/inference")
def add_inference_config(
    request: Request,
    name: str = Form(...),
    provider: str = Form("openrouter"),
    base_url: str = Form(""),
    api_key: str = Form(""),
    make_default: Optional[bool] = Form(False),
    db: Session = Depends(get_db),
):
    _session, current_user = _get_session_and_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)

    if not name.strip():
        raise HTTPException(status_code=400, detail="Endpoint name is required")
    if not api_key.strip():
        raise HTTPException(status_code=400, detail="API key is required")

    config = UserInferenceConfig(
        user_id=current_user.id,
        name=name.strip(),
        provider=provider.strip() or "openrouter",
        base_url=base_url.strip() or None,
        api_key=api_key.strip(),
    )
    db.add(config)
    db.commit()
    db.refresh(config)

    if make_default or not current_user.default_inference_id:
        current_user.default_inference_id = config.id
        db.add(current_user)
        db.commit()

    return RedirectResponse(url="/profile", status_code=303)


@router.post("/profile/inference/{config_id}/default")
def set_default_inference_config(
    config_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    _session, current_user = _get_session_and_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)

    config = db.get(UserInferenceConfig, config_id)
    if not config or config.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Inference config not found")

    current_user.default_inference_id = config.id
    db.add(current_user)
    db.commit()
    return RedirectResponse(url="/profile", status_code=303)


@router.post("/profile/inference/{config_id}/delete")
def delete_inference_config(
    config_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    _session, current_user = _get_session_and_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)

    config = db.get(UserInferenceConfig, config_id)
    if not config or config.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Inference config not found")

    if current_user.default_inference_id == config.id:
        current_user.default_inference_id = None
        db.add(current_user)
    db.delete(config)
    db.commit()
    return RedirectResponse(url="/profile", status_code=303)


@router.post("/profile/groups")
def create_group(
    request: Request,
    name: str = Form(...),
    group_slug: str = Form(...),
    db: Session = Depends(get_db),
):
    _session, current_user = _get_session_and_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)

    slug = _normalize_group_slug(group_slug)
    if not slug or not _valid_group_slug(slug):
        raise HTTPException(status_code=400, detail="Group address must include letters and numbers only.")
    if db.query(Group).filter(Group.slug == slug).first():
        raise HTTPException(status_code=400, detail="Group address already in use.")

    group = Group(owner_id=current_user.id, name=name.strip() or slug, slug=slug)
    db.add(group)
    db.commit()
    return RedirectResponse(url="/profile", status_code=303)


@router.post("/profile/groups/{group_id}/members")
def add_group_member(
    group_id: str,
    request: Request,
    email: str = Form(...),
    db: Session = Depends(get_db),
):
    _session, current_user = _get_session_and_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)

    group = db.get(Group, group_id)
    if not group or group.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Group not found")

    normalized = normalize_email(email)
    user = db.query(User).filter(User.email == normalized).first()
    if not user:
        user = User(email=normalized)
        db.add(user)
        db.flush()

    exists = (
        db.query(GroupMember)
        .filter(GroupMember.group_id == group.id, GroupMember.user_id == user.id)
        .first()
    )
    if not exists:
        db.add(GroupMember(group_id=group.id, user_id=user.id, role="member"))
        db.commit()

    return RedirectResponse(url="/profile", status_code=303)


@router.post("/profile/groups/{group_id}/members/{member_id}/remove")
def remove_group_member(
    group_id: str,
    member_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    _session, current_user = _get_session_and_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)

    group = db.get(Group, group_id)
    if not group or group.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Group not found")

    try:
        member_uuid = uuid.UUID(member_id)
    except ValueError:
        member_uuid = None
    member = None
    if member_uuid:
        member = (
            db.query(GroupMember)
            .filter(GroupMember.group_id == group.id, GroupMember.user_id == member_uuid)
            .first()
        )
    if member:
        db.delete(member)
        db.commit()

    return RedirectResponse(url="/profile", status_code=303)
