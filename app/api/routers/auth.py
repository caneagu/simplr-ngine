from __future__ import annotations

import json
import urllib.request

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.deps import SESSION_COOKIE, _get_session_and_user
from app.models import MagicLinkToken, SessionToken, User
from app.services.auth import (
    generate_token,
    hash_token,
    magiclink_expiry,
    normalize_email,
    now_utc,
    session_expiry,
)
from app.services.emailer import send_magic_link
from app.web import templates


router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def root() -> RedirectResponse:
    return RedirectResponse(url="/insights")


@router.get("/login", response_class=HTMLResponse)
def login_ui(request: Request, db: Session = Depends(get_db)):
    session, current_user = _get_session_and_user(request, db)
    if current_user:
        return RedirectResponse(url="/insights", status_code=303)
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "current_user": None,
            "session_expires_in_days": None,
            "message": None,
            "error": None,
            "resend_email": None,
        },
    )


@router.post("/login", response_class=HTMLResponse)
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
                {
                    "request": request,
                    "current_user": None,
                    "session_expires_in_days": None,
                    "message": None,
                    "error": str(exc),
                    "resend_email": normalized,
                },
            )

    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "current_user": None,
            "session_expires_in_days": None,
            "message": "If that email has articles, a sign-in link is on the way.",
            "error": None,
            "resend_email": normalized,
        },
    )


@router.get("/auth/callback", response_class=HTMLResponse)
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
            {
                "request": request,
                "current_user": None,
                "session_expires_in_days": None,
                "message": None,
                "error": "This sign-in link is invalid or expired.",
                "resend_email": None,
            },
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


@router.post("/logout")
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


@router.get("/api/openrouter/models")
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


@router.get("/health")
def healthcheck(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return {"status": "ok"}
