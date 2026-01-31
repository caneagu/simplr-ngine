from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from email.utils import parseaddr

from app.config import settings


def normalize_email(value: str) -> str:
    _, addr = parseaddr(value or "")
    return (addr or value or "").strip().lower()


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def magiclink_expiry() -> datetime:
    return now_utc() + timedelta(minutes=settings.magiclink_ttl_minutes)


def session_expiry() -> datetime:
    return now_utc() + timedelta(days=settings.session_days)


def generate_token() -> str:
    return secrets.token_urlsafe(32)
