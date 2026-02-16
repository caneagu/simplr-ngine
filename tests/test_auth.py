from __future__ import annotations

from datetime import timedelta

from app.config import settings
from app.services.auth import (
    generate_token,
    hash_token,
    magiclink_expiry,
    normalize_email,
    now_utc,
    session_expiry,
)


def test_normalize_email_handles_display_name() -> None:
    assert normalize_email("Alice Example <Alice@Example.COM>") == "alice@example.com"


def test_hash_token_is_stable_sha256_hex() -> None:
    token = "abc123"
    digest = hash_token(token)
    assert digest == hash_token(token)
    assert len(digest) == 64
    assert all(char in "0123456789abcdef" for char in digest)


def test_magiclink_expiry_uses_configured_ttl() -> None:
    start = now_utc()
    expiry = magiclink_expiry()
    delta = expiry - start
    expected = timedelta(minutes=settings.magiclink_ttl_minutes)
    assert expected - timedelta(seconds=1) <= delta <= expected + timedelta(seconds=1)


def test_session_expiry_uses_configured_days() -> None:
    start = now_utc()
    expiry = session_expiry()
    delta = expiry - start
    expected = timedelta(days=settings.session_days)
    assert expected - timedelta(seconds=1) <= delta <= expected + timedelta(seconds=1)


def test_generate_token_is_urlsafe_non_empty() -> None:
    token = generate_token()
    assert token
    assert isinstance(token, str)
