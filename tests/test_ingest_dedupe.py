from __future__ import annotations

from app.services.ingest import _dedupe_key_for
from app.services.mailersend import InboundEmail


def _inbound(**overrides):
    payload = {
        "sender": "alice@example.com",
        "subject": "Status update",
        "text": "Quarterly update and next steps.",
        "attachments": [],
        "message_id": None,
        "inbound_id": None,
        "in_reply_to": None,
        "references": [],
        "recipients": [],
    }
    payload.update(overrides)
    return InboundEmail(**payload)


def test_dedupe_key_prefers_message_id() -> None:
    inbound = _inbound(message_id=" <ABC-123@example.com> ")
    assert _dedupe_key_for(inbound) == "message:<abc-123@example.com>"


def test_dedupe_key_uses_inbound_id_when_message_id_missing() -> None:
    inbound = _inbound(message_id=None, inbound_id="msg-42")
    assert _dedupe_key_for(inbound) == "inbound:msg-42"


def test_dedupe_key_uses_content_hash_as_last_resort() -> None:
    inbound = _inbound(message_id=None, inbound_id=None)
    key = _dedupe_key_for(inbound)
    assert key is not None
    assert key.startswith("content:")


def test_dedupe_key_is_none_when_payload_is_effectively_empty() -> None:
    inbound = _inbound(
        sender="",
        subject="",
        text="",
        message_id=None,
        inbound_id=None,
    )
    assert _dedupe_key_for(inbound) is None
