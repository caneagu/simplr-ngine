from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Optional

from bs4 import BeautifulSoup


@dataclass
class InboundAttachment:
    filename: str
    content_type: str
    content_bytes: bytes


@dataclass
class InboundEmail:
    sender: str
    subject: str
    text: str
    attachments: list[InboundAttachment]


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n").strip()


def _first_value(payload: dict[str, Any], keys: list[str]) -> Optional[Any]:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return None


def _extract_sender(sender_value: Any) -> str:
    if isinstance(sender_value, dict):
        return (
            sender_value.get("email")
            or sender_value.get("address")
            or sender_value.get("value")
            or "unknown@example.com"
        )
    if isinstance(sender_value, list) and sender_value:
        first = sender_value[0]
        if isinstance(first, dict):
            return first.get("email") or first.get("address") or "unknown@example.com"
        if isinstance(first, str):
            return first
    if isinstance(sender_value, str):
        return sender_value
    return "unknown@example.com"


def _decode_base64(value: Any) -> Optional[bytes]:
    if not value or not isinstance(value, str):
        return None
    payload = value
    if "," in payload and payload.strip().startswith("data:"):
        payload = payload.split(",", 1)[1]
    try:
        return base64.b64decode(payload)
    except (ValueError, TypeError):
        return None


def _normalize_attachments(payload: Any) -> list[dict[str, Any]]:
    if not payload:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list):
            return [item for item in payload["data"] if isinstance(item, dict)]
        return [item for item in payload.values() if isinstance(item, dict)]
    return []


def parse_mailersend_payload(payload: dict[str, Any]) -> InboundEmail:
    data = payload.get("data") or payload.get("message") or payload

    sender_value = _first_value(data, ["from", "sender", "from_email", "mail_from"])
    sender = _extract_sender(sender_value)

    subject = _first_value(data, ["subject", "headers.subject"]) or "(no subject)"

    text = _first_value(data, ["text", "text_plain", "plain", "body", "body_plain"]) or ""
    html = _first_value(data, ["html", "text_html", "body_html"]) or ""

    if not text and html:
        text = _html_to_text(html)

    attachments_payload = _first_value(data, ["attachments", "attachment", "files"]) or []
    attachments_data = _normalize_attachments(attachments_payload)

    attachments: list[InboundAttachment] = []
    for attachment in attachments_data:
        filename = attachment.get("filename") or attachment.get("name") or "attachment"
        content_type = attachment.get("content_type") or attachment.get("content-type") or "application/octet-stream"
        content_b64 = (
            attachment.get("content")
            or attachment.get("data")
            or attachment.get("content_base64")
            or attachment.get("body")
        )
        content_bytes = _decode_base64(content_b64)
        if not content_bytes:
            continue
        attachments.append(
            InboundAttachment(
                filename=filename,
                content_type=content_type,
                content_bytes=content_bytes,
            )
        )

    return InboundEmail(sender=sender, subject=str(subject), text=str(text), attachments=attachments)
