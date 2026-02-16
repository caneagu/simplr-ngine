from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.services.ingest import ingest_email_job
from app.services.mailersend import parse_mailersend_payload


router = APIRouter()
logger = logging.getLogger("simplr")


@router.post("/webhooks/mailersend", status_code=202)
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

    background_tasks.add_task(ingest_email_job, inbound)
    return {"status": "accepted"}
