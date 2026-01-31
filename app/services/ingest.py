from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import settings
from app.db import SessionLocal
from app.models import Article, Chunk, Source, User
from app.services.auth import normalize_email
from app.services.chunking import chunk_text
from app.services.embeddings import embed_texts
from app.services.emailer import send_article_reply
from app.services.llm import categorize_and_extract, summarize_text
from app.services.mailersend import InboundEmail
from app.services.pdf import extract_pdf_text


def ingest_email(inbound: InboundEmail, db: Session) -> Article:
    sender_email = normalize_email(inbound.sender)
    user = db.query(User).filter(User.email == sender_email).first()
    if not user:
        user = User(email=sender_email)
        db.add(user)
        try:
            db.flush()
        except IntegrityError:
            db.rollback()
            user = db.query(User).filter(User.email == sender_email).first()
            if not user:
                raise

    storage_dir = Path(settings.storage_dir).resolve()
    attachments_dir = storage_dir / "attachments"
    attachments_dir.mkdir(parents=True, exist_ok=True)

    content_parts: list[str] = [inbound.text.strip()]
    sources: list[Source] = [
        Source(
            source_type="email",
            source_name="email",
            source_uri=None,
            raw_text=inbound.text,
            metadata_={
                "sender": inbound.sender,
                "subject": inbound.subject,
                "message_id": inbound.message_id,
                "inbound_id": inbound.inbound_id,
            },
        )
    ]

    for attachment in inbound.attachments:
        if "pdf" not in attachment.content_type.lower():
            continue
        attachment_path = attachments_dir / attachment.filename
        attachment_path.write_bytes(attachment.content_bytes)
        extracted_text = extract_pdf_text(attachment_path)
        if extracted_text:
            content_parts.append(extracted_text)
        sources.append(
            Source(
                source_type="attachment",
                source_name=attachment.filename,
                source_uri=str(attachment_path),
                raw_text=None,
                metadata_={"content_type": attachment.content_type},
            )
        )

    full_text = "\n\n".join([part for part in content_parts if part]).strip()
    summary = summarize_text(full_text) if full_text else ""
    classification = categorize_and_extract(full_text) if full_text else {}
    category = classification.get("category", "uncategorized")
    doc_type = classification.get("doc_type", "other")
    document_date = classification.get("document_date")
    language = classification.get("language", "en")
    tags = classification.get("tags", [])
    entities = classification.get("entities", {"people": [], "orgs": [], "locations": []})
    date_mentions = classification.get("date_mentions", [])
    metadata_confidence = classification.get("confidence")

    article = Article(
        owner_id=user.id,
        title=inbound.subject or "(no subject)",
        summary=summary,
        content_text=full_text,
        metadata_={
            "sender": inbound.sender,
            "subject": inbound.subject,
            "category": category,
            "doc_type": doc_type,
            "document_date": document_date,
            "language": language,
            "tags": tags,
            "entities": entities,
            "date_mentions": date_mentions,
            "confidence": metadata_confidence,
            "folder_path": "Root",
            "message_id": inbound.message_id,
            "inbound_id": inbound.inbound_id,
        },
    )

    db.add(article)
    db.flush()

    for source in sources:
        source.article_id = article.id
        db.add(source)

    chunks = chunk_text(full_text) if full_text else []
    embeddings = embed_texts(chunks) if chunks else []

    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        db.add(
            Chunk(
                article_id=article.id,
                chunk_index=index,
                content=chunk,
                embedding=embedding,
                metadata_={"source": "email", "category": category},
            )
        )

    db.commit()
    db.refresh(article)

    if not article.metadata_.get("reply_sent_at"):
        article_url = f"{settings.app_base_url.rstrip('/')}/articles/{article.id}"
        try:
            send_article_reply(user.email, article.title, article.summary, article_url)
        except RuntimeError:
            pass
        else:
            article.metadata_["reply_sent_at"] = datetime.utcnow().isoformat()
            db.add(article)
            db.commit()
    return article


def ingest_email_job(inbound: InboundEmail) -> None:
    db = SessionLocal()
    try:
        ingest_email(inbound, db)
    finally:
        db.close()
