from __future__ import annotations

from datetime import datetime
from typing import Any
import uuid
from pathlib import Path

from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import settings
from app.db import SessionLocal
from app.models import Article, ArticleVersion, Chunk, Group, Source, User, UserInferenceConfig
from app.services.auth import normalize_email
from app.services.chunking import chunk_text
from app.services.embeddings import embed_texts
from app.services.emailer import send_article_reply
from app.services.llm import categorize_and_extract, extract_insights, summarize_text_with_usage
from app.services.mailersend import InboundEmail
from app.services.pdf import extract_pdf_text


def _inference_for_user(db: Session, user: User) -> dict | None:
    config = None
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


def ingest_email(inbound: InboundEmail, db: Session) -> Article:
    sender_email = normalize_email(inbound.sender)
    sender_user = db.query(User).filter(User.email == sender_email).first()
    if not sender_user:
        sender_user = User(email=sender_email)
        db.add(sender_user)
        try:
            db.flush()
        except IntegrityError:
            db.rollback()
            sender_user = db.query(User).filter(User.email == sender_email).first()
            if not sender_user:
                raise

    recipient_emails = {
        normalize_email(recipient)
        for recipient in inbound.recipients
        if recipient and "@" in recipient
    }
    target_users: dict[str, User] = {}
    for email in recipient_emails:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            user = User(email=email)
            db.add(user)
            try:
                db.flush()
            except IntegrityError:
                db.rollback()
                user = db.query(User).filter(User.email == email).first()
                if not user:
                    continue
        target_users[email] = user
    if not target_users:
        target_users[sender_user.email] = sender_user

    recipient_slugs = {recipient.split("@", 1)[0].lower() for recipient in recipient_emails}
    groups = []
    if recipient_slugs:
        groups = db.query(Group).filter(Group.slug.in_(list(recipient_slugs))).all()

    storage_dir = Path(settings.storage_dir).resolve()
    attachments_dir = storage_dir / "attachments"
    attachments_dir.mkdir(parents=True, exist_ok=True)

    content_parts: list[str] = [inbound.text.strip()]
    base_sources: list[Source] = [
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

    for index, attachment in enumerate(inbound.attachments, start=1):
        if "pdf" not in attachment.content_type.lower():
            continue
        original_name = Path(attachment.filename or "").name or "attachment"
        if "." not in original_name and "pdf" in attachment.content_type.lower():
            original_name = f"{original_name}.pdf"
        if original_name == "attachment":
            original_name = f"attachment-{index}.pdf"
        storage_name = f"{uuid.uuid4().hex}_{original_name}"
        attachment_path = attachments_dir / storage_name
        attachment_path.write_bytes(attachment.content_bytes)
        extracted_text = extract_pdf_text(attachment_path)
        if extracted_text:
            content_parts.append(extracted_text)
        base_sources.append(
            Source(
                source_type="attachment",
                source_name=original_name,
                source_uri=str(attachment_path),
                raw_text=None,
                metadata_={
                    "content_type": attachment.content_type,
                    "original_filename": original_name,
                },
            )
        )

    full_text = "\n\n".join([part for part in content_parts if part]).strip()
    analysis_cache: dict[tuple, dict[str, Any]] = {}

    def _analysis_for(inference: dict | None) -> dict[str, Any]:
        key = None
        if inference:
            key = (
                inference.get("provider"),
                inference.get("base_url"),
                inference.get("api_key"),
            )
        if key in analysis_cache:
            return analysis_cache[key]

        summary = ""
        summary_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if full_text:
            summary, summary_usage = summarize_text_with_usage(full_text, inference=inference)
        classification = categorize_and_extract(full_text, inference=inference) if full_text else {}
        insights = extract_insights(full_text, inference=inference) if full_text else {}
        payload = {
            "summary": summary,
            "summary_usage": summary_usage,
            "classification": classification,
            "insights": insights,
        }
        analysis_cache[key] = payload
        return payload

    thread_ids: list[str] = []
    if inbound.message_id:
        thread_ids.append(inbound.message_id)
    if inbound.in_reply_to:
        thread_ids.append(inbound.in_reply_to)
    if inbound.references:
        thread_ids.extend(inbound.references)
    seen_ids: set[str] = set()
    thread_ids = [item for item in thread_ids if item and not (item in seen_ids or seen_ids.add(item))]

    def _clone_sources() -> list[Source]:
        clones: list[Source] = []
        for source in base_sources:
            clones.append(
                Source(
                    source_type=source.source_type,
                    source_name=source.source_name,
                    source_uri=source.source_uri,
                    raw_text=source.raw_text,
                    metadata_=dict(source.metadata_ or {}),
                )
            )
        return clones

    def _process_target(owner: User, group: Group | None) -> Article | None:
        inference = _inference_for_user(db, owner)
        analysis = _analysis_for(inference)
        summary = analysis["summary"]
        summary_usage = analysis["summary_usage"]
        classification = analysis["classification"]
        insights = analysis["insights"]
        category = classification.get("category", "uncategorized")
        doc_type = classification.get("doc_type", "other")
        document_date = classification.get("document_date")
        language = classification.get("language", "en")
        tags = classification.get("tags", [])
        entities = classification.get("entities", {"people": [], "orgs": [], "locations": []})
        date_mentions = classification.get("date_mentions", [])
        metadata_confidence = classification.get("confidence")
        scope_filter = [
            Article.owner_id == owner.id,
            Article.group_id == (group.id if group else None),
        ]

        duplicate_filters = []
        if inbound.message_id:
            duplicate_filters.append(Article.metadata_["message_id"].astext == inbound.message_id)
        if inbound.inbound_id:
            duplicate_filters.append(Article.metadata_["inbound_id"].astext == inbound.inbound_id)
        if duplicate_filters:
            existing = (
                db.query(Article)
                .filter(*scope_filter)
                .filter(or_(*duplicate_filters))
                .first()
            )
            if existing:
                return existing

        existing_article: Article | None = None
        if inbound.in_reply_to or inbound.references:
            candidates = [item for item in thread_ids if item]
            filters = []
            for message_id in candidates:
                filters.append(Article.metadata_["reply_message_id"].astext == message_id)
                filters.append(Article.metadata_["message_id"].astext == message_id)
                filters.append(Article.metadata_["thread_message_ids"].contains([message_id]))
            if filters:
                existing_article = (
                    db.query(Article)
                    .filter(*scope_filter)
                    .filter(or_(*filters))
                    .first()
                )

        metadata = {
            "sender": inbound.sender,
            "subject": inbound.subject,
            "category": category,
            "categories": classification.get("categories", [category]),
            "doc_type": doc_type,
            "document_date": document_date,
            "language": language,
            "tags": tags,
            "entities": entities,
            "date_mentions": date_mentions,
            "confidence": metadata_confidence,
            "insights": insights,
            "summary_tokens": summary_usage,
            "folder_path": "Root",
            "message_id": inbound.message_id,
            "inbound_id": inbound.inbound_id,
            "thread_message_ids": thread_ids,
        }

        if existing_article:
            version_number = len(existing_article.versions) + 1
            db.add(
                ArticleVersion(
                    article_id=existing_article.id,
                    version=version_number,
                    title=existing_article.title,
                    summary=existing_article.summary,
                    content_text=existing_article.content_text,
                    metadata_=existing_article.metadata_,
                )
            )

            if existing_article.metadata_:
                existing_thread_ids = existing_article.metadata_.get("thread_message_ids", [])
                combined_ids = [item for item in existing_thread_ids + thread_ids if item]
                combined_seen: set[str] = set()
                metadata["thread_message_ids"] = [
                    item for item in combined_ids if not (item in combined_seen or combined_seen.add(item))
                ]
                if existing_article.metadata_.get("reply_message_id"):
                    metadata["reply_message_id"] = existing_article.metadata_.get("reply_message_id")
                if existing_article.metadata_.get("reply_sent_at"):
                    metadata["reply_sent_at"] = existing_article.metadata_.get("reply_sent_at")
                if existing_article.metadata_.get("message_id"):
                    metadata["message_id"] = existing_article.metadata_.get("message_id")
                if existing_article.metadata_.get("folder_path"):
                    metadata["folder_path"] = existing_article.metadata_.get("folder_path")
                if existing_article.metadata_.get("folder_id"):
                    metadata["folder_id"] = existing_article.metadata_.get("folder_id")

            existing_article.title = inbound.subject or existing_article.title
            existing_article.summary = summary
            existing_article.content_text = full_text
            existing_article.metadata_ = metadata
            db.add(existing_article)
            db.query(Source).filter(Source.article_id == existing_article.id).delete(synchronize_session=False)
            db.query(Chunk).filter(Chunk.article_id == existing_article.id).delete(synchronize_session=False)
            db.flush()

            for source in _clone_sources():
                source.article_id = existing_article.id
                db.add(source)

            chunks = chunk_text(full_text) if full_text else []
            embeddings = embed_texts(chunks) if chunks else []

            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                db.add(
                    Chunk(
                        article_id=existing_article.id,
                        chunk_index=index,
                        content=chunk,
                        embedding=embedding,
                        metadata_={"source": "email", "category": category},
                    )
                )

            db.commit()
            db.refresh(existing_article)
            return existing_article

        article = Article(
            owner_id=owner.id,
            group_id=group.id if group else None,
            title=inbound.subject or "(no subject)",
            summary=summary,
            content_text=full_text,
            metadata_=metadata,
        )

        db.add(article)
        db.flush()

        for source in _clone_sources():
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
        return article

    processed_articles: list[Article] = []
    for user in target_users.values():
        article = _process_target(user, None)
        if article:
            processed_articles.append(article)

    for group in groups:
        group_owner = db.get(User, group.owner_id)
        if not group_owner:
            continue
        article = _process_target(group_owner, group)
        if article:
            processed_articles.append(article)

    if processed_articles:
        needs_reply = any(not (article.metadata_ or {}).get("reply_sent_at") for article in processed_articles)
        if needs_reply:
            primary = processed_articles[0]
            article_url = f"{settings.app_base_url.rstrip('/')}/articles/{primary.id}"
            try:
                reply_message_id = send_article_reply(
                    sender_user.email,
                    primary.title,
                    primary.summary,
                    article_url,
                    summary_usage,
                )
            except RuntimeError:
                reply_message_id = None
            else:
                reply_sent_at = datetime.utcnow().isoformat()
                for article in processed_articles:
                    if article.metadata_ is None:
                        article.metadata_ = {}
                    article.metadata_["reply_sent_at"] = reply_sent_at
                    if reply_message_id:
                        article.metadata_["reply_message_id"] = reply_message_id
                        thread_message_ids = article.metadata_.get("thread_message_ids", [])
                        if reply_message_id not in thread_message_ids:
                            thread_message_ids.append(reply_message_id)
                            article.metadata_["thread_message_ids"] = thread_message_ids
                    db.add(article)
                db.commit()

    return processed_articles[0]


def ingest_email_job(inbound: InboundEmail) -> None:
    db = SessionLocal()
    try:
        ingest_email(inbound, db)
    finally:
        db.close()
