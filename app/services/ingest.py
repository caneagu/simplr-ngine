from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from app.config import settings
from app.models import Article, Chunk, Source
from app.services.chunking import chunk_text
from app.services.embeddings import embed_texts
from app.services.llm import categorize_and_extract, summarize_text
from app.services.mailersend import InboundEmail
from app.services.pdf import extract_pdf_text


def ingest_email(inbound: InboundEmail, db: Session) -> Article:
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
            metadata_={"sender": inbound.sender, "subject": inbound.subject},
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
    extracted = classification.get("metadata", {})

    article = Article(
        title=inbound.subject or "(no subject)",
        summary=summary,
        content_text=full_text,
        metadata_={
            "sender": inbound.sender,
            "subject": inbound.subject,
            "category": category,
            "extracted": extracted,
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
                metadata_={"source": "email", "category": category, "extracted": extracted},
            )
        )

    db.commit()
    db.refresh(article)
    return article
