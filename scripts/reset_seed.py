from __future__ import annotations

from sqlalchemy import delete

from app.db import SessionLocal
from app.models import Article, Chunk, Source


def reset_seeded_data() -> None:
    db = SessionLocal()
    try:
        seeded_articles = db.query(Article.id).filter(Article.metadata_["seed"].astext == "true").subquery()

        db.execute(delete(Chunk).where(Chunk.article_id.in_(seeded_articles)))
        db.execute(delete(Source).where(Source.article_id.in_(seeded_articles)))
        db.execute(delete(Article).where(Article.id.in_(seeded_articles)))
        db.commit()
    finally:
        db.close()


if __name__ == "__main__":
    reset_seeded_data()
