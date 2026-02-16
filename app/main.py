from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from app.api.routers import articles, auth, chat, folders, profile, search, webhooks
from app.db import engine


app = FastAPI(title="Simplr")
logger = logging.getLogger("simplr")


@app.on_event("startup")
def ensure_ingest_dedupe_schema() -> None:
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE articles ADD COLUMN IF NOT EXISTS external_dedupe_key text"))
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_articles_external_dedupe_key "
                "ON articles(external_dedupe_key)"
            )
        )
        try:
            conn.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_articles_owner_dedupe_null_group "
                    "ON articles(owner_id, external_dedupe_key) "
                    "WHERE group_id IS NULL AND external_dedupe_key IS NOT NULL"
                )
            )
        except Exception as exc:
            logger.warning("Could not create private-scope dedupe index: %s", exc)
        try:
            conn.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_articles_owner_group_dedupe "
                    "ON articles(owner_id, group_id, external_dedupe_key) "
                    "WHERE group_id IS NOT NULL AND external_dedupe_key IS NOT NULL"
                )
            )
        except Exception as exc:
            logger.warning("Could not create group-scope dedupe index: %s", exc)


app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(auth.router)
app.include_router(profile.router)
app.include_router(webhooks.router)
app.include_router(articles.router)
app.include_router(folders.router)
app.include_router(search.router)
app.include_router(chat.router)
