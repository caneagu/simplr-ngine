# Rag Email MVP

MVP web app that ingests inbound email (MailerSend webhook), extracts text + PDF content, and stores knowledge base articles for hybrid search (metadata + vector similarity). Built with FastAPI, Postgres + pgvector, and LangChain.

## Architecture
- FastAPI app exposes webhook + REST APIs + minimal UI
- Ingestion pipeline parses email payload, extracts PDF text, chunks content, summarizes, and embeds via LangChain
- Postgres stores articles, sources, and embedded chunks with metadata for hybrid search

## Tech stack
- FastAPI + Jinja2 templates
- SQLAlchemy + pgvector
- LangChain + OpenAI/OpenRouter
- pypdf for PDF extraction

## Local setup
1) Create and activate a virtualenv
2) Install deps:

```bash
pip install -r requirements.txt
```

3) Create DB and enable pgvector:

```bash
createdb rag_mail
psql rag_mail -f scripts/init_db.sql
```

4) Configure env:

```bash
cp .env.example .env
```

5) Run the app:

```bash
uvicorn app.main:app --reload
```

## Endpoints
- `POST /webhooks/mailersend` inbound email webhook (MailerSend payload)
- `GET /api/articles` list articles
- `GET /api/articles/{id}` get article
- `PUT /api/articles/{id}` update article
- `DELETE /api/articles/{id}` delete article
- `GET /api/search?query=...` vector search
- `GET /articles` UI list

## Notes

## Reset seed data
Remove seeded dummy articles:

```bash
PYTHONPATH=. ./bin/python scripts/reset_seed.py
```


## Seed data
Run to insert 10 business-relevant sample articles:

```bash
python scripts/seed_dummy.py
```

- Embedding dimension in `scripts/init_db.sql` is set to 1536 for `text-embedding-3-small`.
- To change embedding model with different dimensions, update `EMBEDDING_DIM` in `.env` and the `vector(...)` column in `scripts/init_db.sql`.
- MailerSend attachment payloads are expected to include base64 content for PDFs.
- Hybrid search combines Postgres full-text ranking with vector similarity (ensure `scripts/init_db.sql` has been run).

## Testing plan
- Unit: payload parsing, PDF extraction, chunking consistency
- Integration: webhook ingest inserts article + chunks, search returns results
- UI: edit flow updates title/summary/metadata

## Future extensions
- IMAP polling: add a background worker that fetches and normalizes inbound mail into the same schema
- OCR/vision: add a parser that renders PDF pages to images and sends them to a vision model
- Multi-tenancy: add `tenant_id` columns and tenant-aware filters on all queries
- Custom LLM connectors: implement a provider registry to swap LangChain models per user
