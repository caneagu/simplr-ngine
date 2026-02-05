# Architecture Documentation

## Overview

The Simplr is a FastAPI-based web application that implements a complete RAG (Retrieval-Augmented Generation) pipeline for email-based knowledge management. It combines email ingestion, document processing, vector search, and conversational AI in a single cohesive system.

## System Architecture

```
                    ┌─────────────────┐
                    │   MailerSend    │
                    │   (Webhook)     │
                    └────────┬────────┘
                             │
                             ▼
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   OpenAI/   │     │  FastAPI App    │     │   PostgreSQL    │
│   OpenRouter│◀────│                 │────▶│   + pgvector    │
└─────────────┘     └─────────────────┘     └─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   SMTP Server   │
                    │  (Magic Links)  │
                    └─────────────────┘
```

## Data Flow

### 1. Email Ingestion Flow

1. MailerSend sends inbound email webhook to `/webhooks/mailersend`
2. Payload is parsed into `InboundEmail` dataclass
3. User is created/found by normalized email address
4. PDF attachments are extracted and text is parsed
5. Content is summarized and classified by LLM
6. Text is chunked and embeddings are generated
7. Article, Sources, and Chunks are stored in PostgreSQL
8. Confirmation email is sent to user

### 2. Search Flow

1. User submits search query
2. Query is embedded using same model as chunks
3. Hybrid search combines:
   - Vector similarity (cosine distance)
   - Full-text search (PostgreSQL ts_rank)
   - Keyword boost (exact match bonus)
4. Results scored and ranked by combined similarity
5. Top results returned with scoring breakdown

### 3. Chat Flow

1. User submits question via chat interface
2. Question is embedded and relevant chunks retrieved
3. Context is built from top-matching chunks
4. Conversation history is retrieved from memory
5. LLM streams response with context + history
6. Response and references displayed to user
7. Conversation history updated

## Component Details

### Core Models

| Model | Purpose | Key Relationships |
|-------|---------|-------------------|
| **User** | Email-based accounts | Has many Articles, MagicLinks, Sessions |
| **Article** | Knowledge base content | Belongs to User, has Sources and Chunks |
| **Source** | Original sources (email, PDF) | Belongs to Article |
| **Chunk** | Vector-embedded text segments | Belongs to Article, has embedding vector |
| **MagicLinkToken** | One-time auth tokens | Belongs to User |
| **SessionToken** | Persistent sessions | Belongs to User |

### Database Schema

**users**: id (uuid, PK), email (text, unique), created_at (timestamp)

**articles**: id (uuid, PK), owner_id (uuid, FK), title (text), summary (text), content_text (text), metadata (jsonb), created_at, updated_at

**sources**: id (uuid, PK), article_id (uuid, FK), source_type (text), source_name, source_uri, raw_text, metadata (jsonb), created_at

**chunks**: id (uuid, PK), article_id (uuid, FK), chunk_index (int), content (text), embedding (vector(1536)), metadata (jsonb), created_at

**magic_link_tokens**: id (uuid, PK), user_id (uuid, FK), token_hash (text, unique), expires_at, used_at, created_at

**sessions**: id (uuid, PK), user_id (uuid, FK), token_hash (text, unique), expires_at, last_seen_at, created_at

### Service Layer

| Service | File | Purpose |
|---------|------|---------|
| **Auth** | `auth.py` | Token generation, hashing, normalization |
| **Chunking** | `chunking.py` | Text splitting with RecursiveCharacterTextSplitter |
| **Embeddings** | `embeddings.py` | OpenAI/OpenRouter embedding generation |
| **Ingest** | `ingest.py` | Email → Article pipeline orchestration |
| **LLM** | `llm.py` | All LLM interactions (summarize, classify, answer) |
| **MailerSend** | `mailersend.py` | Webhook payload parsing |
| **PDF** | `pdf.py` | PDF text extraction with pypdf |
| **Emailer** | `emailer.py` | SMTP email composition and sending |

### Hybrid Search Scoring

The search combines three signals into a hybrid score (0-1):

1. **Semantic Score** (base): `1 - cosine_distance(embedding, query_vector)`
2. **Keyword Boost**: `+0.03 * matches` (capped at 0.15)
3. **Lexical Score**: `ts_rank()` result (capped at 0.25)

**Formula**: `hybrid = min(1.0, semantic + keyword_boost + min(0.25, lexical))`

## Security Architecture

### Authentication Flow

1. User enters email at `/login`
2. System generates magic link token (32 bytes, URL-safe)
3. Token hash (SHA256) stored in `magic_link_tokens`
4. Magic link emailed to user
5. User clicks link at `/auth/callback?token=xyz`
6. Token verified, marked used, session created
7. Session cookie set (HTTP-only, Secure in prod)

### Security Measures

- Tokens stored as SHA256 hashes only
- Magic links expire in 15 minutes (configurable)
- Sessions expire in 7 days (configurable)
- All data scoped to authenticated user
- Input validation via Pydantic schemas
- CSRF protection via SameSite cookies

## API Endpoints

### Public Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/webhooks/mailersend` | Receive inbound emails |
| GET | `/login` | Login form |
| POST | `/login` | Request magic link |
| GET | `/auth/callback` | Magic link callback |

### Protected API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/articles` | List articles (JSON) |
| GET | `/api/articles/{id}` | Get article (JSON) |
| PUT | `/api/articles/{id}` | Update article (JSON) |
| DELETE | `/api/articles/{id}` | Delete article (JSON) |
| GET | `/api/search` | Hybrid search (JSON) |
| POST | `/chat/stream` | Streaming chat (SSE) |

### Protected UI Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/articles` | Article list/search |
| GET | `/articles/{id}` | Article detail |
| GET/POST | `/articles/{id}/edit` | Edit article |
| POST | `/articles/{id}/delete` | Delete article |
| GET/POST | `/chat` | Chat interface |

## Configuration

All configuration centralized in `app/config.py` using Pydantic Settings:

```python
class Settings(BaseSettings):
    app_env: str = "local"
    database_url: str
    embedding_dim: int = 1536
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    # ... etc
```

Environment variables loaded from `.env` file.

## Performance Considerations

### Database Indexes

- `idx_chunks_embedding`: IVFFlat index for vector search
- `idx_articles_search`: GIN index for full-text search
- `idx_articles_metadata`, `idx_chunks_metadata`: GIN for JSONB queries

### Optimization Strategies

- Connection pooling with `pool_pre_ping=True`
- Background tasks for ingestion (non-blocking webhooks)
- Prepared statements for repeated queries
- In-memory chat history (per-session)

## Scaling Considerations

| Component | Current | Horizontal Scaling |
|-----------|---------|-------------------|
| App | Single instance | Stateless - can scale |
| Database | Single PostgreSQL | Needs read replicas |
| Embeddings | Synchronous | Needs queue (Celery) |
| Chat Memory | In-memory dict | Needs Redis |

## Technology Stack

- **Framework**: FastAPI 0.115.5
- **ORM**: SQLAlchemy 2.0.36
- **Database**: PostgreSQL 14+ with pgvector
- **Embeddings**: OpenAI/OpenRouter via LangChain
- **PDF Processing**: pypdf 5.0.1
- **Templates**: Jinja2 3.1.4
- **Authentication**: Magic links with SMTP

See `requirements.txt` for complete dependency list.
