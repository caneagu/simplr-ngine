# Simplr

A Retrieval-Augmented Generation (RAG) web application that ingests inbound emails via webhooks, extracts text and PDF content, and creates a searchable knowledge base with hybrid search (semantic + lexical) and conversational AI capabilities.

## ✨ Features

- 📧 **Email Ingestion**: Receives emails via MailerSend webhook with PDF attachment support
- 🔍 **Hybrid Search**: Combines vector similarity (pgvector) with PostgreSQL full-text search
- 🤖 **AI Chat**: Conversational interface with streaming responses and conversation memory
- 🔐 **Magic-Link Auth**: Secure passwordless authentication per email owner
- 📄 **PDF Extraction**: Automatic text extraction from PDF attachments
- 🏷️ **Auto-Categorization**: AI-powered content classification and metadata extraction
- 📝 **Article Management**: Create, read, update, delete knowledge base articles
- 📊 **Search Scoring**: Transparent similarity scores (semantic + lexical)

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   MailerSend    │────▶│  FastAPI App    │────▶│   PostgreSQL    │
│   (Webhook)     │     │                 │     │   + pgvector    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│   OpenAI/     │      │   Jinja2      │      │   SMTP        │
│   OpenRouter  │      │   Templates   │      │   (Magic Link)│
└───────────────┘      └───────────────┘      └───────────────┘
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- PostgreSQL 14+ with pgvector extension
- SMTP server (for magic links)
- OpenAI API key (or OpenRouter)

### 1. Clone and Setup

```bash
cd simplr
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Database Setup

```bash
# Create database
createdb rag_mail

# Enable extensions and create tables
psql rag_mail -f scripts/init_db.sql
```

### 3. Environment Configuration

```bash
cp .env.example .env
# Edit .env with your settings
```

**Required environment variables:**

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+psycopg2://user:pass@localhost/rag_mail` |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `SMTP_HOST` | SMTP server host | `smtp.mailersend.net` |
| `SMTP_USER` | SMTP username | `user@example.com` |
| `SMTP_PASSWORD` | SMTP password | `...` |
| `SMTP_SENDER` | From email address | `noreply@example.com` |

See [Environment Variables](#environment-variables) for complete list.

### 4. Run the Application

```bash
uvicorn app.main:app --reload
```

Access the application at `http://localhost:8000`

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture, data flow, and component overview |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Development setup, testing, and debugging |
| [API.md](API.md) | REST API and webhook documentation |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Production deployment guide |
| [PROMPTS.md](PROMPTS.md) | LLM prompt documentation and optimization tips |

## 🔧 Environment Variables

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ENV` | `local` | Environment: local, staging, production |
| `DATABASE_URL` | *required* | PostgreSQL connection string |
| `STORAGE_DIR` | `./storage` | File storage path for attachments |
| `EMBEDDING_DIM` | `1536` | Embedding dimension (1536 for text-embedding-3-small) |

### LLM Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | Provider: `openai` or `openrouter` |
| `LLM_MODEL` | `gpt-4o-mini` | LLM model name |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `OPENROUTER_API_KEY` | - | OpenRouter API key |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter base URL |

### Authentication

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_BASE_URL` | `http://localhost:8000` | Base URL for magic links |
| `MAGICLINK_TTL_MINUTES` | `15` | Magic link expiration time |
| `SESSION_DAYS` | `7` | Session cookie duration |
| `COOKIE_SECURE` | `false` | Use secure cookies (HTTPS only) |

### SMTP Configuration

| Variable | Description |
|----------|-------------|
| `SMTP_HOST` | SMTP server hostname |
| `SMTP_PORT` | SMTP port (default: 587) |
| `SMTP_USER` | SMTP username |
| `SMTP_PASSWORD` | SMTP password |
| `SMTP_SENDER` | Sender email address |
| `SMTP_SENDER_NAME` | Sender display name |
| `EMAIL_BRAND_NAME` | Brand name in emails (default: Simplr) |
| `EMAIL_LOGO_URL` | Logo URL for email templates |

## 🎯 Usage

### Webhook Setup (MailerSend)

1. In MailerSend dashboard, create an inbound route
2. Set webhook URL to: `https://your-domain.com/webhooks/mailersend`
3. Configure to forward emails to your application

### User Flow

1. **First Contact**: Send an email to the configured address
2. **Auto-Registration**: System creates user account from sender email
3. **Magic Link Login**: Visit `/login`, enter email, click magic link
4. **Access Articles**: Browse, search, and chat with your knowledge base

### Search & Chat

- **Hybrid Search**: Searches combine vector similarity + keyword matching
- **Chat Interface**: Conversational AI with context from your articles
- **Streaming Responses**: Real-time token-by-token responses
- **Conversation Memory**: Maintains context across messages (last 6 exchanges)

## 🧪 Testing

```bash
# Run tests
pytest

# Run specific test
pytest tests/test_mailersend.py -v
```

## 📦 Project Structure

```
simplr/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application & routes
│   ├── config.py         # Settings & configuration
│   ├── db.py             # Database connection
│   ├── models.py         # SQLAlchemy ORM models
│   ├── schemas.py        # Pydantic schemas
│   ├── services/         # Business logic
│   │   ├── auth.py       # Authentication utilities
│   │   ├── chunking.py   # Text chunking
│   │   ├── emailer.py    # SMTP email sending
│   │   ├── embeddings.py # Vector embeddings
│   │   ├── ingest.py     # Email ingestion pipeline
│   │   ├── llm.py        # LLM interactions
│   │   ├── mailersend.py # Webhook payload parsing
│   │   └── pdf.py        # PDF text extraction
│   ├── static/           # CSS, JS assets
│   └── templates/        # Jinja2 HTML templates
├── scripts/
│   ├── init_db.sql       # Database schema
│   ├── seed_dummy.py     # Seed test data
│   └── reset_seed.py     # Reset seed data
├── tests/                # Test suite
├── storage/              # File uploads (gitignored)
├── .env.example          # Environment template
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## 🔒 Security Considerations

- Magic links expire after 15 minutes (configurable)
- Sessions are HTTP-only, secure cookies
- All database queries are scoped to the authenticated user
- Token hashes stored (not raw tokens)
- Input validation on all endpoints

## 🛣️ Roadmap

- [ ] IMAP polling for email ingestion
- [ ] OCR/vision for image-based PDFs
- [ ] Multi-tenancy with tenant isolation
- [ ] Custom LLM provider registry
- [ ] API rate limiting
- [ ] Webhook signature verification

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

See [DEVELOPMENT.md](DEVELOPMENT.md) for contribution guidelines.

## 🆘 Support

For issues and questions:
1. Check [DEVELOPMENT.md](DEVELOPMENT.md) troubleshooting section
2. Review [API.md](API.md) for endpoint details
3. Open an issue on the repository
