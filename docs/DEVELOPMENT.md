# Development Guide

Development setup, testing, and contribution guidelines.

## Setup

### Prerequisites

```bash
# macOS
brew install postgresql@14
brew services start postgresql@14

# Ubuntu
sudo apt-get install postgresql-14 postgresql-contrib
sudo systemctl start postgresql
```

### Environment Setup

```bash
git clone <repo-url>
cd simplr
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Database

```bash
createdb rag_mail
psql rag_mail -f scripts/init_db.sql
```

### Environment Variables

```bash
cp .env.example .env
# Edit .env with your settings
```

### Run Server

```bash
uvicorn app.main:app --reload
```

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=app --cov-report=html

# Specific test
pytest tests/test_mailersend.py -v
```

## Code Structure

```
app/
├── main.py           # FastAPI app & routes
├── config.py         # Settings
├── db.py             # Database connection
├── models.py         # ORM models
├── schemas.py        # Pydantic schemas
├── services/         # Business logic
│   ├── auth.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── ingest.py
│   ├── llm.py
│   ├── mailersend.py
│   └── pdf.py
├── static/           # CSS, JS
└── templates/        # HTML templates
```

## Debugging

### Enable SQL Logging

```python
# app/db.py
import logging
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
```

### Test Webhook Locally

```bash
# Using ngrok
ngrok http 8000

# Test manually
curl -X POST http://localhost:8000/webhooks/mailersend \
  -H "Content-Type: application/json" \
  -d '{"data":{"from":{"email":"test@test.com"},"subject":"Test","text":"Hello"}}'
```

### Database Queries

```bash
psql rag_mail
\dt                    # List tables
\d articles            # Describe table
SELECT * FROM users;   # Query data
```

## Common Issues

### pgvector not found

```bash
# macOS
brew install pgvector

# Ubuntu
sudo apt-get install postgresql-14-pgvector
```

### Embedding dimension mismatch

1. Check `EMBEDDING_DIM` in `.env`
2. Recreate chunks table if needed:
```sql
TRUNCATE chunks;
ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(1536);
```

### SMTP connection failed

Use local debug server for testing:
```bash
python -m smtpd -c DebuggingServer -n localhost:1025
```

## Contributing

### Pull Request Process

1. Create branch: `git checkout -b feature/name`
2. Make changes with tests
3. Run tests: `pytest`
4. Commit with clear messages
5. Push and create PR

### Commit Format

```
type(scope): description

Types: feat, fix, docs, style, refactor, test, chore
```

Examples:
- `feat(search): add category filtering`
- `fix(ingest): handle missing message_id`
- `docs(readme): update env vars`

## Best Practices

- Use type hints (Python 3.9+ syntax)
- Add docstrings for functions
- Scope queries to authenticated user
- Hash tokens before storage
- Validate input with Pydantic

## Resources

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [SQLAlchemy 2.0](https://docs.sqlalchemy.org/en/20/)
- [pgvector](https://github.com/pgvector/pgvector)
- [LangChain](https://python.langchain.com/)
