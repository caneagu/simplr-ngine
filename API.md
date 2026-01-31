# API Documentation

Complete reference for all API endpoints, webhook payloads, and request/response schemas.

## Base URL

```
Local:      http://localhost:8000
Production: https://your-domain.com
```

## Authentication

Session-based authentication via HTTP-only cookies.

### Login Flow

1. Request magic link: `POST /login` with email
2. Receive magic link email
3. Click link: `GET /auth/callback?token=xyz`
4. Session cookie set automatically

### Session Cookie

- **Name**: `rag_session`
- **HTTP Only**: Yes
- **Secure**: Configurable (`COOKIE_SECURE`)
- **SameSite**: Lax
- **Max Age**: 7 days

## Webhooks

### POST `/webhooks/mailersend`

Receive inbound emails from MailerSend.

**Request Body**:
```json
{
  "data": {
    "from": {"email": "sender@example.com"},
    "subject": "Email Subject",
    "text": "Plain text content",
    "html": "<p>HTML content</p>",
    "attachments": [{
      "filename": "doc.pdf",
      "content_type": "application/pdf",
      "content": "base64-content"
    }],
    "headers": {"Message-ID": "<id@example.com>"},
    "id": "message-id"
  }
}
```

**Response** (202 Accepted):
```json
{"status": "accepted"}
```

**Duplicate** (200 OK):
```json
{"status": "duplicate", "article_id": "uuid"}
```

## REST API Endpoints

### Articles

#### GET `/api/articles`

List all articles for authenticated user.

**Response** (200 OK):
```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "title": "Article Title",
    "summary": "Bullet point summary...",
    "content_text": "Full content...",
    "metadata": {
      "sender": "sender@example.com",
      "category": "projects",
      "extracted": {"people": ["Alice"], "dates": ["2024-01-15"]}
    },
    "created_at": "2024-01-15T10:30:00",
    "updated_at": "2024-01-15T10:30:00"
  }
]
```

#### GET `/api/articles/{article_id}`

Get specific article.

**Response**: Same as list item.

**Error** (404):
```json
{"detail": "Article not found"}
```

#### PUT `/api/articles/{article_id}`

Update article metadata.

**Request Body**:
```json
{
  "title": "Updated Title",
  "summary": "Updated summary...",
  "metadata": {"category": "documentation", "tags": ["important"]}
}
```

**Response**: Updated article object.

#### DELETE `/api/articles/{article_id}`

Delete article.

**Response** (200 OK):
```json
{"status": "deleted"}
```

### Search

#### GET `/api/search`

Hybrid search combining vector similarity and full-text search.

**Query Parameters**:
- `query` (required): Search string
- `limit` (optional): Max results (default: 10)

**Example**:
```http
GET /api/search?query=quarterly%20revenue&limit=5
```

**Response** (200 OK):
```json
[
  {
    "article_id": "uuid",
    "title": "Q1 2024 Results",
    "summary": "Quarterly summary...",
    "chunk": "Revenue grew by 25%...",
    "category": "projects",
    "similarity": 0.893,
    "semantic_score": 0.812,
    "lexical_score": 0.156
  }
]
```

**Error** (400):
```json
{"detail": "Query is required"}
```

### Chat

#### POST `/chat/stream`

Streaming chat with Server-Sent Events.

**Request Body**:
```json
{"question": "What was our Q1 revenue?"}
```

**Response**: `text/event-stream`

**Event Format**:
```
event: refs
data: [{"id": "article-uuid", "title": "Article Title"}]

event: answer
data: "Our"

event: answer
data: " revenue"
...

event: done
data: {}
```

**JavaScript Client**:
```javascript
const es = new EventSource('/chat/stream', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({question: '...'})
});
es.addEventListener('answer', e => appendToken(JSON.parse(e.data)));
```

## UI Routes (HTML)

### Public Routes

| Route | Description |
|-------|-------------|
| `GET /login` | Login form |
| `POST /login` | Submit email for magic link |
| `GET /auth/callback` | Magic link callback |

### Protected Routes

| Route | Description |
|-------|-------------|
| `GET /articles` | Article list/search page |
| `GET /articles/{id}` | Article detail page |
| `GET/POST /articles/{id}/edit` | Edit article form |
| `POST /articles/{id}/delete` | Delete article |
| `GET/POST /chat` | Chat interface |
| `POST /logout` | End session |

**Query Parameters for `/articles`**:
- `query`: Search query (optional)
- `category`: Filter by category (optional)
- Categories: `all`, `support_tickets`, `policies`, `documentation`, `projects`, `other`, `uncategorized`

## Error Responses

### HTTP Status Codes

| Status | Meaning |
|--------|---------|
| 200 | OK |
| 202 | Accepted (webhook) |
| 303 | Redirect |
| 400 | Bad Request |
| 401 | Unauthorized |
| 404 | Not Found |
| 500 | Server Error |

### Error Format (JSON)
```json
{"detail": "Error message"}
```

## Data Schemas

### ArticleRead

```json
{
  "id": "uuid",
  "title": "string",
  "summary": "string",
  "content_text": "string",
  "metadata": {
    "sender": "string",
    "subject": "string",
    "category": "string",
    "extracted": {
      "people": ["string"],
      "dates": ["string"],
      "timeline": "string",
      "progress": "string"
    },
    "entities": {
      "people": ["string"],
      "orgs": ["string"],
      "locations": ["string"],
      "tags": ["string"]
    }
  },
  "created_at": "ISO-8601 datetime",
  "updated_at": "ISO-8601 datetime"
}
```

## Interactive Documentation

FastAPI auto-generates interactive docs:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`
