# Deployment Guide

Complete guide for deploying the Simplr to production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Production Checklist](#production-checklist)
3. [Deployment Options](#deployment-options)
4. [Database Setup](#database-setup)
5. [Environment Configuration](#environment-configuration)
6. [Security Hardening](#security-hardening)
7. [Monitoring & Logging](#monitoring--logging)
8. [Backup & Recovery](#backup--recovery)
9. [Troubleshooting](#troubleshooting)

## Prerequisites

### Infrastructure Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 1 core | 2+ cores |
| **RAM** | 1 GB | 2+ GB |
| **Storage** | 10 GB | 50+ GB |
| **Database** | PostgreSQL 14 | PostgreSQL 15+ |
| **Network** | Stable internet | Dedicated bandwidth |

### External Services

- **SMTP Provider**: Mailgun, SendGrid, AWS SES, or MailerSend
- **LLM Provider**: OpenAI or OpenRouter account with API key
- **Domain**: Configured with DNS records
- **SSL Certificate**: Let's Encrypt or purchased certificate

## Production Checklist

Before deploying to production, verify:

### Security
- [ ] Strong database password (32+ chars)
- [ ] `COOKIE_SECURE=true` in production
- [ ] `APP_ENV=production` set
- [ ] API keys rotated and secured
- [ ] SMTP credentials use app-specific passwords
- [ ] Webhook signature verification implemented
- [ ] HTTPS only (HSTS headers)
- [ ] Database not exposed to internet

### Configuration
- [ ] `APP_BASE_URL` uses HTTPS domain
- [ ] `DATABASE_URL` points to production DB
- [ ] `STORAGE_DIR` is persistent volume
- [ ] SMTP settings verified with test email
- [ ] LLM API keys have sufficient quota

### Monitoring
- [ ] Error tracking configured (Sentry)
- [ ] Application logs shipping to aggregator
- [ ] Database monitoring enabled
- [ ] Health check endpoint accessible
- [ ] Uptime monitoring configured

### Backups
- [ ] Automated database backups scheduled
- [ ] Backup restoration tested
- [ ] File storage backed up (attachments)
- [ ] Disaster recovery plan documented

## Deployment Options

### Option 1: Docker (Recommended)

#### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/
COPY scripts/ ./scripts/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

#### docker-compose.yml

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - APP_ENV=production
      - DATABASE_URL=postgresql+psycopg2://rag_user:${DB_PASSWORD}@db:5432/rag_mail
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - SMTP_HOST=${SMTP_HOST}
      - SMTP_USER=${SMTP_USER}
      - SMTP_PASSWORD=${SMTP_PASSWORD}
      - APP_BASE_URL=https://your-domain.com
      - COOKIE_SECURE=true
    volumes:
      - app-storage:/app/storage
    depends_on:
      - db
    restart: unless-stopped

  db:
    image: pgvector/pgvector:pg15
    environment:
      - POSTGRES_USER=rag_user
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_DB=rag_mail
    volumes:
      - postgres-data:/var/lib/postgresql/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - app
    restart: unless-stopped

volumes:
  postgres-data:
  app-storage:
```

#### nginx.conf

```nginx
events {
    worker_connections 1024;
}

http {
    upstream app {
        server app:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=webhook:10m rate=100r/m;

    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;

        # Security headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

        # Webhook endpoint with higher rate limit
        location /webhooks/ {
            limit_req zone=webhook burst=20 nodelay;
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # API endpoints
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Static files
        location /static/ {
            alias /app/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }

        # All other routes
        location / {
            limit_req zone=api burst=10 nodelay;
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

#### Deployment Steps

```bash
# 1. Clone repository on server
git clone <repo-url>
cd simplr

# 2. Create environment file
cat > .env << EOF
DB_PASSWORD=your-secure-password
OPENAI_API_KEY=sk-...
SMTP_HOST=smtp.mailgun.org
SMTP_USER=postmaster@your-domain.com
SMTP_PASSWORD=...
EOF

# 3. Build and start
docker-compose up -d --build

# 4. Run database migrations
docker-compose exec db psql -U rag_user -d rag_mail -f /app/scripts/init_db.sql

# 5. Verify health
curl https://your-domain.com/health
```

### Option 2: Railway / Render (Platform-as-a-Service)

#### Railway Deployment

1. **Create Project**: Connect GitHub repository
2. **Add PostgreSQL**: Railway provides managed PostgreSQL with pgvector
3. **Environment Variables**:
   ```
   APP_ENV=production
   DATABASE_URL=${{Postgres.DATABASE_URL}}
   OPENAI_API_KEY=sk-...
   SMTP_HOST=smtp.mailgun.org
   ...
   ```
4. **Deploy**: Automatic on git push

#### Render Deployment

1. **Create Web Service**: Connect repository
2. **Select Runtime**: Python 3
3. **Build Command**: `pip install -r requirements.txt`
4. **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. **Add PostgreSQL**: Create managed PostgreSQL instance
6. **Environment Variables**: Add all required vars

### Option 3: AWS Deployment

#### Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Route 53  │────▶│ CloudFront  │────▶│    ALB      │
│   (DNS)     │     │   (CDN)     │     │  (HTTPS)    │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                                │
                                       ┌────────▼────────┐
                                       │   ECS/Fargate   │
                                       │  (FastAPI App)  │
                                       └────────┬────────┘
                                                │
                                       ┌────────▼────────┐
                                       │   RDS Postgres  │
                                       │    + pgvector   │
                                       └─────────────────┘
```

#### Terraform Example (Simplified)

```hcl
# main.tf - Simplified AWS deployment
resource "aws_ecs_cluster" "main" {
  name = "rag-email-cluster"
}

resource "aws_ecs_task_definition" "app" {
  family                   = "rag-email-app"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"

  container_definitions = jsonencode([{
    name  = "app"
    image = "your-registry/simplr:latest"
    portMappings = [{
      containerPort = 8000
    }]
    environment = [
      { name = "APP_ENV", value = "production" },
      { name = "DATABASE_URL", value = aws_db_instance.main.endpoint },
    ]
    secrets = [
      { name = "OPENAI_API_KEY", valueFrom = aws_secretsmanager_secret.openai.arn },
    ]
  }])
}

resource "aws_db_instance" "main" {
  identifier     = "rag-email-db"
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.t3.micro"
  
  # Note: pgvector requires custom parameter group or manual installation
}
```

## Database Setup

### Managed PostgreSQL with pgvector

#### AWS RDS

pgvector not available in RDS. Options:
1. Use RDS + EC2 with pgvector extension
2. Use Aurora PostgreSQL (self-managed extensions)
3. Use managed service: Supabase, Neon, or Timescale Cloud

#### Supabase (Recommended)

1. Create project at [supabase.com](https://supabase.com)
2. pgvector pre-installed
3. Copy connection string from Settings → Database

#### Neon

1. Create project at [neon.tech](https://neon.tech)
2. Enable pgvector: `CREATE EXTENSION vector;`
3. Use connection string in environment variables

### Self-Hosted PostgreSQL

```bash
# Ubuntu 22.04 installation
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
sudo apt-get update
sudo apt-get install postgresql-15 postgresql-15-pgvector

# Configure
sudo -u postgres psql -c "CREATE USER rag_user WITH PASSWORD 'secure-password';"
sudo -u postgres psql -c "CREATE DATABASE rag_mail OWNER rag_user;"
sudo -u postgres psql -d rag_mail -c "CREATE EXTENSION vector;"
```

### Database Optimization

```sql
-- Connection pooling (PgBouncer recommended for high traffic)
-- Max connections: 100 for db.t3.micro, scale up for production

-- Autovacuum settings for high write volume
ALTER SYSTEM SET autovacuum_vacuum_scale_factor = 0.1;
ALTER SYSTEM SET autovacuum_analyze_scale_factor = 0.05;

-- Work memory for queries
ALTER SYSTEM SET work_mem = '16MB';
```

## Environment Configuration

### Production .env Template

```env
# Core
APP_ENV=production
APP_BASE_URL=https://your-domain.com
DEBUG=false

# Database (use connection pooler for high traffic)
DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/dbname

# Storage
STORAGE_DIR=/app/storage

# LLM
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-production-key

# Auth
MAGICLINK_TTL_MINUTES=15
SESSION_DAYS=7
COOKIE_SECURE=true

# SMTP (use reputable provider)
SMTP_HOST=smtp.mailgun.org
SMTP_PORT=587
SMTP_USER=postmaster@mg.your-domain.com
SMTP_PASSWORD=secure-smtp-password
SMTP_SENDER=noreply@your-domain.com
SMTP_SENDER_NAME=Your App Name
EMAIL_LOGO_URL=https://your-domain.com/logo.png
EMAIL_BRAND_NAME=Your Brand
```

### Secret Management

#### AWS Secrets Manager

```bash
# Store secrets
aws secretsmanager create-secret \
    --name rag-email/production \
    --secret-string file://secrets.json

# Retrieve in application (requires IAM role)
```

#### HashiCorp Vault

```bash
# Write secrets
vault kv put secret/rag-email \
    database_url="postgresql://..." \
    openai_key="sk-..."

# Application integration
export DATABASE_URL=$(vault kv get -field=database_url secret/rag-email)
```

#### Docker Secrets (Swarm)

```yaml
# docker-compose.yml (Swarm mode)
secrets:
  openai_key:
    external: true

services:
  app:
    secrets:
      - openai_key
    environment:
      - OPENAI_API_KEY_FILE=/run/secrets/openai_key
```

## Security Hardening

### Webhook Security

Add signature verification to prevent spoofing:

```python
# app/services/mailersend.py
import hmac
import hashlib

def verify_webhook_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify MailerSend webhook signature."""
    expected = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)

# In main.py route
@app.post("/webhooks/mailersend")
async def mailersend_webhook(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Mailersend-Signature")
    
    if not verify_webhook_signature(body, signature, settings.webhook_secret):
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    payload = json.loads(body)
    # ... process
```

### Database Security

1. **Use SSL**: `sslmode=require` in connection string
2. **Network Isolation**: Database in private subnet
3. **Connection Limits**: Set per-user limits
4. **Audit Logging**: Enable PostgreSQL audit logs

```sql
-- SSL enforcement
ALTER SYSTEM SET ssl = on;

-- Connection limit per user
ALTER USER rag_user CONNECTION LIMIT 20;

-- Audit logging (pgaudit extension)
CREATE EXTENSION IF NOT EXISTS pgaudit;
ALTER SYSTEM SET pgaudit.log = 'write, ddl';
```

### Application Security

1. **Dependency Scanning**:
   ```bash
   pip install safety
   safety check -r requirements.txt
   ```

2. **Container Scanning**:
   ```bash
   docker scan your-image:tag
   ```

3. **Secrets Detection**:
   ```bash
   # Pre-commit hook
   pip install detect-secrets
   detect-secrets scan > .secrets.baseline
   ```

## Monitoring & Logging

### Health Check Endpoint

Add to `app/main.py`:

```python
@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Health check endpoint for load balancers."""
    try:
        db.execute("SELECT 1")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
```

### Structured Logging

```python
# app/config.py
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# JSON logging for production
if settings.app_env == "production":
    import pythonjsonlogger.jsonlogger
    formatter = pythonjsonlogger.jsonlogger.JsonFormatter(
        "%(timestamp)s %(level)s %(name)s %(message)s"
    )
```

### Error Tracking (Sentry)

```python
# app/main.py
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

if settings.app_env == "production":
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        integrations=[
            FastApiIntegration(),
            SqlalchemyIntegration(),
        ],
        traces_sample_rate=0.1,
    )
```

### Metrics (Prometheus)

```python
# app/main.py
from prometheus_client import Counter, Histogram, generate_latest

request_count = Counter("requests_total", "Total requests", ["method", "endpoint"])
request_duration = Histogram("request_duration_seconds", "Request duration")

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    
    request_count.labels(
        method=request.method,
        endpoint=request.url.path
    ).inc()
    request_duration.observe(duration)
    
    return response

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

## Backup & Recovery

### Database Backups

#### Automated Daily Backups

```bash
#!/bin/bash
# backup.sh - Run via cron daily

BACKUP_DIR="/backups/postgres"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="rag_mail"

docker exec rag-email-db pg_dump -U rag_user $DB_NAME | gzip > "$BACKUP_DIR/rag_mail_$DATE.sql.gz"

# Keep only last 30 days
find $BACKUP_DIR -name "rag_mail_*.sql.gz" -mtime +30 -delete

# Upload to S3 (optional)
aws s3 cp "$BACKUP_DIR/rag_mail_$DATE.sql.gz" s3://your-backup-bucket/
```

#### Point-in-Time Recovery (RDS)

Enable automated backups in RDS:
- Backup retention: 7-35 days
- Backup window: Low-traffic period
- Enable cross-region copy for disaster recovery

### File Storage Backups

```bash
# Sync attachments to S3
aws s3 sync /app/storage/attachments s3://your-bucket/attachments/ \
    --delete \
    --storage-class STANDARD_IA
```

### Recovery Procedures

#### Database Restore

```bash
# From pg_dump
gunzip -c rag_mail_20240115.sql.gz | docker exec -i rag-email-db psql -U rag_user -d rag_mail

# From RDS snapshot
aws rds restore-db-instance-from-db-snapshot \
    --db-instance-identifier rag-email-db-restored \
    --db-snapshot-identifier rag-mail-snapshot
```

## Troubleshooting

### Common Production Issues

#### High Memory Usage

**Symptoms**: Container OOM killed

**Solutions**:
1. Reduce chunk size for embeddings
2. Limit concurrent chat streams
3. Add memory limits to containers
4. Enable swap (emergency only)

```python
# Limit chunk processing
MAX_CHUNKS_PER_ARTICLE = 100
```

#### Slow Search Queries

**Symptoms**: Search taking >2 seconds

**Solutions**:
1. Verify index usage: `EXPLAIN ANALYZE`
2. Increase `ivfflat.probes` for better recall
3. Add read replicas for search queries
4. Implement caching for popular queries

```sql
-- Tune vector index
SET ivfflat.probes = 10;  -- Default is 1
```

#### SMTP Failures

**Symptoms**: Magic links not sent

**Solutions**:
1. Check SMTP credentials
2. Verify sender domain DNS (SPF, DKIM, DMARC)
3. Check provider reputation
4. Implement retry with exponential backoff

#### Webhook Timeouts

**Symptoms**: MailerSend retries webhooks

**Solutions**:
1. Ensure 202 response within 10 seconds
2. Verify background tasks are running
3. Check database connection pool
4. Add webhook queue (Celery/RQ)

### Log Analysis

```bash
# Find errors
docker-compose logs app | grep ERROR

# Slow queries
docker-compose exec db psql -U rag_user -c "SELECT query, mean_exec_time FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;"

# Active connections
docker-compose exec db psql -U rag_user -c "SELECT count(*), state FROM pg_stat_activity GROUP BY state;"
```

### Performance Tuning

```ini
# postgresql.conf optimizations
shared_buffers = 256MB
effective_cache_size = 768MB
work_mem = 16MB
maintenance_work_mem = 64MB
max_connections = 100
```

## Maintenance Windows

### Regular Tasks

| Task | Frequency | Command/Action |
|------|-----------|----------------|
| Database backup | Daily | Automated script |
| Log rotation | Daily | `logrotate` |
| Dependency updates | Weekly | `pip list --outdated` |
| Security patches | As needed | `apt-get update` |
| Database vacuum | Weekly | `VACUUM ANALYZE;` |
| Index rebuild | Monthly | `REINDEX DATABASE;` |

### Deployment Process

1. **Pre-deployment**:
   - Run tests: `pytest`
   - Build container: `docker build .`
   - Scan for vulnerabilities

2. **Deployment**:
   - Backup database
   - Deploy to staging first
   - Run smoke tests
   - Deploy to production
   - Monitor for 30 minutes

3. **Rollback**:
   - `docker-compose up -d --no-deps --build app` (previous image)
   - Restore database if needed
