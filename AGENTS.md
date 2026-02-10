# AGENTS.md

This document defines how coding agents should work in this repository.

## 1. Mission
- Maintain and improve a FastAPI RAG app for email ingestion, search, and chat.
- Prioritize correctness, data safety, and user-scoped access control.
- Keep changes small, testable, and easy to review.

## 2. Tech Snapshot
- Python 3.9+
- FastAPI + Jinja2
- SQLAlchemy 2.x
- PostgreSQL + pgvector
- pytest

Key paths:
- `app/main.py`: routes, auth/session checks, search/chat endpoints, startup schema guards
- `app/services/`: business logic (ingest, auth, embeddings, llm, webhook parsing, pdf)
- `app/models.py`, `app/db.py`, `scripts/init_db.sql`: persistence layer
- `tests/`: automated tests (currently minimal)

## 3. Core Rules
- Never bypass user scoping: data reads/writes must remain constrained to the authenticated user (or explicit group membership rules).
- Keep API and UI behavior aligned: when changing route behavior, verify both JSON and template-driven paths if applicable.
- Prefer dependency injection/mocking for external providers (LLM, embeddings, SMTP, webhook providers).
- Do not silently change environment variable semantics. Document config changes.
- If a task touches auth, ingestion, or deletion flows, add/adjust tests in the same PR.

## 4. Standard Workflow
1. Understand scope
- Identify touched modules and side effects (DB, auth, external calls, templates).

2. Plan minimally
- Prefer smallest safe diff.
- Avoid broad refactors unless requested.

3. Implement
- Keep functions cohesive.
- Add brief comments only where logic is non-obvious.

4. Verify
- Run targeted tests first, then broader regression checks.

5. Summarize
- State what changed, why, and what was tested.

## 5. Regression Testing Strategy
Use this layered approach for every meaningful change.

### Layer A: Fast targeted tests (always)
- Run tests for touched area first.
- Example:
```bash
PYTHONPATH=. ./bin/pytest -q tests/test_mailersend.py
```

### Layer B: Full automated suite (default before handoff)
```bash
PYTHONPATH=. ./bin/pytest -q
```

### Layer C: Risk-based manual regression (when relevant)
Run local app:
```bash
uvicorn app.main:app --reload
```

Then validate impacted flows:
- Auth flow: login request, callback, protected page access.
- Ingestion flow: webhook payload acceptance/rejection and article creation behavior.
- Search/chat flow: query still returns scoped results; streaming/chat endpoint remains functional.
- CRUD flow: edit/delete/list behavior and ownership checks.

## 6. Mandatory Tests by Change Type
When editing these areas, include at least these regressions.

### `app/services/mailersend.py`
- Payload parsing with `data` envelope and direct payload format.
- Header/message-id normalization behavior.
- Missing/optional fields do not crash parser.

### `app/services/auth.py` or session/magic-link code in `app/main.py`
- Token hashing/verification logic.
- Expired token/session behavior.
- Unauthorized requests return expected status.

### `app/services/ingest.py`
- Duplicate handling (dedupe keys / idempotency expectations).
- Attachment parsing fallback behavior.
- Failures surface cleanly (no partial inconsistent state).

### Search/chat code in `app/main.py` and `app/services/llm.py`
- Query still respects ownership/group scope.
- Ranking/filter changes do not expose other users' data.
- Streaming endpoint behavior still returns expected event format.

### DB schema/index logic (`scripts/init_db.sql`, startup DDL in `app/main.py`)
- New schema changes are additive/safe for existing data.
- Index creation remains idempotent.

## 7. Test Design Guidelines
- Prefer unit tests for service logic; integration tests for route-level auth/scope checks.
- Mock external services (OpenAI/OpenRouter, SMTP, remote PDF fetch).
- Use explicit fixtures for payloads and edge cases.
- Include one success case and one failure/edge case per changed behavior.

## 8. Definition of Done (DoD)
A change is done when:
- Code is implemented and readable.
- Tests for changed behavior exist/updated.
- `pytest -q` passes locally (or documented if blocked).
- No obvious auth/scope regression introduced.
- Docs updated for behavior/config changes.

## 9. Recommended Next Test Additions (High ROI)
Current suite is thin. Prioritize adding:
1. Route-level auth tests for protected endpoints in `app/main.py`.
2. Ingestion pipeline tests with mocked embeddings/LLM/PDF functions.
3. Search scoping tests to prevent cross-user leakage.
4. Delete/edit ownership tests.

## 10. Agent Handoff Template
Use this in PR/task summaries:
- Change: <what was changed>
- Why: <bug/risk/feature>
- Risk: <main regression risk>
- Tests run: <commands + result>
- Follow-ups: <optional>

Command note:
- If `pytest` is not on PATH in this repo, use `PYTHONPATH=. ./bin/pytest`.
