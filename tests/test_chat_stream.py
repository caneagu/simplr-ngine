from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main


class _DummyDB:
    pass


def _override_get_db():
    yield _DummyDB()


def test_chat_stream_requires_login() -> None:
    main.app.dependency_overrides[main.get_db] = _override_get_db
    try:
        client = TestClient(main.app)
        response = client.post("/chat/stream", json={"question": "hello"})
        assert response.status_code == 401
        assert response.json() == {"detail": "Login required"}
    finally:
        main.app.dependency_overrides.clear()


def test_chat_stream_stats_path_emits_expected_sse_events() -> None:
    main.app.dependency_overrides[main.get_db] = _override_get_db
    try:
        client = TestClient(main.app)
        with patch("app.main._get_current_user", return_value=SimpleNamespace(id="u1")), patch(
            "app.main._get_user_inference_config",
            return_value=None,
        ), patch("app.main._maybe_stats_answer", return_value="You have 3 article(s) in the knowledge base."):
            response = client.post("/chat/stream", json={"question": "how many articles do I have?"})

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = response.text
        assert "event: context" in body
        assert "event: refs" in body
        assert "event: answer" in body
        assert "event: usage" in body
        assert "event: done" in body
    finally:
        main.app.dependency_overrides.clear()
