from __future__ import annotations

import uuid
from types import SimpleNamespace

from app.main import _article_accessible, _folder_accessible


def test_article_accessible_in_personal_scope() -> None:
    user_id = uuid.uuid4()
    user = SimpleNamespace(id=user_id)
    article = SimpleNamespace(owner_id=user_id, group_id=None)
    assert _article_accessible(article, user, set())


def test_article_accessible_in_group_scope() -> None:
    user = SimpleNamespace(id=uuid.uuid4())
    group_id = uuid.uuid4()
    article = SimpleNamespace(owner_id=uuid.uuid4(), group_id=group_id)
    assert _article_accessible(article, user, {group_id})


def test_article_not_accessible_without_matching_owner_or_group() -> None:
    user = SimpleNamespace(id=uuid.uuid4())
    article = SimpleNamespace(owner_id=uuid.uuid4(), group_id=None)
    assert not _article_accessible(article, user, set())


def test_folder_accessible_in_personal_scope() -> None:
    user_id = uuid.uuid4()
    user = SimpleNamespace(id=user_id)
    folder = SimpleNamespace(owner_id=user_id, group_id=None)
    assert _folder_accessible(folder, user, set())


def test_folder_not_accessible_without_group_membership() -> None:
    user = SimpleNamespace(id=uuid.uuid4())
    folder = SimpleNamespace(owner_id=uuid.uuid4(), group_id=uuid.uuid4())
    assert not _folder_accessible(folder, user, set())
