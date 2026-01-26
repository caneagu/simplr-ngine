from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ArticleBase(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    title: str
    summary: str
    content_text: str
    metadata: dict[str, Any] = Field(
        default_factory=dict, validation_alias="metadata_", serialization_alias="metadata"
    )


class ArticleCreate(ArticleBase):
    pass


class ArticleUpdate(BaseModel):
    title: Optional[str] = None
    summary: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class ArticleRead(ArticleBase):
    id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None


class ChunkRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: UUID
    article_id: UUID
    chunk_index: int
    content: str
    metadata: dict[str, Any] = Field(
        default_factory=dict, validation_alias="metadata_", serialization_alias="metadata"
    )


class SourceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: UUID
    article_id: UUID
    source_type: str
    source_name: Optional[str] = None
    source_uri: Optional[str] = None
    raw_text: Optional[str] = None
    metadata: dict[str, Any] = Field(
        default_factory=dict, validation_alias="metadata_", serialization_alias="metadata"
    )
