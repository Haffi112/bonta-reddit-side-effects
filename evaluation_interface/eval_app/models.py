from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


db = SQLAlchemy(model_class=Base)


def utc_now() -> datetime:
    return datetime.now(UTC)


class Annotator(db.Model):
    __tablename__ = "annotators"

    id: Mapped[int] = mapped_column(primary_key=True)
    label: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    key_digest: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    annotations: Mapped[list[Annotation]] = relationship(
        back_populates="annotator", cascade="all, delete-orphan"
    )


class Submission(db.Model):
    __tablename__ = "submissions"

    id: Mapped[int] = mapped_column(primary_key=True)
    record_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    selftext: Mapped[str] = mapped_column(Text, nullable=False)
    source_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    outputs: Mapped[list[ModelOutput]] = relationship(
        back_populates="submission", cascade="all, delete-orphan"
    )


class ModelOutput(db.Model):
    __tablename__ = "model_outputs"
    __table_args__ = (UniqueConstraint("submission_id", "model_key"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    submission_id: Mapped[int] = mapped_column(
        ForeignKey("submissions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_key: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    model_label: Mapped[str] = mapped_column(String(200), nullable=False)
    blinded_id: Mapped[str] = mapped_column(String(24), unique=True, nullable=False, index=True)
    random_order: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    extraction: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    extraction_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    submission: Mapped[Submission] = relationship(back_populates="outputs")
    annotations: Mapped[list[Annotation]] = relationship(
        back_populates="model_output", cascade="all, delete-orphan"
    )


class Annotation(db.Model):
    __tablename__ = "annotations"
    __table_args__ = (UniqueConstraint("annotator_id", "model_output_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    annotator_id: Mapped[int] = mapped_column(
        ForeignKey("annotators.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_output_id: Mapped[int] = mapped_column(
        ForeignKey("model_outputs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    decisions: Mapped[dict[str, str]] = mapped_column(JSON, nullable=False)
    notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    annotator: Mapped[Annotator] = relationship(back_populates="annotations")
    model_output: Mapped[ModelOutput] = relationship(back_populates="annotations")
