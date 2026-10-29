"""Typed domain models shared across the Email Analytics Agent layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp in stable ISO-8601 form."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class EmailMessage:
    """Represent one raw email as received from the mocked mail connector."""

    message_id: str
    sender: str
    subject: str
    body: str
    received_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EmailMessage":
        """Build an email from JSON-compatible input while preserving metadata."""

        known = {"message_id", "sender", "subject", "body", "received_at"}
        metadata = dict(value.get("metadata", {}))
        metadata.update({key: item for key, item in value.items() if key not in known | {"metadata"}})
        return cls(
            message_id=str(value["message_id"]),
            sender=str(value["sender"]),
            subject=str(value["subject"]),
            body=str(value["body"]),
            received_at=str(value["received_at"]),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible representation of the email."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class CTDLabel:
    """Represent a Category / Type / Department routing classification."""

    category: str
    request_type: str
    department: str

    def to_dict(self) -> dict[str, str]:
        """Serialize the CTD triplet for storage and auditing."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class Prediction:
    """Capture one classifier result, its confidence, and model provenance."""

    label: CTDLabel
    confidence: float
    source: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize a prediction without exposing any source email content."""

        value = asdict(self)
        return value


@dataclass(frozen=True, slots=True)
class ExtractedEntity:
    """Describe a sensitive entity discovered before text redaction."""

    entity_type: str
    masked_value: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class PreprocessingResult:
    """Contain only redacted model text plus non-reversible entity metadata."""

    redacted_subject: str
    redacted_body: str
    entities: tuple[ExtractedEntity, ...]
    language: str

    @property
    def model_text(self) -> str:
        """Combine safe subject and body text for downstream model calls."""

        return f"Subject: {self.redacted_subject}\nBody: {self.redacted_body}"


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """Return the observable outcome of a completed orchestration run."""

    run_id: str
    message_id: str
    decision: str
    prediction: Prediction
    target_queue: str
    summary: str
    decision_path: tuple[str, ...]
    risk_flags: tuple[str, ...] = ()
    confidence_scores: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the outcome for the CLI and integration boundaries."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class QueueItem:
    """Represent the PII-safe payload emitted to an operational queue."""

    run_id: str
    message_id: str
    target_queue: str
    summary: str
    prediction: Prediction
    risk_flags: tuple[str, ...]
    enqueued_at: str = field(default_factory=utc_now_iso)

