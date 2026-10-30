"""In-memory adapters that model MongoDB and a vector policy database."""

from __future__ import annotations

import math
import re
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from threading import RLock
from typing import Any, Protocol

from .models import CTDLabel, EmailMessage, Prediction, ProcessResult, utc_now_iso


class ProcessingStatus(str, Enum):
    """Describe the durable business-processing state of one email."""

    RAW_SAVED = "RAW_SAVED"
    PROCESSING = "PROCESSING"
    AUTO_ROUTED = "AUTO_ROUTED"
    HUMAN_REVIEW_PENDING = "HUMAN_REVIEW_PENDING"
    COMPLETED = "COMPLETED"
    RETRY_PENDING = "RETRY_PENDING"
    QUARANTINED = "QUARANTINED"


class AuditStatus(str, Enum):
    """Describe whether a finalized outcome reached the audit ledger."""

    NOT_STARTED = "NOT_STARTED"
    PENDING = "PENDING"
    APPENDED = "APPENDED"
    FAILED = "FAILED"


class ClaimAction(str, Enum):
    """Tell an ingestion worker how to handle an atomically claimed email."""

    NEW = "NEW"
    RESUME = "RESUME"
    AUDIT_ONLY = "AUDIT_ONLY"
    IN_PROGRESS = "IN_PROGRESS"
    SKIP = "SKIP"


@dataclass(frozen=True, slots=True)
class EmailClaim:
    """Return the result of the document store's atomic idempotency check."""

    action: ClaimAction
    idempotency_key: str
    message_id: str
    status: ProcessingStatus
    run_id: str


class EmailDocumentStore(Protocol):
    """Define persistence operations required by the orchestration layer."""

    def claim_email(self, email: EmailMessage, run_id: str) -> EmailClaim:
        """Atomically insert, resume, audit, or reject one email workflow."""

    def save_email(self, email: EmailMessage) -> None:
        """Persist raw email content in the secured document-store boundary."""

    def save_classification(self, message_id: str, prediction: Prediction) -> None:
        """Attach the final CTD prediction to an existing email document."""

    def mark_processing(self, message_id: str, run_id: str) -> None:
        """Mark a claimed raw email as actively processed by one workflow run."""

    def save_outcome(self, result: ProcessResult) -> None:
        """Persist a routed or escalated outcome before its audit append."""

    def mark_audited(self, message_id: str, run_id: str) -> None:
        """Record successful audit persistence and close the automated run."""

    def mark_audit_failed(self, message_id: str, run_id: str, error_code: str) -> None:
        """Preserve an outcome whose audit append requires a later retry."""

    def mark_retry_pending(
        self, message_id: str, run_id: str, stage: str, error_code: str
    ) -> None:
        """Release a processing lease and retain a safe retry checkpoint."""

    def is_completed(self, message_id: str) -> bool:
        """Return whether ingestion should skip the automated workflow."""

    def load_outcome(self, message_id: str) -> ProcessResult | None:
        """Rehydrate a saved safe outcome for an audit-only retry."""

    def quarantine_ingestion_failure(self, source_key: str, error_code: str) -> None:
        """Durably record a safe conversion failure before cursor advancement."""


class MockMongoStore:
    """Simulate MongoDB workflow state, atomic claims, and detached reads."""

    _TERMINAL_STATUSES = {
        ProcessingStatus.COMPLETED,
        ProcessingStatus.HUMAN_REVIEW_PENDING,
        ProcessingStatus.QUARANTINED,
    }

    def __init__(self, lease_seconds: int = 300) -> None:
        """Initialize an empty collection and processing-lease duration."""

        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        self._documents: dict[str, dict[str, Any]] = {}
        self._message_index: dict[str, set[str]] = {}
        self._quarantined_failures: dict[str, dict[str, Any]] = {}
        self._lease_seconds = lease_seconds
        self._lock = RLock()

    @staticmethod
    def _idempotency_key(email: EmailMessage) -> str:
        """Build a stable key from Graph scope while supporting legacy inputs."""

        explicit = email.metadata.get("idempotency_key")
        if explicit:
            return str(explicit)
        tenant_id = email.metadata.get("tenant_id")
        mailbox_id = email.metadata.get("mailbox_id")
        immutable_id = (
            email.metadata.get("immutable_message_id")
            or email.metadata.get("graph_message_id")
            or email.message_id
        )
        if tenant_id and mailbox_id:
            return f"{tenant_id}:{mailbox_id}:{immutable_id}"
        return email.message_id

    @staticmethod
    def _now() -> datetime:
        """Return the current aware UTC instant for lease comparisons."""

        return datetime.now(timezone.utc)

    def _lease_expiration(self) -> str:
        """Return the next processing-lease expiry in stable ISO form."""

        expires = self._now() + timedelta(seconds=self._lease_seconds)
        return expires.isoformat().replace("+00:00", "Z")

    @staticmethod
    def _parse_timestamp(value: str | None) -> datetime | None:
        """Parse one stored UTC timestamp, tolerating an absent lease."""

        if not value:
            return None
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    def _resolve_key(self, message_id: str) -> str:
        """Resolve a legacy message ID or an explicit idempotency key."""

        if message_id in self._documents:
            return message_id
        keys = self._message_index.get(message_id, set())
        if not keys:
            raise KeyError(f"unknown message_id: {message_id}")
        if len(keys) > 1:
            raise ValueError(f"ambiguous message_id across mailboxes: {message_id}")
        return next(iter(keys))

    def _new_document(
        self, email: EmailMessage, run_id: str, idempotency_key: str
    ) -> dict[str, Any]:
        """Create the initial secured raw-email and workflow document."""

        now = utc_now_iso()
        return {
            "idempotency_key": idempotency_key,
            "email": email.to_dict(),
            "classification": None,
            "outcome": None,
            "workflow": {
                "status": ProcessingStatus.RAW_SAVED.value,
                "run_id": run_id,
                "attempt": 1,
                "decision": None,
                "decision_path": (),
                "confidence_scores": {},
                "risk_flags": (),
                "target_queue": None,
                "summary": "",
                "downstream_id": "",
                "last_stage": "raw_saved",
                "last_error_code": None,
                "lease_owner": run_id,
                "lease_expires_at": self._lease_expiration(),
            },
            "audit": {
                "status": AuditStatus.NOT_STARTED.value,
                "last_error_code": None,
                "appended_at": None,
            },
            "created_at": now,
            "updated_at": now,
        }

    @staticmethod
    def _claim_from(
        document: dict[str, Any], action: ClaimAction, idempotency_key: str
    ) -> EmailClaim:
        """Build a detached immutable claim from one stored document."""

        workflow = document["workflow"]
        return EmailClaim(
            action=action,
            idempotency_key=idempotency_key,
            message_id=str(document["email"]["message_id"]),
            status=ProcessingStatus(workflow["status"]),
            run_id=str(workflow["run_id"]),
        )

    def claim_email(self, email: EmailMessage, run_id: str) -> EmailClaim:
        """Atomically claim new work without duplicating completed outcomes."""

        if not run_id:
            raise ValueError("run_id must not be empty")
        idempotency_key = self._idempotency_key(email)
        with self._lock:
            document = self._documents.get(idempotency_key)
            if document is None:
                document = self._new_document(email, run_id, idempotency_key)
                self._documents[idempotency_key] = document
                self._message_index.setdefault(email.message_id, set()).add(idempotency_key)
                return self._claim_from(document, ClaimAction.NEW, idempotency_key)

            workflow = document["workflow"]
            status = ProcessingStatus(workflow["status"])
            audit_status = AuditStatus(document["audit"]["status"])
            if audit_status in {AuditStatus.PENDING, AuditStatus.FAILED} and document["outcome"]:
                return self._claim_from(document, ClaimAction.AUDIT_ONLY, idempotency_key)
            if status in self._TERMINAL_STATUSES and (
                audit_status == AuditStatus.APPENDED or status == ProcessingStatus.QUARANTINED
            ):
                return self._claim_from(document, ClaimAction.SKIP, idempotency_key)

            lease_owner = workflow.get("lease_owner")
            lease_expires = self._parse_timestamp(workflow.get("lease_expires_at"))
            if (
                lease_owner
                and lease_owner != run_id
                and lease_expires is not None
                and lease_expires > self._now()
                and status in {ProcessingStatus.RAW_SAVED, ProcessingStatus.PROCESSING}
            ):
                return self._claim_from(document, ClaimAction.IN_PROGRESS, idempotency_key)

            workflow.update(
                {
                    "run_id": run_id,
                    "attempt": int(workflow.get("attempt", 0)) + 1,
                    "lease_owner": run_id,
                    "lease_expires_at": self._lease_expiration(),
                    "last_error_code": None,
                }
            )
            document["updated_at"] = utc_now_iso()
            return self._claim_from(document, ClaimAction.RESUME, idempotency_key)

    def save_email(self, email: EmailMessage) -> None:
        """Insert a legacy raw email while retaining duplicate rejection."""

        idempotency_key = self._idempotency_key(email)
        with self._lock:
            if idempotency_key in self._documents:
                raise ValueError(f"duplicate message_id: {email.message_id}")
            self._documents[idempotency_key] = self._new_document(
                email, "legacy-save", idempotency_key
            )
            self._message_index.setdefault(email.message_id, set()).add(idempotency_key)

    def save_classification(self, message_id: str, prediction: Prediction) -> None:
        """Update only the CTD section of an existing stored email."""

        with self._lock:
            key = self._resolve_key(message_id)
            self._documents[key]["classification"] = prediction.to_dict()
            self._documents[key]["classified_at"] = utc_now_iso()
            self._documents[key]["updated_at"] = utc_now_iso()

    def mark_processing(self, message_id: str, run_id: str) -> None:
        """Move a claimed raw record into active LangGraph processing."""

        with self._lock:
            key = self._resolve_key(message_id)
            document = self._documents[key]
            workflow = document["workflow"]
            if workflow["run_id"] != run_id:
                raise ValueError("run_id does not own the processing lease")
            workflow.update(
                {
                    "status": ProcessingStatus.PROCESSING.value,
                    "last_stage": "processing",
                    "lease_owner": run_id,
                    "lease_expires_at": self._lease_expiration(),
                }
            )
            document["updated_at"] = utc_now_iso()

    def save_outcome(self, result: ProcessResult) -> None:
        """Persist a safe route or escalation before appending its audit event."""

        with self._lock:
            key = self._resolve_key(result.storage_key or result.message_id)
            document = self._documents[key]
            workflow = document["workflow"]
            if workflow["run_id"] != result.run_id:
                raise ValueError("run_id does not own the processing lease")
            if result.decision == "auto_route":
                status = ProcessingStatus.AUTO_ROUTED
            elif result.decision == "human_review":
                status = ProcessingStatus.HUMAN_REVIEW_PENDING
            else:
                raise ValueError(f"unsupported workflow decision: {result.decision}")

            prediction = result.prediction
            document["classification"] = prediction.to_dict() if prediction is not None else None
            document["outcome"] = result.to_dict()
            workflow.update(
                {
                    "status": status.value,
                    "decision": result.decision,
                    "decision_path": tuple(result.decision_path),
                    "confidence_scores": dict(result.confidence_scores),
                    "risk_flags": tuple(result.risk_flags),
                    "target_queue": result.target_queue,
                    "summary": result.summary,
                    "downstream_id": getattr(result, "downstream_id", ""),
                    "last_stage": "outcome_saved",
                    "last_error_code": None,
                    "lease_owner": None,
                    "lease_expires_at": None,
                }
            )
            document["audit"] = {
                "status": AuditStatus.PENDING.value,
                "last_error_code": None,
                "appended_at": None,
            }
            document["updated_at"] = utc_now_iso()

    def mark_audited(self, message_id: str, run_id: str) -> None:
        """Complete audit persistence without replaying prior routing work."""

        with self._lock:
            key = self._resolve_key(message_id)
            document = self._documents[key]
            workflow = document["workflow"]
            if workflow["run_id"] != run_id:
                raise ValueError("run_id does not match the stored outcome")
            document["audit"].update(
                {
                    "status": AuditStatus.APPENDED.value,
                    "last_error_code": None,
                    "appended_at": utc_now_iso(),
                }
            )
            if ProcessingStatus(workflow["status"]) == ProcessingStatus.AUTO_ROUTED:
                workflow["status"] = ProcessingStatus.COMPLETED.value
            workflow["last_stage"] = "audit_appended"
            document["updated_at"] = utc_now_iso()

    def mark_audit_failed(self, message_id: str, run_id: str, error_code: str) -> None:
        """Retain a finalized outcome for audit-only recovery."""

        with self._lock:
            key = self._resolve_key(message_id)
            document = self._documents[key]
            if document["workflow"]["run_id"] != run_id:
                raise ValueError("run_id does not match the stored outcome")
            document["audit"].update(
                {"status": AuditStatus.FAILED.value, "last_error_code": str(error_code)}
            )
            document["workflow"]["last_stage"] = "audit_pending"
            document["updated_at"] = utc_now_iso()

    def mark_retry_pending(
        self, message_id: str, run_id: str, stage: str, error_code: str
    ) -> None:
        """Persist a retry checkpoint and release the current processing lease."""

        with self._lock:
            key = self._resolve_key(message_id)
            document = self._documents[key]
            workflow = document["workflow"]
            if workflow["run_id"] != run_id:
                raise ValueError("run_id does not own the processing lease")
            workflow.update(
                {
                    "status": ProcessingStatus.RETRY_PENDING.value,
                    "last_stage": str(stage),
                    "last_error_code": str(error_code),
                    "lease_owner": None,
                    "lease_expires_at": None,
                }
            )
            document["updated_at"] = utc_now_iso()

    def is_completed(self, message_id: str) -> bool:
        """Return whether polling should skip classification and routing."""

        with self._lock:
            try:
                key = self._resolve_key(message_id)
            except KeyError:
                return False
            document = self._documents[key]
            status = ProcessingStatus(document["workflow"]["status"])
            audit_status = AuditStatus(document["audit"]["status"])
            return status in self._TERMINAL_STATUSES and (
                audit_status == AuditStatus.APPENDED or status == ProcessingStatus.QUARANTINED
            )

    def load_outcome(self, message_id: str) -> ProcessResult | None:
        """Rehydrate the safe saved result needed for audit-only recovery."""

        with self._lock:
            try:
                key = self._resolve_key(message_id)
            except KeyError:
                return None
            payload = deepcopy(self._documents[key].get("outcome"))
        if payload is None:
            return None

        prediction_value = payload.get("prediction")
        prediction = None
        if prediction_value is not None:
            prediction = Prediction(
                label=CTDLabel(**prediction_value["label"]),
                confidence=float(prediction_value["confidence"]),
                source=str(prediction_value["source"]),
                rationale=str(prediction_value["rationale"]),
            )
        values: dict[str, Any] = {
            "run_id": str(payload["run_id"]),
            "message_id": str(payload["message_id"]),
            "decision": str(payload["decision"]),
            "prediction": prediction,
            "target_queue": str(payload["target_queue"]),
            "summary": str(payload["summary"]),
            "decision_path": tuple(payload["decision_path"]),
            "risk_flags": tuple(payload.get("risk_flags", ())),
            "confidence_scores": dict(payload.get("confidence_scores", {})),
        }
        fields = getattr(ProcessResult, "__dataclass_fields__", {})
        if "downstream_id" in fields:
            values["downstream_id"] = str(payload.get("downstream_id", ""))
        if "was_skipped" in fields:
            values["was_skipped"] = bool(payload.get("was_skipped", False))
        if "storage_key" in fields:
            values["storage_key"] = str(payload.get("storage_key", key))
        return ProcessResult(**values)

    def quarantine_ingestion_failure(self, source_key: str, error_code: str) -> None:
        """Store non-sensitive poison-record metadata idempotently."""

        if not source_key.strip() or not error_code.strip():
            raise ValueError("source_key and error_code must not be empty")
        with self._lock:
            existing = self._quarantined_failures.get(source_key)
            now = utc_now_iso()
            if existing is None:
                self._quarantined_failures[source_key] = {
                    "source_key": source_key,
                    "error_code": error_code,
                    "status": ProcessingStatus.QUARANTINED.value,
                    "attempt": 1,
                    "created_at": now,
                    "updated_at": now,
                }
                return
            existing.update(
                {
                    "error_code": error_code,
                    "attempt": int(existing["attempt"]) + 1,
                    "updated_at": now,
                }
            )

    def quarantined_failures(self) -> tuple[dict[str, Any], ...]:
        """Return detached poison-record metadata for operational review."""

        with self._lock:
            return tuple(
                deepcopy(self._quarantined_failures[key])
                for key in sorted(self._quarantined_failures)
            )

    def get(self, message_id: str) -> dict[str, Any] | None:
        """Return a defensive copy of one document when it exists."""

        with self._lock:
            try:
                key = self._resolve_key(message_id)
            except KeyError:
                return None
            return deepcopy(self._documents[key])

    def count(self) -> int:
        """Return the number of stored email documents."""

        with self._lock:
            return len(self._documents)


@dataclass(frozen=True, slots=True)
class PolicyDocument:
    """Store a routing-policy chunk and its safe retrieval metadata."""

    document_id: str
    text: str
    metadata: dict[str, str]


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Pair a retrieved policy document with its cosine similarity score."""

    document: PolicyDocument
    score: float


class MockVectorStore:
    """Provide deterministic bag-of-words embeddings and semantic retrieval."""

    _TOKEN_RE = re.compile(r"[a-z][a-z0-9-]+", re.IGNORECASE)

    def __init__(self) -> None:
        """Create an empty policy-document index."""

        self._documents: dict[str, PolicyDocument] = {}
        self._vectors: dict[str, Counter[str]] = {}

    @classmethod
    def _embed(cls, text: str) -> Counter[str]:
        """Map text into a normalized-token frequency vector."""

        return Counter(token.lower() for token in cls._TOKEN_RE.findall(text))

    @staticmethod
    def _cosine(left: Counter[str], right: Counter[str]) -> float:
        """Compute cosine similarity between two sparse frequency vectors."""

        if not left or not right:
            return 0.0
        dot = sum(value * right.get(token, 0) for token, value in left.items())
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        return dot / (left_norm * right_norm)

    def add(self, document: PolicyDocument) -> None:
        """Insert or replace a policy chunk and its deterministic embedding."""

        safe_document = deepcopy(document)
        self._documents[document.document_id] = safe_document
        self._vectors[document.document_id] = self._embed(document.text)

    def search(self, query: str, limit: int = 3) -> tuple[SearchResult, ...]:
        """Return the most similar policy chunks with deterministic tie-breaking."""

        if limit < 1:
            raise ValueError("limit must be positive")
        query_vector = self._embed(query)
        ranked = [
            SearchResult(deepcopy(document), self._cosine(query_vector, self._vectors[document_id]))
            for document_id, document in self._documents.items()
        ]
        ranked.sort(key=lambda item: (-item.score, item.document.document_id))
        return tuple(ranked[:limit])

