"""Tamper-evident append-only audit logging for every routing decision."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from .models import ProcessResult, utc_now_iso


@dataclass(frozen=True, slots=True)
class AuditEntry:
    """Represent one immutable, hash-linked routing decision record."""

    schema_version: str
    sequence: int
    event_id: str
    timestamp: str
    run_id: str
    message_id: str
    event_type: str
    decision: str
    ctd: dict[str, str]
    confidence_scores: dict[str, float]
    final_confidence: float
    model_source: str
    target_queue: str
    risk_flags: tuple[str, ...]
    decision_path: tuple[str, ...]
    previous_hash: str
    entry_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible copy of this audit record."""

        return asdict(self)


class ImmutableAuditLog:
    """Expose append and verification operations over a SHA-256 hash chain.

    The POC cannot enforce storage-level WORM semantics, so immutability is
    represented by a deliberately append-only API plus tamper detection across
    every serialized JSONL record. Production should replace this adapter with
    a retention-locked object store or equivalent immutable ledger.
    """

    GENESIS_HASH = "0" * 64

    def __init__(self, path: str | Path | None = None) -> None:
        """Load and validate an existing JSONL chain, or start an empty ledger."""

        self.path = Path(path) if path is not None else None
        self._entries: list[AuditEntry] = []
        self._lock = Lock()
        if self.path is not None and self.path.exists():
            self._load()
            if not self.verify_integrity():
                raise ValueError(f"audit chain integrity check failed: {self.path}")

    @staticmethod
    def _calculate_hash(payload: dict[str, Any]) -> str:
        """Hash canonical compact JSON so verification is platform-independent."""

        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _load(self) -> None:
        """Deserialize existing audit entries without allowing partial records."""

        assert self.path is not None
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                value["risk_flags"] = tuple(value["risk_flags"])
                value["decision_path"] = tuple(value["decision_path"])
                self._entries.append(AuditEntry(**value))
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid audit entry on line {line_number}") from error

    def record_decision(self, result: ProcessResult) -> AuditEntry:
        """Append a structured routing result and return its immutable record."""

        with self._lock:
            previous_hash = self._entries[-1].entry_hash if self._entries else self.GENESIS_HASH
            payload: dict[str, Any] = {
                "schema_version": "1.0",
                "sequence": len(self._entries),
                "event_id": str(uuid4()),
                "timestamp": utc_now_iso(),
                "run_id": result.run_id,
                "message_id": result.message_id,
                "event_type": "routing_decision",
                "decision": result.decision,
                "ctd": result.prediction.label.to_dict(),
                "confidence_scores": dict(result.confidence_scores),
                "final_confidence": result.prediction.confidence,
                "model_source": result.prediction.source,
                "target_queue": result.target_queue,
                "risk_flags": tuple(result.risk_flags),
                "decision_path": tuple(result.decision_path),
                "previous_hash": previous_hash,
            }
            entry = AuditEntry(**payload, entry_hash=self._calculate_hash(payload))
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(entry.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")
                    stream.flush()
            self._entries.append(entry)
            return entry

    def entries(self, run_id: str | None = None) -> tuple[AuditEntry, ...]:
        """Return an immutable snapshot, optionally filtered to one workflow run."""

        if run_id is None:
            return tuple(self._entries)
        return tuple(entry for entry in self._entries if entry.run_id == run_id)

    def verify_integrity(self) -> bool:
        """Validate sequence numbers, previous hashes, and all entry digests."""

        expected_previous = self.GENESIS_HASH
        for expected_sequence, entry in enumerate(self._entries):
            payload = entry.to_dict()
            entry_hash = payload.pop("entry_hash")
            if entry.sequence != expected_sequence or entry.previous_hash != expected_previous:
                return False
            if self._calculate_hash(payload) != entry_hash:
                return False
            expected_previous = entry_hash
        return True

