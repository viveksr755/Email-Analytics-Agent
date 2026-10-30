"""Workflow-state and idempotency tests for the MongoDB-shaped adapter."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from email_analytics_agent.models import CTDLabel, EmailMessage, Prediction, ProcessResult
from email_analytics_agent.storage import (
    AuditStatus,
    ClaimAction,
    MockMongoStore,
    ProcessingStatus,
)


def _email(message_id: str = "GRAPH-1") -> EmailMessage:
    """Build one Graph-shaped email with a composite mailbox identity."""

    return EmailMessage(
        message_id=message_id,
        sender="customer@example.test",
        subject="Mortgage repayment",
        body="Please review account 991100221234.",
        received_at="2026-01-01T00:00:00Z",
        metadata={
            "tenant_id": "tenant-1",
            "mailbox_id": "mailbox-1",
            "immutable_message_id": message_id,
        },
    )


def _result(run_id: str, decision: str = "auto_route") -> ProcessResult:
    """Build one PII-safe persisted routing outcome."""

    prediction = Prediction(
        CTDLabel("Lending", "Loan servicing", "Loans"),
        0.93,
        "model-v1",
        "Calibrated prediction.",
    )
    return ProcessResult(
        run_id=run_id,
        message_id="GRAPH-1",
        decision=decision,
        prediction=prediction,
        target_queue="loans-operations" if decision == "auto_route" else "rm-review",
        summary="Loans case - account [ACCOUNT_NUMBER].",
        decision_path=("preprocess", "ml_classify", "finalize"),
        downstream_id="queue-item-1",
        risk_flags=() if decision == "auto_route" else ("suspected_fraud",),
        confidence_scores={"ml": 0.93},
    )


def test_atomic_claim_saves_raw_email_and_blocks_an_active_second_worker() -> None:
    """Ensure only one worker owns a new message during its processing lease."""

    store = MockMongoStore()
    first = store.claim_email(_email(), "run-1")
    second = store.claim_email(_email(), "run-2")

    assert first.action == ClaimAction.NEW
    assert first.idempotency_key == "tenant-1:mailbox-1:GRAPH-1"
    assert second.action == ClaimAction.IN_PROGRESS
    assert second.run_id == "run-1"
    assert store.count() == 1
    document = store.get("GRAPH-1")
    assert document is not None
    assert document["workflow"]["status"] == ProcessingStatus.RAW_SAVED.value
    assert document["email"]["body"] == "Please review account 991100221234."


def test_retryable_claim_resumes_without_overwriting_the_original() -> None:
    """Verify a released failure checkpoint resumes the same durable record."""

    store = MockMongoStore()
    store.claim_email(_email(), "run-1")
    store.mark_processing("GRAPH-1", "run-1")
    store.mark_retry_pending("GRAPH-1", "run-1", "ml_classify", "model_unavailable")

    resumed = store.claim_email(_email(), "run-2")
    document = store.get("GRAPH-1")

    assert resumed.action == ClaimAction.RESUME
    assert resumed.run_id == "run-2"
    assert document is not None
    assert document["workflow"]["attempt"] == 2
    assert document["email"]["body"] == "Please review account 991100221234."


def test_auto_route_is_completed_only_after_audit_append() -> None:
    """Persist routing before audit and skip rediscovery only after audit succeeds."""

    store = MockMongoStore()
    store.claim_email(_email(), "run-1")
    store.mark_processing("GRAPH-1", "run-1")
    expected = _result("run-1")
    store.save_outcome(expected)

    pending = store.get("GRAPH-1")
    assert pending is not None
    assert pending["workflow"]["status"] == ProcessingStatus.AUTO_ROUTED.value
    assert pending["audit"]["status"] == AuditStatus.PENDING.value
    assert store.claim_email(_email(), "run-2").action == ClaimAction.AUDIT_ONLY
    assert store.load_outcome("GRAPH-1") == expected

    store.mark_audited("GRAPH-1", "run-1")
    completed = store.get("GRAPH-1")
    assert completed is not None
    assert completed["workflow"]["status"] == ProcessingStatus.COMPLETED.value
    assert completed["audit"]["status"] == AuditStatus.APPENDED.value
    assert store.is_completed("GRAPH-1")
    assert store.claim_email(_email(), "run-3").action == ClaimAction.SKIP


def test_human_review_and_audit_failure_resume_only_the_audit() -> None:
    """Keep an RM case pending while protecting it from model or queue replay."""

    store = MockMongoStore()
    store.claim_email(_email(), "run-risk")
    store.mark_processing("GRAPH-1", "run-risk")
    expected = _result("run-risk", decision="human_review")
    store.save_outcome(expected)
    store.mark_audit_failed("GRAPH-1", "run-risk", "audit_unavailable")

    claim = store.claim_email(_email(), "another-run")
    assert claim.action == ClaimAction.AUDIT_ONLY
    assert claim.run_id == "run-risk"
    assert store.load_outcome("GRAPH-1") == expected

    store.mark_audited("GRAPH-1", "run-risk")
    document = store.get("GRAPH-1")
    assert document is not None
    assert document["workflow"]["status"] == ProcessingStatus.HUMAN_REVIEW_PENDING.value
    assert document["audit"]["status"] == AuditStatus.APPENDED.value
    assert store.is_completed("GRAPH-1")


def test_concurrent_claims_create_exactly_one_record() -> None:
    """Exercise the in-memory equivalent of a MongoDB atomic unique upsert."""

    store = MockMongoStore()
    with ThreadPoolExecutor(max_workers=8) as executor:
        claims = tuple(
            executor.map(
                lambda number: store.claim_email(_email(), f"run-{number}"),
                range(8),
            )
        )

    assert sum(claim.action == ClaimAction.NEW for claim in claims) == 1
    assert sum(claim.action == ClaimAction.IN_PROGRESS for claim in claims) == 7
    assert store.count() == 1


def test_legacy_save_email_still_rejects_duplicates_and_returns_copies() -> None:
    """Preserve the original POC storage contract for existing callers."""

    store = MockMongoStore()
    email = EmailMessage(
        "LEGACY-1",
        "customer@example.test",
        "Subject",
        "Body",
        "2026-01-01T00:00:00Z",
    )
    store.save_email(email)
    with pytest.raises(ValueError, match="duplicate message_id"):
        store.save_email(email)

    document = store.get("LEGACY-1")
    assert document is not None
    document["email"]["subject"] = "mutated"
    assert store.get("LEGACY-1")["email"]["subject"] == "Subject"


def test_ingestion_quarantine_is_idempotent_and_contains_no_email_body() -> None:
    """Persist safe poison-record metadata outside the raw-email collection."""

    store = MockMongoStore()
    source_key = "tenant-1:mailbox-1:BROKEN-1"

    store.quarantine_ingestion_failure(source_key, "missing_received_datetime")
    store.quarantine_ingestion_failure(source_key, "missing_received_datetime")

    failures = store.quarantined_failures()
    assert len(failures) == 1
    assert failures[0]["attempt"] == 2
    assert failures[0]["status"] == ProcessingStatus.QUARANTINED.value
    assert "body" not in failures[0]
