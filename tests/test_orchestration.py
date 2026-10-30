"""Phase-four tests for LangGraph confidence edges and routing outcomes."""

from dataclasses import replace

import pytest

from email_analytics_agent.audit import ImmutableAuditLog
from email_analytics_agent.classification import CTDClassifier, SklearnCTDClassifier
from email_analytics_agent.config import AgentConfig
from email_analytics_agent.guardrails import ComplianceGuardrailAgent
from email_analytics_agent.models import EmailMessage, Prediction
from email_analytics_agent.orchestration import ManagerAgent
from email_analytics_agent.preprocessing import EmailPreprocessor, PIIRedactor
from email_analytics_agent.rag import LocalLLMClient, RAGClassificationAgent, seed_routing_policies
from email_analytics_agent.routing import (
    MockQueueRouter,
    QueueDestinationConflict,
    SummarizerRoutingAgent,
)
from email_analytics_agent.storage import ClaimAction, MockMongoStore, MockVectorStore


def _email(message_id: str, subject: str, body: str) -> EmailMessage:
    """Create a test email with a stable synthetic receipt timestamp."""

    return EmailMessage(message_id, "customer@example.test", subject, body, "2026-01-01T00:00:00Z")


def _build_manager(
    ml_classifier: CTDClassifier | None = None,
    audit_log: ImmutableAuditLog | None = None,
) -> tuple[ManagerAgent, LocalLLMClient, MockQueueRouter, MockMongoStore]:
    """Construct isolated orchestration dependencies for one test."""

    config = AgentConfig()
    redactor = PIIRedactor()
    guardrail = ComplianceGuardrailAgent(redactor)
    vectors = MockVectorStore()
    seed_routing_policies(vectors)
    llm = LocalLLMClient()
    rag = RAGClassificationAgent(vectors, llm, guardrail, config)
    queues = MockQueueRouter()
    mongo = MockMongoStore()
    manager = ManagerAgent(
        EmailPreprocessor(redactor),
        ml_classifier or SklearnCTDClassifier(),
        rag,
        guardrail,
        SummarizerRoutingAgent(config),
        queues,
        mongo,
        config,
        audit_log,
    )
    return manager, llm, queues, mongo


class RecordingMLClassifier:
    """Record model-facing text while delegating to the genuine classifier."""

    def __init__(self) -> None:
        """Create a genuine classifier and an empty input ledger."""

        self.delegate = SklearnCTDClassifier()
        self.inputs: list[str] = []

    def predict(self, redacted_text: str) -> Prediction:
        """Capture the exact LangGraph payload before genuine inference."""

        self.inputs.append(redacted_text)
        return self.delegate.predict(redacted_text)


class FailingOnceAuditLog(ImmutableAuditLog):
    """Fail one append so an audit-only retry can be exercised."""

    def __init__(self) -> None:
        """Create an empty ledger whose first decision append fails."""

        super().__init__()
        self.fail_next = True

    def record_decision(self, result):
        """Raise once, then delegate to the real idempotent audit log."""

        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("synthetic audit outage")
        return super().record_decision(result)


def test_high_confidence_email_bypasses_llm_and_auto_routes() -> None:
    """Verify the high-confidence conditional edge never invokes RAG/LLM."""

    manager, llm, queues, mongo = _build_manager()
    result = manager.process_email(
        _email(
            "HIGH-1",
            "Home loan mortgage application",
            "Please review my loan EMI disbursement for account 991100221234.",
        )
    )

    assert result.decision == "auto_route"
    assert result.target_queue == "loans-operations"
    assert result.prediction.label.department == "Loans"
    assert result.prediction.source == "sklearn-word-char-logreg-v2"
    assert result.decision_path == ("preprocess", "ml_classify", "summarize", "auto_route", "finalize")
    assert llm.calls == []
    assert queues.count("loans-operations") == 1
    assert result.downstream_id == queues.receipt("HIGH-1", "loans-operations")
    assert "991100221234" not in queues.items("loans-operations")[0].summary
    assert mongo.get("HIGH-1")["classification"]["label"]["department"] == "Loans"


def test_langgraph_passes_only_redacted_text_to_the_trained_model() -> None:
    """Prove preprocessing protects the genuine ML boundary as well as the LLM."""

    classifier = RecordingMLClassifier()
    manager, _, _, _ = _build_manager(classifier)
    manager.process_email(
        _email(
            "ML-PII-1",
            "Mortgage for account 991100221234",
            "My PAN ABCDE1234F concerns this home loan EMI application.",
        )
    )

    assert len(classifier.inputs) == 1
    assert "991100221234" not in classifier.inputs[0]
    assert "ABCDE1234F" not in classifier.inputs[0]
    assert "[ACCOUNT_NUMBER]" in classifier.inputs[0]
    assert "[PAN]" in classifier.inputs[0]


def test_low_confidence_email_triggers_rag_and_auto_routes() -> None:
    """Verify semantic fallback runs only after the trained ML score is low."""

    manager, llm, queues, _ = _build_manager()
    result = manager.process_email(
        _email(
            "LOW-1",
            "Position extension",
            "Please arrange a rollover for this position and account 991100221234.",
        )
    )

    assert result.decision == "auto_route"
    assert result.target_queue == "treasury-operations"
    assert result.prediction.source == "local-rag-llm-v1"
    assert result.decision_path == (
        "preprocess",
        "ml_classify",
        "rag_classify",
        "summarize",
        "auto_route",
        "finalize",
    )
    assert len(llm.calls) == 1
    assert "991100221234" not in llm.calls[0].email_text
    assert queues.count("treasury-operations") == 1


def test_risky_email_routes_to_rm_review_without_llm() -> None:
    """Verify a high-risk email bypasses both ML and LLM entirely."""

    classifier = RecordingMLClassifier()
    manager, llm, queues, _ = _build_manager(classifier)
    result = manager.process_email(
        _email(
            "RISK-1",
            "Unauthorized fraud on home loan",
            "This mortgage EMI appears to be an unauthorized fraud on account 991100221234.",
        )
    )

    assert result.decision == "human_review"
    assert result.target_queue == "rm-review"
    assert result.prediction.label.department == "RM Review"
    assert result.prediction.source == "compliance-guardrail-v1"
    assert "suspected_fraud" in result.risk_flags
    assert result.confidence_scores == {}
    assert result.decision_path == ("preprocess", "rm_review", "finalize")
    assert classifier.inputs == []
    assert llm.calls == []
    assert queues.count("rm-review") == 1


def test_ambiguous_rag_result_routes_to_rm_review() -> None:
    """Verify tied fallback evidence cannot be auto-routed."""

    manager, llm, queues, _ = _build_manager()
    result = manager.process_email(
        _email("AMB-1", "Need advice", "I am unhappy and need a rollover.")
    )

    assert result.decision == "human_review"
    assert result.prediction.label.department == "RM Review"
    assert result.decision_path == (
        "preprocess",
        "ml_classify",
        "rag_classify",
        "rm_review",
        "finalize",
    )
    assert len(llm.calls) == 1
    assert queues.count("rm-review") == 1


def test_completed_email_is_skipped_without_duplicate_queue_or_audit() -> None:
    """Verify MongoDB idempotency suppresses all repeated side effects."""

    manager, _, queues, mongo = _build_manager()
    email = _email(
        "DUP-1",
        "Home loan mortgage application",
        "Please review my loan EMI disbursement.",
    )

    first = manager.process_email(email)
    second = manager.process_email(email)

    assert not first.was_skipped
    assert second.was_skipped
    assert second.run_id == first.run_id
    assert second.downstream_id == first.downstream_id
    assert queues.count("loans-operations") == 1
    assert mongo.count() == 1
    assert len(manager.audit_log.entries()) == 1


def test_process_saved_email_accepts_an_ingestion_claim() -> None:
    """Verify Graph ingestion can persist raw mail before LangGraph starts."""

    manager, _, queues, mongo = _build_manager()
    email = _email(
        "CLAIM-1",
        "Home loan mortgage application",
        "Please review my loan EMI disbursement.",
    )
    claim = mongo.claim_email(email, "preclaimed-run")

    assert claim.action is ClaimAction.NEW
    result = manager.process_saved_email(email, claim.run_id)

    assert result.run_id == "preclaimed-run"
    assert result.decision == "auto_route"
    assert queues.count("loans-operations") == 1
    assert mongo.count() == 1


def test_audit_failure_retries_only_audit_and_not_models_or_queue() -> None:
    """Verify a persisted outcome resumes through the audit-only claim path."""

    audit = FailingOnceAuditLog()
    classifier = RecordingMLClassifier()
    manager, _, queues, mongo = _build_manager(classifier, audit)
    email = _email(
        "AUDIT-RETRY-1",
        "Home loan mortgage application",
        "Please review my loan EMI disbursement.",
    )

    with pytest.raises(RuntimeError, match="synthetic audit outage"):
        manager.process_email(email)

    assert mongo.load_outcome(email.message_id) is not None
    assert len(classifier.inputs) == 1
    assert queues.count("loans-operations") == 1

    repaired = manager.process_email(email)

    assert repaired.was_skipped
    assert len(classifier.inputs) == 1
    assert queues.count("loans-operations") == 1
    assert len(audit.entries()) == 1
    assert audit.verify_integrity()


def test_queue_and_audit_retries_are_idempotent_for_one_result() -> None:
    """Verify stable queue receipts and one audit record per workflow run."""

    manager, _, queues, _ = _build_manager()
    result = manager.process_email(
        _email("IDEMP-1", "Home loan mortgage", "Please provide my loan EMI schedule.")
    )
    item = queues.items(result.target_queue)[0]

    assert queues.enqueue(item) == result.downstream_id
    assert queues.count(result.target_queue) == 1
    first_entry = manager.audit_log.entries(result.run_id)[0]
    repeated_entry = manager.audit_log.record_decision(result)
    assert repeated_entry == first_entry
    assert len(manager.audit_log.entries(result.run_id)) == 1

    conflicting = replace(item, target_queue="credit-operations")
    with pytest.raises(QueueDestinationConflict):
        queues.enqueue(conflicting)
    assert queues.count() == 1


def test_same_outlook_id_in_two_mailboxes_uses_composite_identity() -> None:
    """Keep storage and queue side effects separate across mailbox scopes."""

    manager, _, queues, mongo = _build_manager()
    common = {
        "immutable_message_id": "SHARED-ID",
        "tenant_id": "tenant-1",
    }
    first = EmailMessage(
        "SHARED-ID",
        "customer@example.test",
        "Home loan mortgage",
        "Please provide my loan EMI schedule.",
        "2026-01-01T00:00:00Z",
        metadata={
            **common,
            "mailbox_id": "mailbox-1",
            "idempotency_key": "tenant-1:mailbox-1:SHARED-ID",
        },
    )
    second = EmailMessage(
        "SHARED-ID",
        "customer@example.test",
        "Home loan mortgage",
        "Please provide my loan EMI schedule.",
        "2026-01-01T00:00:01Z",
        metadata={
            **common,
            "mailbox_id": "mailbox-2",
            "idempotency_key": "tenant-1:mailbox-2:SHARED-ID",
        },
    )

    first_result = manager.process_email(first)
    second_result = manager.process_email(second)

    assert first_result.storage_key == "tenant-1:mailbox-1:SHARED-ID"
    assert second_result.storage_key == "tenant-1:mailbox-2:SHARED-ID"
    assert first_result.downstream_id != second_result.downstream_id
    assert mongo.count() == 2
    assert queues.count("loans-operations") == 2
    assert len(manager.audit_log.entries()) == 2
    assert {entry.storage_key for entry in manager.audit_log.entries()} == {
        "tenant-1:mailbox-1:SHARED-ID",
        "tenant-1:mailbox-2:SHARED-ID",
    }

