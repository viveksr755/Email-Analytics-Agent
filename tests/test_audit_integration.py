"""Phase-five audit and end-to-end integration tests."""

import json
from pathlib import Path

from email_analytics_agent.audit import ImmutableAuditLog
from email_analytics_agent.cli import load_messages
from email_analytics_agent.factory import build_default_system
from email_analytics_agent.ingestion import MockEventStream, MockIngestionPipeline, MockMailConnector
from email_analytics_agent.models import EmailMessage


DATASET = Path(__file__).parents[1] / "data" / "mock_emails.json"


def _email(message_id: str, subject: str, body: str) -> EmailMessage:
    """Build an isolated email used to validate audit decision paths."""

    return EmailMessage(message_id, "customer@example.test", subject, body, "2026-01-01T00:00:00Z")


def test_audit_captures_exact_paths_and_all_confidence_scores(tmp_path: Path) -> None:
    """Verify one hash-linked final decision record exists for every run."""

    audit_path = tmp_path / "audit.jsonl"
    system = build_default_system(audit_path)
    direct = system.manager.process_email(
        _email("AUDIT-HIGH", "Home loan mortgage", "Loan EMI disbursement for 991100221234.")
    )
    fallback = system.manager.process_email(
        _email("AUDIT-LOW", "Position extension", "Please arrange a rollover for this position.")
    )

    entries = system.audit_log.entries()
    assert len(entries) == 2
    assert entries[0].run_id == direct.run_id
    assert entries[0].storage_key == direct.storage_key
    assert entries[0].model_source == "sklearn-word-char-logreg-v2"
    assert entries[0].decision_path == direct.decision_path
    assert set(entries[0].confidence_scores) == {"ml"}
    assert entries[1].run_id == fallback.run_id
    assert entries[1].decision_path == fallback.decision_path
    assert set(entries[1].confidence_scores) == {"ml", "rag"}
    assert entries[1].previous_hash == entries[0].entry_hash
    assert system.audit_log.verify_integrity()

    serialized = audit_path.read_text(encoding="utf-8")
    assert "991100221234" not in serialized
    assert len(serialized.splitlines()) == 2
    assert ImmutableAuditLog(audit_path).verify_integrity()


def test_full_twenty_email_pipeline_routes_and_audits_every_message(tmp_path: Path) -> None:
    """Exercise ingestion, storage, preprocessing, graph, queues, and audit end to end."""

    messages = load_messages(DATASET)
    stream = MockEventStream()
    assert MockIngestionPipeline(MockMailConnector(messages), stream).ingest() == 20
    system = build_default_system(tmp_path / "full-audit.jsonl")
    results = []
    while event := stream.consume("raw-emails"):
        results.append(system.manager.process_email(event.email))

    assert len(results) == 20
    assert system.document_store.count() == 20
    assert len(system.audit_log.entries()) == 20
    assert system.audit_log.verify_integrity()
    assert all(result.decision == "auto_route" for result in results)
    assert system.llm_client.calls == []
    assert {
        queue: system.queues.count(queue)
        for queue in (
            "loans-operations",
            "credit-operations",
            "complaints-resolution",
            "treasury-operations",
        )
    } == {
        "loans-operations": 5,
        "credit-operations": 5,
        "complaints-resolution": 5,
        "treasury-operations": 5,
    }

    raw_audit = (tmp_path / "full-audit.jsonl").read_text(encoding="utf-8")
    assert "CUST000001" not in raw_audit
    assert "991100220001" not in raw_audit
    assert len([json.loads(line) for line in raw_audit.splitlines()]) == 20

