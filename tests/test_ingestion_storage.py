"""Phase-one tests for mock ingestion and persistence boundaries."""

import json
from pathlib import Path

from email_analytics_agent.ingestion import MockEventStream, MockIngestionPipeline, MockMailConnector
from email_analytics_agent.models import EmailMessage
from email_analytics_agent.storage import MockMongoStore, MockVectorStore, PolicyDocument


DATASET = Path(__file__).parents[1] / "data" / "mock_emails.json"


def _load_messages() -> list[EmailMessage]:
    """Load checked-in synthetic messages for integration-style tests."""

    rows = json.loads(DATASET.read_text(encoding="utf-8"))
    return [EmailMessage.from_dict(row) for row in rows]


def test_dataset_has_twenty_balanced_messages_with_pii() -> None:
    """Ensure mock data covers all requested departments and contains PII."""

    rows = json.loads(DATASET.read_text(encoding="utf-8"))
    assert len(rows) == 20
    assert {row["expected_department"] for row in rows} == {
        "Loans",
        "Credit",
        "Complaints",
        "Treasury",
    }
    assert all("CUST" in row["body"] or "@example.test" in row["body"] or "PAN" in row["body"] for row in rows)


def test_ingestion_publishes_ordered_replayable_events() -> None:
    """Verify mocked O365/IMAP messages behave like an event stream."""

    messages = _load_messages()
    stream = MockEventStream()
    pipeline = MockIngestionPipeline(MockMailConnector(messages), stream)

    assert pipeline.ingest() == 20
    first = stream.consume("raw-emails")
    assert first is not None
    assert first.offset == 0
    assert first.email.message_id == "MSG-001"
    assert len(stream.replay("raw-emails")) == 20


def test_mock_storage_detaches_documents_and_vector_results() -> None:
    """Verify document isolation and semantically relevant policy retrieval."""

    message = _load_messages()[0]
    mongo = MockMongoStore()
    mongo.save_email(message)
    document = mongo.get(message.message_id)
    assert document is not None
    document["email"]["subject"] = "mutated"
    assert mongo.get(message.message_id)["email"]["subject"] == message.subject

    vectors = MockVectorStore()
    vectors.add(PolicyDocument("loan", "Mortgage and EMI requests route to Loans.", {"department": "Loans"}))
    vectors.add(PolicyDocument("fx", "Foreign exchange settlement routes to Treasury.", {"department": "Treasury"}))
    result = vectors.search("mortgage EMI assistance", limit=1)
    assert result[0].document.document_id == "loan"

