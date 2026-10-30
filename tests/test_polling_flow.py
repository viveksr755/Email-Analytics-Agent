"""Integration tests for the redesigned Graph polling coordinator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from email_analytics_agent.factory import build_default_system, build_graph_polling_system
from email_analytics_agent.graph_ingestion import (
    GraphClientConfig,
    GraphConversionFailure,
    GraphPollResult,
)
from email_analytics_agent.polling import GraphPollingCoordinator, InMemoryDeltaCursorStore


@dataclass
class StubGraphClient:
    """Return queued Graph batches and record supplied delta cursors."""

    config: GraphClientConfig
    batches: list[GraphPollResult]

    def __post_init__(self) -> None:
        """Initialize an inspectable cursor ledger."""

        self.cursors: list[str | None] = []

    def fetch_recent(self, delta_link: str | None = None) -> GraphPollResult:
        """Return the next deterministic batch."""

        self.cursors.append(delta_link)
        return self.batches.pop(0)


class StaticTokenProvider:
    """Supply a deterministic non-secret token to the full wiring test."""

    def get_access_token(self) -> str:
        """Return a fixed test token."""

        return "test-token"

    def invalidate(self) -> None:
        """Implement the token-provider contract without cached state."""


class SinglePageTransport:
    """Return one Graph delta page and capture its request headers."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        """Store the deterministic response and initialize the call ledger."""

        self.payload = payload
        self.calls: list[tuple[str, Mapping[str, str]]] = []

    def get_json(
        self,
        url: str,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        """Record one request and return the configured page."""

        assert timeout_seconds > 0
        self.calls.append((url, dict(headers)))
        return self.payload


def _graph_message(message_id: str):
    """Build one already-converted safe test message."""

    from email_analytics_agent.models import EmailMessage

    return EmailMessage(
        message_id=message_id,
        sender="customer@example.test",
        subject="Home loan mortgage application",
        body="Please provide my loan EMI schedule for account 991100221234.",
        received_at="2026-01-01T00:00:00Z",
        metadata={
            "provider": "microsoft-graph",
            "tenant_id": "tenant-1",
            "mailbox_id": "mailbox-1",
            "immutable_message_id": message_id,
            "idempotency_key": f"tenant-1:mailbox-1:{message_id}",
        },
    )


def test_polling_processes_new_email_commits_cursor_and_skips_replay() -> None:
    """Prove a repeated Graph page cannot create a second downstream case."""

    email = _graph_message("GRAPH-1")
    delta_one = "https://graph.microsoft.test/delta?$deltatoken=one"
    delta_two = "https://graph.microsoft.test/delta?$deltatoken=two"
    client = StubGraphClient(
        GraphClientConfig("mailbox-1", tenant_id="tenant-1"),
        [
            GraphPollResult((email,), delta_one, (), (), 1),
            GraphPollResult((email,), delta_two, (), (), 1),
        ],
    )
    system = build_default_system()
    cursors = InMemoryDeltaCursorStore()
    coordinator = GraphPollingCoordinator(client, system.manager, cursors)

    first = coordinator.poll_once()
    second = coordinator.poll_once()

    assert first.processed == 1
    assert first.skipped == 0
    assert first.cursor_committed
    assert second.processed == 0
    assert second.skipped == 1
    assert second.cursor_committed
    assert client.cursors == [None, delta_one]
    assert cursors.get("mailbox-1", "inbox") == delta_two
    assert system.document_store.count() == 1
    assert system.queues.count("loans-operations") == 1
    assert len(system.audit_log.entries()) == 1


def test_polling_does_not_advance_cursor_for_unidentifiable_conversion_failure() -> None:
    """Keep the old cursor when a poison record cannot be durably identified."""

    prior = "https://graph.microsoft.test/delta?$deltatoken=prior"
    proposed = "https://graph.microsoft.test/delta?$deltatoken=proposed"
    client = StubGraphClient(
        GraphClientConfig("mailbox-1", tenant_id="tenant-1"),
        [
            GraphPollResult(
                (),
                proposed,
                (),
                (GraphConversionFailure(None, "missing_message_id"),),
                1,
            )
        ],
    )
    system = build_default_system()
    cursors = InMemoryDeltaCursorStore()
    cursors.commit("mailbox-1", "inbox", prior)

    result = GraphPollingCoordinator(client, system.manager, cursors).poll_once()

    assert result.failed == 1
    assert not result.cursor_committed
    assert result.quarantined == 0
    assert result.failures[0].error_code == "missing_message_id"
    assert cursors.get("mailbox-1", "inbox") == prior


def test_polling_quarantines_identifiable_poison_record_before_cursor_commit() -> None:
    """Prevent one malformed Graph message from blocking every later poll."""

    proposed = "https://graph.microsoft.test/delta?$deltatoken=proposed"
    client = StubGraphClient(
        GraphClientConfig("mailbox-1", tenant_id="tenant-1"),
        [
            GraphPollResult(
                (),
                proposed,
                (),
                (GraphConversionFailure("BAD-1", "missing_received_datetime"),),
                1,
            )
        ],
    )
    system = build_default_system()
    cursors = InMemoryDeltaCursorStore()

    result = GraphPollingCoordinator(client, system.manager, cursors).poll_once()

    assert result.failed == 1
    assert result.quarantined == 1
    assert result.cursor_committed
    assert cursors.get("mailbox-1", "inbox") == proposed
    quarantined = system.document_store.quarantined_failures()
    assert quarantined[0]["source_key"] == "tenant-1:mailbox-1:BAD-1"
    assert quarantined[0]["error_code"] == "missing_received_datetime"


def test_factory_wires_real_graph_client_through_mongo_langgraph_and_audit() -> None:
    """Exercise the complete redesigned path with offline Graph HTTP transport."""

    delta_link = "https://graph.test/v1.0/delta?$deltatoken=complete"
    transport = SinglePageTransport(
        {
            "value": [
                {
                    "id": "IMMUTABLE-GRAPH-1",
                    "internetMessageId": "<graph-1@example.test>",
                    "subject": "Home loan mortgage application",
                    "from": {
                        "emailAddress": {
                            "address": "customer@example.test",
                        }
                    },
                    "receivedDateTime": "2026-01-01T00:00:00Z",
                    "body": {
                        "contentType": "text",
                        "content": "Please provide my loan EMI schedule for account 991100221234.",
                    },
                    "hasAttachments": False,
                }
            ],
            "@odata.deltaLink": delta_link,
        }
    )
    cursors = InMemoryDeltaCursorStore()
    runtime = build_graph_polling_system(
        GraphClientConfig(
            "mailbox-1",
            tenant_id="tenant-1",
            base_url="https://graph.test/v1.0",
        ),
        StaticTokenProvider(),
        transport=transport,
        cursor_store=cursors,
    )

    result = runtime.coordinator.poll_once()

    assert result.processed == 1
    assert result.failed == 0
    assert result.cursor_committed
    assert cursors.get("mailbox-1", "inbox") == delta_link
    assert runtime.components.queues.count("loans-operations") == 1
    queue_item = runtime.components.queues.items("loans-operations")[0]
    assert "991100221234" not in queue_item.summary
    assert "[ACCOUNT_NUMBER]" in queue_item.summary
    document = runtime.components.document_store.get("IMMUTABLE-GRAPH-1")
    assert document is not None
    assert document["workflow"]["status"] == "COMPLETED"
    assert document["audit"]["status"] == "APPENDED"
    assert len(runtime.components.audit_log.entries()) == 1
    assert 'IdType="ImmutableId"' in transport.calls[0][1]["Prefer"]
