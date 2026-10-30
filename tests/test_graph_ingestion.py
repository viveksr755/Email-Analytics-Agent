"""Offline tests for Microsoft Graph polling and message conversion."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request

import pytest

from email_analytics_agent.graph_ingestion import (
    EnvironmentAccessTokenProvider,
    GraphAuthenticationError,
    GraphClientConfig,
    GraphHTTPError,
    GraphMessageConversionError,
    GraphMessageConverter,
    GraphNetworkError,
    GraphPayloadError,
    MicrosoftGraphMailClient,
    _SameOriginRedirectHandler,
)


class FakeTokenProvider:
    """Return deterministic tokens and expose invalidation behavior to tests."""

    def __init__(self, *tokens: str) -> None:
        """Store one or more tokens returned after successive invalidations."""

        self.tokens = tokens or ("test-token",)
        self.index = 0
        self.invalidations = 0

    def get_access_token(self) -> str:
        """Return the token selected by the current cache generation."""

        return self.tokens[min(self.index, len(self.tokens) - 1)]

    def invalidate(self) -> None:
        """Advance to the next token and record the invalidation."""

        self.invalidations += 1
        self.index += 1


class RecordingTransport:
    """Replay queued JSON responses or exceptions without network access."""

    def __init__(self, *responses: Mapping[str, Any] | Exception) -> None:
        """Configure ordered outcomes and initialize a request ledger."""

        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, str], float]] = []

    def get_json(
        self,
        url: str,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        """Record one request and return or raise its configured outcome."""

        self.calls.append((url, dict(headers), timeout_seconds))
        if not self.responses:
            raise AssertionError("test transport received an unexpected request")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _graph_message(message_id: str = "IMMUTABLE-1") -> dict[str, Any]:
    """Build a representative Graph message response for conversion tests."""

    return {
        "@odata.etag": 'W/"etag-1"',
        "id": message_id,
        "internetMessageId": "<mail-1@example.test>",
        "conversationId": "CONV-1",
        "subject": "Mortgage repayment request",
        "from": {"emailAddress": {"name": "A Customer", "address": "customer@example.test"}},
        "toRecipients": [
            {"emailAddress": {"name": "Operations", "address": "ops@example.test"}}
        ],
        "ccRecipients": [
            {"emailAddress": {"name": "RM", "address": "rm@example.test"}}
        ],
        "receivedDateTime": "2026-10-30T10:00:00Z",
        "body": {"contentType": "text", "content": "Loan account 991100221234."},
        "bodyPreview": "Loan account...",
        "hasAttachments": True,
        "importance": "high",
        "changeKey": "CHANGE-1",
    }


def _config(**overrides: Any) -> GraphClientConfig:
    """Create a small deterministic Graph configuration for tests."""

    values: dict[str, Any] = {
        "mailbox_id": "mailbox-object-id",
        "base_url": "https://graph.test/v1.0",
        "base_backoff_seconds": 0.25,
        "max_backoff_seconds": 1.0,
    }
    values.update(overrides)
    return GraphClientConfig(**values)


def _client(
    transport: RecordingTransport,
    token_provider: FakeTokenProvider | None = None,
    sleeps: list[float] | None = None,
    **config_overrides: Any,
) -> MicrosoftGraphMailClient:
    """Construct a client with deterministic retry timing and fake I/O."""

    observed_sleeps = sleeps if sleeps is not None else []
    return MicrosoftGraphMailClient(
        _config(**config_overrides),
        token_provider or FakeTokenProvider(),
        transport,
        sleeper=observed_sleeps.append,
        jitter=lambda delay: delay,
    )


def test_converter_maps_graph_message_to_provider_independent_email() -> None:
    """Verify immutable identity, content, recipients, and metadata mapping."""

    email = GraphMessageConverter().convert(_graph_message(), "mailbox-object-id")

    assert email.message_id == "IMMUTABLE-1"
    assert email.sender == "customer@example.test"
    assert email.subject == "Mortgage repayment request"
    assert email.body == "Loan account 991100221234."
    assert email.received_at == "2026-10-30T10:00:00Z"
    assert email.metadata["provider"] == "microsoft-graph"
    assert email.metadata["mailbox_id"] == "mailbox-object-id"
    assert email.metadata["to_recipients"] == ["ops@example.test"]
    assert email.metadata["cc_recipients"] == ["rm@example.test"]
    assert email.metadata["source_etag"] == 'W/"etag-1"'
    assert email.metadata["id_type"] == "immutable"


def test_converter_requires_message_id_and_received_timestamp() -> None:
    """Ensure records without stable identity or receipt time are quarantinable."""

    message = _graph_message()
    message.pop("id")
    with pytest.raises(GraphMessageConversionError, match="missing_message_id"):
        GraphMessageConverter().convert(message, "mailbox-object-id")

    message = _graph_message()
    message.pop("receivedDateTime")
    with pytest.raises(GraphMessageConversionError, match="missing_received_datetime"):
        GraphMessageConverter().convert(message, "mailbox-object-id")


def test_fetch_recent_follows_pages_and_returns_uncommitted_delta_cursor() -> None:
    """Verify paging, conversion failures, removals, and final delta handling."""

    next_link = "https://graph.test/v1.0/next?page=2&token=opaque"
    delta_link = "https://graph.test/v1.0/delta?token=final-opaque"
    malformed = _graph_message("BROKEN-1")
    malformed.pop("receivedDateTime")
    transport = RecordingTransport(
        {
            "value": [_graph_message("IMMUTABLE-1"), malformed],
            "@odata.nextLink": next_link,
        },
        {
            "value": [
                _graph_message("IMMUTABLE-2"),
                {"id": "REMOVED-1", "@removed": {"reason": "deleted"}},
            ],
            "@odata.deltaLink": delta_link,
        },
    )
    result = _client(transport).fetch_recent()

    assert [email.message_id for email in result.messages] == ["IMMUTABLE-1", "IMMUTABLE-2"]
    assert result.delta_link == delta_link
    assert result.removed_message_ids == ("REMOVED-1",)
    assert result.conversion_failures[0].message_id == "BROKEN-1"
    assert result.conversion_failures[0].reason_code == "missing_received_datetime"
    assert result.pages_read == 2
    assert transport.calls[1][0] == next_link

    initial_url, initial_headers, timeout = transport.calls[0]
    parsed = urlsplit(initial_url)
    query = parse_qs(parsed.query)
    assert parsed.path.endswith("/users/mailbox-object-id/mailFolders/inbox/messages/delta")
    assert query["changeType"] == ["created"]
    assert "id" in query["$select"][0].split(",")
    assert initial_headers["Authorization"] == "Bearer test-token"
    assert 'IdType="ImmutableId"' in initial_headers["Prefer"]
    assert 'outlook.body-content-type="text"' in initial_headers["Prefer"]
    assert timeout == 30.0


def test_fetch_recent_uses_saved_delta_link_verbatim() -> None:
    """Ensure a stored Graph cursor is reused without query reconstruction."""

    saved_link = "https://graph.test/v1.0/delta?$deltatoken=opaque%2Bvalue"
    next_link = "https://graph.test/v1.0/delta?$deltatoken=new-value"
    transport = RecordingTransport({"value": [], "@odata.deltaLink": next_link})

    result = _client(transport).fetch_recent(saved_link)

    assert transport.calls[0][0] == saved_link
    assert result.delta_link == next_link
    assert result.messages == ()


def test_initial_fetch_can_be_bounded_to_a_recent_received_timestamp() -> None:
    """Allow the first delta snapshot to avoid importing unlimited history."""

    transport = RecordingTransport(
        {
            "value": [],
            "@odata.deltaLink": "https://graph.test/v1.0/delta?token=done",
        }
    )

    _client(
        transport,
        initial_received_after="2026-10-01T00:00:00Z",
    ).fetch_recent()

    query = parse_qs(urlsplit(transport.calls[0][0]).query)
    assert query["$filter"] == ["receivedDateTime ge 2026-10-01T00:00:00Z"]


def test_throttled_request_honors_retry_after_without_leaking_token() -> None:
    """Verify a 429 delay uses server guidance and succeeds on a later attempt."""

    sleeps: list[float] = []
    delta_link = "https://graph.test/v1.0/delta?token=done"
    transport = RecordingTransport(
        GraphHTTPError(429, retry_after_seconds=2.5),
        {"value": [], "@odata.deltaLink": delta_link},
    )

    result = _client(transport, sleeps=sleeps).fetch_recent()

    assert result.delta_link == delta_link
    assert sleeps == [2.5]
    assert len(transport.calls) == 2
    assert "test-token" not in str(GraphHTTPError(429, 2.5))


def test_transient_network_failure_uses_bounded_exponential_backoff() -> None:
    """Verify transport failures retry with injectable deterministic timing."""

    sleeps: list[float] = []
    transport = RecordingTransport(
        GraphNetworkError("safe network failure"),
        GraphHTTPError(503),
        {"value": [], "@odata.deltaLink": "https://graph.test/v1.0/delta?token=done"},
    )

    _client(transport, sleeps=sleeps).fetch_recent()

    assert sleeps == [0.25, 0.5]
    assert len(transport.calls) == 3


def test_unauthorized_request_invalidates_token_before_retry() -> None:
    """Verify a 401 obtains a replacement token without exposing either value."""

    provider = FakeTokenProvider("expired-secret-token", "fresh-secret-token")
    transport = RecordingTransport(
        GraphHTTPError(401),
        {"value": [], "@odata.deltaLink": "https://graph.test/v1.0/delta?token=done"},
    )

    _client(transport, token_provider=provider).fetch_recent()

    assert provider.invalidations == 1
    assert transport.calls[0][1]["Authorization"] == "Bearer expired-secret-token"
    assert transport.calls[1][1]["Authorization"] == "Bearer fresh-secret-token"
    assert "expired-secret-token" not in str(GraphHTTPError(401))


def test_forbidden_request_is_not_retried() -> None:
    """Treat authorization configuration failures as non-transient."""

    sleeps: list[float] = []
    transport = RecordingTransport(GraphHTTPError(403))

    with pytest.raises(GraphHTTPError) as raised:
        _client(transport, sleeps=sleeps).fetch_recent()

    assert raised.value.status_code == 403
    assert len(transport.calls) == 1
    assert sleeps == []


def test_untrusted_paging_link_is_rejected_before_credentials_are_sent() -> None:
    """Prevent a malicious next link from receiving the Graph bearer token."""

    transport = RecordingTransport(
        {
            "value": [],
            "@odata.nextLink": "https://attacker.example/collect",
        }
    )

    with pytest.raises(GraphPayloadError, match="untrusted_paging_url"):
        _client(transport).fetch_recent()

    assert len(transport.calls) == 1


def test_cross_origin_http_redirect_is_rejected_before_token_forwarding() -> None:
    """Harden urllib itself in addition to validating Graph paging links."""

    request = Request(
        "https://graph.test/v1.0/messages",
        headers={"Authorization": "Bearer secret-token"},
    )

    redirected = _SameOriginRedirectHandler().redirect_request(
        request,
        None,
        302,
        "Found",
        {},
        "https://attacker.example/collect",
    )

    assert redirected is None


def test_paging_cycle_is_rejected_instead_of_looping() -> None:
    """Stop when Graph repeats a next link already visited in the same round."""

    repeated = "https://graph.test/v1.0/repeated?page=1"
    transport = RecordingTransport(
        {"value": [], "@odata.nextLink": repeated},
        {"value": [], "@odata.nextLink": repeated},
    )

    with pytest.raises(GraphPayloadError, match="paging_cycle"):
        _client(transport).fetch_recent()

    assert len(transport.calls) == 2


def test_environment_token_provider_reads_named_value_and_fails_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify environment injection without returning or printing missing secrets."""

    variable = "MERIDIAN_TEST_GRAPH_TOKEN"
    provider = EnvironmentAccessTokenProvider(variable)
    monkeypatch.setenv(variable, "  short-lived-token  ")
    assert provider.get_access_token() == "short-lived-token"
    provider.invalidate()

    monkeypatch.delenv(variable)
    with pytest.raises(GraphAuthenticationError) as raised:
        provider.get_access_token()
    assert variable in str(raised.value)
    assert "short-lived-token" not in str(raised.value)
