"""Microsoft Graph polling and provider-to-domain conversion boundaries.

The module deliberately separates authentication, HTTP transport, conversion,
and paging. Tests can therefore exercise Graph behavior without network access,
while a deployed process can use :class:`UrllibGraphTransport` with an injected
access-token provider.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, OpenerDirector, Request, build_opener

from .models import EmailMessage


DEFAULT_MESSAGE_FIELDS = (
    "id",
    "internetMessageId",
    "conversationId",
    "subject",
    "from",
    "toRecipients",
    "ccRecipients",
    "receivedDateTime",
    "body",
    "bodyPreview",
    "hasAttachments",
    "importance",
    "changeKey",
)


class GraphIngestionError(RuntimeError):
    """Base class for safe errors raised by the Graph ingestion boundary."""


class GraphAuthenticationError(GraphIngestionError):
    """Report unavailable credentials without including any secret material."""


class GraphNetworkError(GraphIngestionError):
    """Represent a transient network failure without leaking request details."""


class GraphPayloadError(GraphIngestionError):
    """Report an invalid Graph response using a non-sensitive reason code."""

    def __init__(self, reason_code: str) -> None:
        """Store and expose only a stable, non-sensitive failure code."""

        self.reason_code = reason_code
        super().__init__(f"invalid Microsoft Graph response: {reason_code}")


class GraphMessageConversionError(GraphIngestionError):
    """Identify a message that cannot be converted without echoing its data."""

    def __init__(self, reason_code: str) -> None:
        """Store and expose only a stable, non-sensitive conversion code."""

        self.reason_code = reason_code
        super().__init__(f"Microsoft Graph message conversion failed: {reason_code}")


class GraphHTTPError(GraphIngestionError):
    """Represent an HTTP failure using status and optional retry guidance."""

    def __init__(self, status_code: int, retry_after_seconds: float | None = None) -> None:
        """Create a safe HTTP error that never includes URLs, bodies, or tokens."""

        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"Microsoft Graph request failed with HTTP {status_code}")


class AccessTokenProvider(Protocol):
    """Provide replaceable authentication for Microsoft Graph requests."""

    def get_access_token(self) -> str:
        """Return a current bearer token without logging or persisting it."""

    def invalidate(self) -> None:
        """Discard any cached token after an authentication failure."""


class GraphTransport(Protocol):
    """Define the narrow JSON-over-HTTP operation required by the client."""

    def get_json(
        self,
        url: str,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        """Perform one authenticated GET and return a JSON object."""


class EnvironmentAccessTokenProvider:
    """Read a pre-acquired Graph bearer token from a named environment value.

    This adapter is useful for a POC or for a runtime whose platform injects a
    short-lived token. Production deployments can instead inject an MSAL,
    Azure Identity, certificate, or workload-identity implementation of
    :class:`AccessTokenProvider` without changing the Graph client.
    """

    def __init__(self, variable_name: str = "MERIDIAN_GRAPH_ACCESS_TOKEN") -> None:
        """Configure the environment variable without reading its secret yet."""

        if not variable_name.strip():
            raise ValueError("token environment variable name must not be empty")
        self._variable_name = variable_name

    def get_access_token(self) -> str:
        """Return a non-empty injected token or raise a safe configuration error."""

        token = os.environ.get(self._variable_name, "").strip()
        if not token:
            raise GraphAuthenticationError(
                f"Microsoft Graph access token is unavailable in {self._variable_name}"
            )
        return token

    def invalidate(self) -> None:
        """Perform no action because this provider does not cache the token."""


@dataclass(frozen=True, slots=True)
class GraphClientConfig:
    """Configure one mailbox's incremental Microsoft Graph polling client."""

    mailbox_id: str
    tenant_id: str = ""
    folder_id: str = "inbox"
    base_url: str = "https://graph.microsoft.com/v1.0"
    page_size: int = 50
    request_timeout_seconds: float = 30.0
    max_attempts: int = 3
    base_backoff_seconds: float = 0.5
    max_backoff_seconds: float = 8.0
    max_pages: int = 1000
    initial_received_after: str | None = None
    message_fields: tuple[str, ...] = field(default=DEFAULT_MESSAGE_FIELDS)

    def __post_init__(self) -> None:
        """Reject invalid or unsafe polling settings before network access."""

        if not self.mailbox_id.strip() or not self.folder_id.strip():
            raise ValueError("mailbox_id and folder_id must not be empty")
        parsed_base = urlsplit(self.base_url)
        if parsed_base.scheme != "https" or not parsed_base.netloc:
            raise ValueError("base_url must be an absolute HTTPS URL")
        if self.page_size < 1 or self.max_attempts < 1 or self.max_pages < 1:
            raise ValueError("page_size, max_attempts, and max_pages must be positive")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if self.base_backoff_seconds < 0 or self.max_backoff_seconds < 0:
            raise ValueError("backoff values must not be negative")
        if self.max_backoff_seconds < self.base_backoff_seconds:
            raise ValueError("max_backoff_seconds must be at least base_backoff_seconds")
        if not self.message_fields or any(not value.strip() for value in self.message_fields):
            raise ValueError("message_fields must contain non-empty field names")
        if self.initial_received_after is not None and not self.initial_received_after.strip():
            raise ValueError("initial_received_after must be a non-empty timestamp when set")


@dataclass(frozen=True, slots=True)
class GraphConversionFailure:
    """Describe one quarantinable conversion failure without retaining content."""

    message_id: str | None
    reason_code: str

    @property
    def error_code(self) -> str:
        """Expose the reason through the coordinator's generic error-code name."""

        return self.reason_code


@dataclass(frozen=True, slots=True)
class GraphPollResult:
    """Return fetched emails and the uncommitted cursor for a polling round."""

    messages: tuple[EmailMessage, ...]
    delta_link: str = field(repr=False)
    removed_message_ids: tuple[str, ...]
    conversion_failures: tuple[GraphConversionFailure, ...]
    pages_read: int


def _retry_after_seconds(value: str | None) -> float | None:
    """Convert a Retry-After seconds or HTTP-date value into a safe delay."""

    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    try:
        return max(0.0, float(normalized))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(normalized)
        except (TypeError, ValueError, OverflowError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())


class _SameOriginRedirectHandler(HTTPRedirectHandler):
    """Allow redirects only when the bearer token stays on the same origin."""

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        """Reject cross-origin redirects before urllib can forward headers."""

        resolved_url = urljoin(req.full_url, newurl)
        current = urlsplit(req.full_url)
        target = urlsplit(resolved_url)
        current_origin = (current.scheme.lower(), current.netloc.lower())
        target_origin = (target.scheme.lower(), target.netloc.lower())
        if current_origin != target_origin:
            return None
        return super().redirect_request(req, fp, code, msg, headers, resolved_url)


class UrllibGraphTransport:
    """Perform production-shaped Graph GET requests with the Python standard library."""

    def __init__(self, opener: OpenerDirector | None = None) -> None:
        """Create a transport whose redirect policy preserves token origin."""

        self._opener = opener or build_opener(_SameOriginRedirectHandler())

    def get_json(
        self,
        url: str,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        """Return one JSON object while translating failures into safe exceptions."""

        request = Request(url, headers=dict(headers), method="GET")
        try:
            with self._opener.open(request, timeout=timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            retry_after = _retry_after_seconds(
                exc.headers.get("Retry-After") if exc.headers is not None else None
            )
            raise GraphHTTPError(exc.code, retry_after) from None
        except URLError:
            raise GraphNetworkError("Microsoft Graph network request failed") from None
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise GraphPayloadError("invalid_json") from None
        if not isinstance(payload, Mapping):
            raise GraphPayloadError("root_is_not_object")
        return payload


class GraphMessageConverter:
    """Convert Graph message resources into the application's email domain model."""

    @staticmethod
    def _address_list(value: Any) -> list[str]:
        """Extract recipient addresses while ignoring malformed optional entries."""

        if not isinstance(value, list):
            return []
        addresses: list[str] = []
        for item in value:
            if not isinstance(item, Mapping):
                continue
            email_address = item.get("emailAddress")
            if not isinstance(email_address, Mapping):
                continue
            address = email_address.get("address")
            if isinstance(address, str) and address.strip():
                addresses.append(address.strip())
        return addresses

    def convert(
        self,
        value: Mapping[str, Any],
        mailbox_id: str,
        tenant_id: str = "",
    ) -> EmailMessage:
        """Translate one valid Graph message without performing business decisions."""

        message_id = value.get("id")
        if not isinstance(message_id, str) or not message_id.strip():
            raise GraphMessageConversionError("missing_message_id")
        received_at = value.get("receivedDateTime")
        if not isinstance(received_at, str) or not received_at.strip():
            raise GraphMessageConversionError("missing_received_datetime")

        sender = ""
        sender_value = value.get("from")
        if isinstance(sender_value, Mapping):
            email_address = sender_value.get("emailAddress")
            if isinstance(email_address, Mapping):
                address = email_address.get("address")
                if isinstance(address, str):
                    sender = address.strip()

        body_value = value.get("body")
        body = ""
        body_content_type = "unknown"
        if isinstance(body_value, Mapping):
            content = body_value.get("content")
            content_type = body_value.get("contentType")
            if isinstance(content, str):
                body = content
            if isinstance(content_type, str) and content_type.strip():
                body_content_type = content_type.strip().lower()
        if not body:
            body_preview = value.get("bodyPreview")
            if isinstance(body_preview, str):
                body = body_preview

        metadata: dict[str, Any] = {
            "provider": "microsoft-graph",
            "tenant_id": tenant_id,
            "mailbox_id": mailbox_id,
            "immutable_message_id": message_id.strip(),
            "idempotency_key": (
                f"{tenant_id}:{mailbox_id}:{message_id.strip()}"
                if tenant_id
                else f"{mailbox_id}:{message_id.strip()}"
            ),
            "internet_message_id": value.get("internetMessageId"),
            "conversation_id": value.get("conversationId"),
            "to_recipients": self._address_list(value.get("toRecipients")),
            "cc_recipients": self._address_list(value.get("ccRecipients")),
            "has_attachments": bool(value.get("hasAttachments", False)),
            "importance": value.get("importance"),
            "change_key": value.get("changeKey"),
            "source_etag": value.get("@odata.etag"),
            "body_content_type": body_content_type,
            "id_type": "immutable",
            "attachment_refs": [],
        }
        subject = value.get("subject")
        return EmailMessage(
            message_id=message_id.strip(),
            sender=sender,
            subject=subject if isinstance(subject, str) else "",
            body=body,
            received_at=received_at.strip(),
            metadata=metadata,
        )


def _default_jitter(delay_seconds: float) -> float:
    """Apply bounded jitter so multiple pollers do not retry in lockstep."""

    if delay_seconds <= 0:
        return 0.0
    return random.uniform(delay_seconds * 0.8, delay_seconds * 1.2)


class MicrosoftGraphMailClient:
    """Fetch incrementally changed Outlook messages through Microsoft Graph."""

    _RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})

    def __init__(
        self,
        config: GraphClientConfig,
        token_provider: AccessTokenProvider,
        transport: GraphTransport | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        jitter: Callable[[float], float] = _default_jitter,
    ) -> None:
        """Inject configuration and independently testable I/O dependencies."""

        self.config = config
        self.token_provider = token_provider
        self.transport = transport or UrllibGraphTransport()
        self._sleeper = sleeper
        self._jitter = jitter
        base = urlsplit(config.base_url)
        self._allowed_origin = (base.scheme.lower(), base.netloc.lower())

    def _initial_delta_url(self) -> str:
        """Build the first created-message delta request for the configured folder."""

        mailbox = quote(self.config.mailbox_id, safe="")
        folder = quote(self.config.folder_id, safe="")
        query_values = {
            "changeType": "created",
            "$select": ",".join(self.config.message_fields),
        }
        if self.config.initial_received_after is not None:
            query_values["$filter"] = (
                f"receivedDateTime ge {self.config.initial_received_after.strip()}"
            )
        query = urlencode(query_values)
        base = self.config.base_url.rstrip("/")
        return f"{base}/users/{mailbox}/mailFolders/{folder}/messages/delta?{query}"

    def _validate_graph_url(self, url: str) -> str:
        """Reject malformed or cross-origin paging links before authentication."""

        parsed = urlsplit(url)
        origin = (parsed.scheme.lower(), parsed.netloc.lower())
        if origin != self._allowed_origin or not parsed.path:
            raise GraphPayloadError("untrusted_paging_url")
        return url

    def _headers(self, token: str) -> dict[str, str]:
        """Create Graph request headers without retaining the bearer token."""

        if not token.strip():
            raise GraphAuthenticationError("Microsoft Graph access token is empty")
        preferences = ", ".join(
            (
                'IdType="ImmutableId"',
                'outlook.body-content-type="text"',
                f"odata.maxpagesize={self.config.page_size}",
            )
        )
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "Prefer": preferences,
        }

    def _backoff_seconds(self, attempt: int) -> float:
        """Return a bounded exponential delay for the completed attempt number."""

        raw_delay = self.config.base_backoff_seconds * (2 ** max(0, attempt - 1))
        return self._jitter(min(raw_delay, self.config.max_backoff_seconds))

    def _get_json(self, url: str) -> Mapping[str, Any]:
        """GET one page with bounded authentication, throttle, and network retries."""

        trusted_url = self._validate_graph_url(url)
        for attempt in range(1, self.config.max_attempts + 1):
            token = self.token_provider.get_access_token()
            try:
                return self.transport.get_json(
                    trusted_url,
                    self._headers(token),
                    self.config.request_timeout_seconds,
                )
            except GraphHTTPError as exc:
                if attempt >= self.config.max_attempts:
                    raise
                if exc.status_code == 401:
                    self.token_provider.invalidate()
                    continue
                if exc.status_code not in self._RETRYABLE_STATUS_CODES:
                    raise
                delay = (
                    exc.retry_after_seconds
                    if exc.retry_after_seconds is not None
                    else self._backoff_seconds(attempt)
                )
                self._sleeper(max(0.0, delay))
            except GraphNetworkError:
                if attempt >= self.config.max_attempts:
                    raise
                self._sleeper(max(0.0, self._backoff_seconds(attempt)))
        raise AssertionError("retry loop exited without returning or raising")

    @staticmethod
    def _safe_message_id(value: Mapping[str, Any]) -> str | None:
        """Return a safe identifier for diagnostics without inspecting content."""

        message_id = value.get("id")
        return message_id.strip() if isinstance(message_id, str) and message_id.strip() else None

    def fetch_recent(self, delta_link: str | None = None) -> GraphPollResult:
        """Fetch a complete delta round and return its cursor without committing it.

        The caller must durably claim all returned messages and quarantine all
        conversion failures before persisting ``result.delta_link``. This rule
        prevents cursor advancement from losing an email after a process crash.
        """

        current_url = self._validate_graph_url(delta_link) if delta_link else self._initial_delta_url()
        seen_page_urls: set[str] = set()
        messages_by_id: dict[str, EmailMessage] = {}
        removed_ids: list[str] = []
        failures: list[GraphConversionFailure] = []
        pages_read = 0
        final_delta_link: str | None = None
        converter = GraphMessageConverter()

        while True:
            if current_url in seen_page_urls:
                raise GraphPayloadError("paging_cycle")
            seen_page_urls.add(current_url)
            pages_read += 1
            if pages_read > self.config.max_pages:
                raise GraphPayloadError("page_limit_exceeded")

            payload = self._get_json(current_url)
            values = payload.get("value")
            if not isinstance(values, list):
                raise GraphPayloadError("missing_message_collection")

            for value in values:
                if not isinstance(value, Mapping):
                    failures.append(GraphConversionFailure(None, "message_is_not_object"))
                    continue
                message_id = self._safe_message_id(value)
                if "@removed" in value:
                    if message_id is None:
                        failures.append(GraphConversionFailure(None, "removed_message_missing_id"))
                    else:
                        removed_ids.append(message_id)
                    continue
                try:
                    email = converter.convert(
                        value,
                        self.config.mailbox_id,
                        self.config.tenant_id,
                    )
                except GraphMessageConversionError as exc:
                    failures.append(GraphConversionFailure(message_id, exc.reason_code))
                    continue
                messages_by_id[email.message_id] = email

            next_link = payload.get("@odata.nextLink")
            if next_link is not None:
                if not isinstance(next_link, str) or not next_link.strip():
                    raise GraphPayloadError("invalid_next_link")
                current_url = self._validate_graph_url(next_link)
                continue

            delta_value = payload.get("@odata.deltaLink")
            if not isinstance(delta_value, str) or not delta_value.strip():
                raise GraphPayloadError("missing_delta_link")
            final_delta_link = self._validate_graph_url(delta_value)
            break

        return GraphPollResult(
            messages=tuple(messages_by_id.values()),
            delta_link=final_delta_link,
            removed_message_ids=tuple(dict.fromkeys(removed_ids)),
            conversion_failures=tuple(failures),
            pages_read=pages_read,
        )
