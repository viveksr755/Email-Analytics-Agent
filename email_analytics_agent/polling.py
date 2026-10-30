"""Polling coordinator for the Microsoft Graph to LangGraph runtime flow."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Protocol

from .graph_ingestion import GraphConversionFailure, MicrosoftGraphMailClient
from .models import ProcessResult
from .orchestration import ManagerAgent


class DeltaCursorStore(Protocol):
    """Persist opaque Microsoft Graph delta links by mailbox and folder."""

    def get(self, mailbox_id: str, folder_id: str) -> str | None:
        """Return the last committed delta link, if one exists."""

    def commit(self, mailbox_id: str, folder_id: str, delta_link: str) -> None:
        """Atomically replace the committed delta link for one folder."""


class InMemoryDeltaCursorStore:
    """Provide a thread-safe POC cursor store with production-shaped semantics."""

    def __init__(self) -> None:
        """Initialize an empty mailbox-folder cursor map."""

        self._cursors: dict[tuple[str, str], str] = {}
        self._lock = RLock()

    def get(self, mailbox_id: str, folder_id: str) -> str | None:
        """Return the currently committed cursor without exposing mutable state."""

        with self._lock:
            return self._cursors.get((mailbox_id, folder_id))

    def commit(self, mailbox_id: str, folder_id: str, delta_link: str) -> None:
        """Commit a non-empty opaque delta link for a mailbox folder."""

        if not delta_link:
            raise ValueError("delta_link must not be empty")
        with self._lock:
            self._cursors[(mailbox_id, folder_id)] = delta_link


@dataclass(frozen=True, slots=True)
class PollingFailure:
    """Describe one safe polling failure without retaining email content."""

    message_id: str
    error_code: str
    quarantined: bool = False


@dataclass(frozen=True, slots=True)
class PollingRunResult:
    """Summarize one Graph polling cycle and its durable processing outcomes."""

    fetched: int
    processed: int
    skipped: int
    failed: int
    quarantined: int
    removed: int
    pages_read: int
    cursor_committed: bool
    outcomes: tuple[ProcessResult, ...]
    failures: tuple[PollingFailure, ...]


class GraphPollingCoordinator:
    """Fetch Graph mail, invoke the idempotent manager, then commit the cursor.

    The cursor is advanced only after every converted email has reached a
    durable automated outcome and each identifiable conversion failure has
    been quarantined. Replayed pages are harmless because the manager's
    document store claims messages by a stable idempotency key before entering
    LangGraph.
    """

    def __init__(
        self,
        graph_client: MicrosoftGraphMailClient,
        manager: ManagerAgent,
        cursor_store: DeltaCursorStore | None = None,
    ) -> None:
        """Inject the Graph connector, workflow manager, and cursor boundary."""

        self.graph_client = graph_client
        self.manager = manager
        self.cursor_store = cursor_store or InMemoryDeltaCursorStore()

    def _conversion_failure(
        self,
        value: GraphConversionFailure,
    ) -> PollingFailure:
        """Quarantine an identifiable poison record without copying content."""

        message_id = value.message_id or "unknown"
        if value.message_id is None:
            return PollingFailure(message_id, value.error_code)
        config = self.graph_client.config
        source_key = ":".join(
            part
            for part in (config.tenant_id, config.mailbox_id, value.message_id)
            if part
        )
        try:
            self.manager.document_store.quarantine_ingestion_failure(
                source_key,
                value.error_code,
            )
        except Exception:  # noqa: BLE001 - cursor must remain blocked on storage failure
            return PollingFailure(message_id, value.error_code)
        return PollingFailure(message_id, value.error_code, quarantined=True)

    def poll_once(self) -> PollingRunResult:
        """Execute one complete mailbox polling cycle.

        Fetch-level errors propagate to the scheduler so it can apply its
        operational retry policy. Per-message workflow failures are isolated,
        reported with safe exception class names, and prevent cursor commit.
        """

        config = self.graph_client.config
        current_cursor = self.cursor_store.get(config.mailbox_id, config.folder_id)
        batch = self.graph_client.fetch_recent(current_cursor)
        failures = [self._conversion_failure(value) for value in batch.conversion_failures]
        has_blocking_failure = any(not value.quarantined for value in failures)
        outcomes: list[ProcessResult] = []
        processed = 0
        skipped = 0

        for email in batch.messages:
            try:
                outcome = self.manager.process_email(email)
            except Exception as error:  # noqa: BLE001 - isolation is the coordinator's boundary
                failures.append(PollingFailure(email.message_id, type(error).__name__))
                has_blocking_failure = True
                continue
            outcomes.append(outcome)
            if outcome.was_skipped:
                skipped += 1
            else:
                processed += 1

        cursor_committed = False
        if not has_blocking_failure and batch.delta_link:
            self.cursor_store.commit(config.mailbox_id, config.folder_id, batch.delta_link)
            cursor_committed = True

        return PollingRunResult(
            fetched=len(batch.messages),
            processed=processed,
            skipped=skipped,
            failed=len(failures),
            quarantined=sum(value.quarantined for value in failures),
            removed=len(batch.removed_message_ids),
            pages_read=batch.pages_read,
            cursor_committed=cursor_committed,
            outcomes=tuple(outcomes),
            failures=tuple(failures),
        )
