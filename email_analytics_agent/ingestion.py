"""Mock mail ingestion and event-stream components used by the POC."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol

from .models import EmailMessage


class EventPublisher(Protocol):
    """Define the narrow event publication contract used by ingestion."""

    def publish(self, topic: str, email: EmailMessage) -> None:
        """Publish one email to a named event-stream topic."""


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """Represent an immutable message read from the mocked event stream."""

    offset: int
    topic: str
    email: EmailMessage


class MockEventStream:
    """Simulate Kafka topics with ordered, replayable in-memory events."""

    def __init__(self) -> None:
        """Create an empty stream whose offsets begin at zero."""

        self._events: list[StreamEvent] = []
        self._pending: dict[str, deque[int]] = {}

    def publish(self, topic: str, email: EmailMessage) -> None:
        """Append a detached email event and make it available to consumers."""

        offset = len(self._events)
        event = StreamEvent(offset=offset, topic=topic, email=deepcopy(email))
        self._events.append(event)
        self._pending.setdefault(topic, deque()).append(offset)

    def consume(self, topic: str) -> StreamEvent | None:
        """Consume the next pending event for a topic, or return ``None``."""

        offsets = self._pending.setdefault(topic, deque())
        if not offsets:
            return None
        return deepcopy(self._events[offsets.popleft()])

    def replay(self, topic: str, from_offset: int = 0) -> tuple[StreamEvent, ...]:
        """Return topic events from an offset without altering consumer state."""

        return tuple(
            deepcopy(event)
            for event in self._events
            if event.topic == topic and event.offset >= from_offset
        )


class MockMailConnector:
    """Simulate polling O365/IMAP and publishing raw emails to a stream."""

    def __init__(self, messages: Iterable[EmailMessage]) -> None:
        """Snapshot source messages to make repeated test runs deterministic."""

        self._messages = tuple(deepcopy(tuple(messages)))

    def poll(self) -> tuple[EmailMessage, ...]:
        """Return the synthetic mailbox contents as detached objects."""

        return deepcopy(self._messages)


class MockIngestionPipeline:
    """Coordinate the mock mail connector and raw-email event topic."""

    def __init__(
        self,
        connector: MockMailConnector,
        publisher: EventPublisher,
        topic: str = "raw-emails",
    ) -> None:
        """Configure connector, publisher, and destination topic."""

        self._connector = connector
        self._publisher = publisher
        self._topic = topic

    def ingest(self) -> int:
        """Publish all currently polled messages and return the event count."""

        messages = self._connector.poll()
        for message in messages:
            self._publisher.publish(self._topic, message)
        return len(messages)

