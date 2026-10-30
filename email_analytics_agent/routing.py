"""PII-safe case summarization and simulated downstream queue adapters."""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from threading import RLock
from uuid import NAMESPACE_URL, uuid5

from .config import AgentConfig
from .models import Prediction, PreprocessingResult, QueueItem
from .taxonomy import queue_for


class QueueDestinationConflict(RuntimeError):
    """Require reconciliation when one email is retried with a new destination."""


class SummarizerRoutingAgent:
    """Generate concise receiving-team context and resolve a target queue."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        """Configure the deterministic maximum size of case summaries."""

        self.config = config or AgentConfig()

    def summarize(self, preprocessed: PreprocessingResult, prediction: Prediction) -> str:
        """Summarize only already-redacted content without invoking an LLM."""

        words = preprocessed.model_text.replace("\n", " ").split()
        excerpt = " ".join(words[: self.config.summary_word_limit])
        if len(words) > self.config.summary_word_limit:
            excerpt += " …"
        return f"{prediction.label.department} case — {excerpt}"

    def target_queue(self, prediction: Prediction) -> str:
        """Map the prediction's CTD department to a downstream queue name."""

        return queue_for(prediction.label)


class MockQueueRouter:
    """Simulate ticketing queues while isolating enqueued payloads from callers."""

    def __init__(self) -> None:
        """Initialize empty named first-in-first-out queues."""

        self._queues: dict[str, deque[QueueItem]] = defaultdict(deque)
        self._receipts: dict[str, tuple[str, str]] = {}
        self._lock = RLock()

    def enqueue(self, item: QueueItem) -> str:
        """Enqueue one case idempotently and return its stable receipt ID."""

        with self._lock:
            existing = self._receipts.get(item.idempotency_key)
            if existing is not None:
                existing_receipt, existing_queue = existing
                if existing_queue != item.target_queue:
                    raise QueueDestinationConflict(
                        "routing destination changed for an existing email operation"
                    )
                return existing_receipt
            receipt = str(uuid5(NAMESPACE_URL, f"email-analytics:{item.idempotency_key}"))
            self._queues[item.target_queue].append(deepcopy(item))
            self._receipts[item.idempotency_key] = (receipt, item.target_queue)
            return receipt

    def receipt(self, message_id: str, target_queue: str) -> str | None:
        """Return the receipt for an existing queue command, when present."""

        with self._lock:
            existing = self._receipts.get(f"{message_id}:routing:v1")
            if existing is None or existing[1] != target_queue:
                return None
            return existing[0]

    def receipt_for_source(self, source_idempotency_key: str) -> str | None:
        """Return a receipt by composite source identity when one exists."""

        with self._lock:
            existing = self._receipts.get(f"{source_idempotency_key}:routing:v1")
            return existing[0] if existing is not None else None

    def items(self, queue_name: str) -> tuple[QueueItem, ...]:
        """Return a read-only snapshot of all cases in a named queue."""

        with self._lock:
            return tuple(deepcopy(tuple(self._queues[queue_name])))

    def count(self, queue_name: str | None = None) -> int:
        """Count one queue or all downstream and RM-review queues."""

        with self._lock:
            if queue_name is not None:
                return len(self._queues[queue_name])
            return sum(len(items) for items in self._queues.values())

