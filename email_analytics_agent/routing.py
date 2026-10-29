"""PII-safe case summarization and simulated downstream queue adapters."""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy

from .config import AgentConfig
from .models import Prediction, PreprocessingResult, QueueItem
from .taxonomy import queue_for


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

    def enqueue(self, item: QueueItem) -> None:
        """Append a detached case to its declared target queue."""

        self._queues[item.target_queue].append(deepcopy(item))

    def items(self, queue_name: str) -> tuple[QueueItem, ...]:
        """Return a read-only snapshot of all cases in a named queue."""

        return tuple(deepcopy(tuple(self._queues[queue_name])))

    def count(self, queue_name: str | None = None) -> int:
        """Count one queue or all downstream and RM-review queues."""

        if queue_name is not None:
            return len(self._queues[queue_name])
        return sum(len(items) for items in self._queues.values())

