"""In-memory adapters that model MongoDB and a vector policy database."""

from __future__ import annotations

import math
import re
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

from .models import EmailMessage, Prediction, utc_now_iso


class EmailDocumentStore(Protocol):
    """Define persistence operations required by the orchestration layer."""

    def save_email(self, email: EmailMessage) -> None:
        """Persist raw email content in the secured document-store boundary."""

    def save_classification(self, message_id: str, prediction: Prediction) -> None:
        """Attach the final CTD prediction to an existing email document."""


class MockMongoStore:
    """Simulate MongoDB with detached documents indexed by message ID."""

    def __init__(self) -> None:
        """Initialize an empty document collection."""

        self._documents: dict[str, dict[str, Any]] = {}

    def save_email(self, email: EmailMessage) -> None:
        """Insert an email document, rejecting duplicate message IDs."""

        if email.message_id in self._documents:
            raise ValueError(f"duplicate message_id: {email.message_id}")
        self._documents[email.message_id] = {
            "email": email.to_dict(),
            "classification": None,
            "created_at": utc_now_iso(),
        }

    def save_classification(self, message_id: str, prediction: Prediction) -> None:
        """Update only the CTD section of an existing stored email."""

        if message_id not in self._documents:
            raise KeyError(f"unknown message_id: {message_id}")
        self._documents[message_id]["classification"] = prediction.to_dict()
        self._documents[message_id]["classified_at"] = utc_now_iso()

    def get(self, message_id: str) -> dict[str, Any] | None:
        """Return a defensive copy of one document when it exists."""

        document = self._documents.get(message_id)
        return deepcopy(document) if document is not None else None

    def count(self) -> int:
        """Return the number of stored email documents."""

        return len(self._documents)


@dataclass(frozen=True, slots=True)
class PolicyDocument:
    """Store a routing-policy chunk and its safe retrieval metadata."""

    document_id: str
    text: str
    metadata: dict[str, str]


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Pair a retrieved policy document with its cosine similarity score."""

    document: PolicyDocument
    score: float


class MockVectorStore:
    """Provide deterministic bag-of-words embeddings and semantic retrieval."""

    _TOKEN_RE = re.compile(r"[a-z][a-z0-9-]+", re.IGNORECASE)

    def __init__(self) -> None:
        """Create an empty policy-document index."""

        self._documents: dict[str, PolicyDocument] = {}
        self._vectors: dict[str, Counter[str]] = {}

    @classmethod
    def _embed(cls, text: str) -> Counter[str]:
        """Map text into a normalized-token frequency vector."""

        return Counter(token.lower() for token in cls._TOKEN_RE.findall(text))

    @staticmethod
    def _cosine(left: Counter[str], right: Counter[str]) -> float:
        """Compute cosine similarity between two sparse frequency vectors."""

        if not left or not right:
            return 0.0
        dot = sum(value * right.get(token, 0) for token, value in left.items())
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        return dot / (left_norm * right_norm)

    def add(self, document: PolicyDocument) -> None:
        """Insert or replace a policy chunk and its deterministic embedding."""

        safe_document = deepcopy(document)
        self._documents[document.document_id] = safe_document
        self._vectors[document.document_id] = self._embed(document.text)

    def search(self, query: str, limit: int = 3) -> tuple[SearchResult, ...]:
        """Return the most similar policy chunks with deterministic tie-breaking."""

        if limit < 1:
            raise ValueError("limit must be positive")
        query_vector = self._embed(query)
        ranked = [
            SearchResult(deepcopy(document), self._cosine(query_vector, self._vectors[document_id]))
            for document_id, document in self._documents.items()
        ]
        ranked.sort(key=lambda item: (-item.score, item.document.document_id))
        return tuple(ranked[:limit])

