"""Meridian Email Analytics Agent proof-of-concept package."""

from .config import AgentConfig
from .classification import SklearnCTDClassifier
from .factory import GraphPollingSystem, build_graph_polling_system
from .graph_ingestion import (
    EnvironmentAccessTokenProvider,
    GraphClientConfig,
    MicrosoftGraphMailClient,
)
from .models import CTDLabel, EmailMessage, ProcessResult
from .polling import GraphPollingCoordinator, InMemoryDeltaCursorStore, PollingRunResult

__all__ = [
    "AgentConfig",
    "CTDLabel",
    "EmailMessage",
    "EnvironmentAccessTokenProvider",
    "GraphClientConfig",
    "GraphPollingCoordinator",
    "GraphPollingSystem",
    "InMemoryDeltaCursorStore",
    "MicrosoftGraphMailClient",
    "PollingRunResult",
    "ProcessResult",
    "SklearnCTDClassifier",
    "build_graph_polling_system",
]

