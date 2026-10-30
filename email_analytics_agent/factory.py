"""Composition root for constructing the complete local proof-of-concept."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .audit import ImmutableAuditLog
from .classification import SklearnCTDClassifier
from .config import AgentConfig
from .guardrails import ComplianceGuardrailAgent
from .graph_ingestion import (
    AccessTokenProvider,
    GraphClientConfig,
    GraphTransport,
    MicrosoftGraphMailClient,
)
from .orchestration import ManagerAgent
from .polling import DeltaCursorStore, GraphPollingCoordinator
from .preprocessing import EmailPreprocessor, PIIRedactor
from .rag import LocalLLMClient, RAGClassificationAgent, seed_routing_policies
from .routing import MockQueueRouter, SummarizerRoutingAgent
from .storage import MockMongoStore, MockVectorStore


@dataclass(frozen=True, slots=True)
class SystemComponents:
    """Expose POC boundaries for CLI execution, tests, and demonstrations."""

    manager: ManagerAgent
    llm_client: LocalLLMClient
    queues: MockQueueRouter
    document_store: MockMongoStore
    vector_store: MockVectorStore
    audit_log: ImmutableAuditLog


@dataclass(frozen=True, slots=True)
class GraphPollingSystem:
    """Expose the fully wired Graph polling flow and its POC components."""

    components: SystemComponents
    graph_client: MicrosoftGraphMailClient
    coordinator: GraphPollingCoordinator


def build_default_system(
    audit_path: str | Path | None = None,
    config: AgentConfig | None = None,
) -> SystemComponents:
    """Wire deterministic adapters into the production-shaped LangGraph manager."""

    ml_classifier = SklearnCTDClassifier()
    resolved_config = config or AgentConfig(
        ml_confidence_threshold=float(ml_classifier.metadata["selected_routing_threshold"])
    )
    redactor = PIIRedactor()
    guardrail = ComplianceGuardrailAgent(redactor)
    vectors = MockVectorStore()
    seed_routing_policies(vectors)
    llm_client = LocalLLMClient()
    queues = MockQueueRouter()
    documents = MockMongoStore()
    audit_log = ImmutableAuditLog(audit_path)
    rag = RAGClassificationAgent(vectors, llm_client, guardrail, resolved_config)
    manager = ManagerAgent(
        EmailPreprocessor(redactor),
        ml_classifier,
        rag,
        guardrail,
        SummarizerRoutingAgent(resolved_config),
        queues,
        documents,
        resolved_config,
        audit_log,
    )
    return SystemComponents(manager, llm_client, queues, documents, vectors, audit_log)


def build_graph_polling_system(
    graph_config: GraphClientConfig,
    token_provider: AccessTokenProvider,
    audit_path: str | Path | None = None,
    agent_config: AgentConfig | None = None,
    transport: GraphTransport | None = None,
    cursor_store: DeltaCursorStore | None = None,
) -> GraphPollingSystem:
    """Wire Microsoft Graph polling into the complete idempotent agent flow.

    The returned coordinator performs one folder-delta round, converts Graph
    messages, claims them in the document store, invokes LangGraph, persists
    the queue outcome before audit, and commits the delta cursor only after the
    complete batch is durable.
    """

    components = build_default_system(audit_path, agent_config)
    graph_client = MicrosoftGraphMailClient(
        graph_config,
        token_provider,
        transport=transport,
    )
    coordinator = GraphPollingCoordinator(
        graph_client,
        components.manager,
        cursor_store,
    )
    return GraphPollingSystem(components, graph_client, coordinator)

