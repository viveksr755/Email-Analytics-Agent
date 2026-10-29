"""Composition root for constructing the complete local proof-of-concept."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .audit import ImmutableAuditLog
from .classification import BaselineCTDClassifier
from .config import AgentConfig
from .guardrails import ComplianceGuardrailAgent
from .orchestration import ManagerAgent
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


def build_default_system(
    audit_path: str | Path | None = None,
    config: AgentConfig | None = None,
) -> SystemComponents:
    """Wire deterministic adapters into the production-shaped LangGraph manager."""

    resolved_config = config or AgentConfig()
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
        BaselineCTDClassifier(),
        rag,
        guardrail,
        SummarizerRoutingAgent(resolved_config),
        queues,
        documents,
        resolved_config,
        audit_log,
    )
    return SystemComponents(manager, llm_client, queues, documents, vectors, audit_log)

