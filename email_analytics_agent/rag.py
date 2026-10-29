"""Retrieval-augmented local LLM mock with mandatory safety boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .config import AgentConfig
from .guardrails import ComplianceGuardrailAgent
from .models import Prediction
from .storage import MockVectorStore, PolicyDocument
from .taxonomy import CTD_BY_DEPARTMENT, UNKNOWN_CTD


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """Record exactly what the local mock received for privacy assertions."""

    email_text: str
    policy_context: tuple[str, ...]


class LLMClient(Protocol):
    """Define the structured classification call used by the RAG agent."""

    def classify(self, email_text: str, policy_context: tuple[str, ...]) -> Prediction:
        """Return one CTD prediction from safe text and retrieved policy chunks."""


class LocalLLMClient:
    """Simulate an LLM locally with broader semantics than the baseline model."""

    MODEL_VERSION = "local-rag-llm-v1"
    _SEMANTIC_TERMS: dict[str, tuple[str, ...]] = {
        "Loans": ("refinance", "borrowing", "interest rate", "repayment holiday", "mortgage", "loan"),
        "Credit": ("card upgrade", "credit line", "spending limit", "credit card", "limit"),
        "Complaints": ("unhappy", "dissatisfied", "poor service", "grievance", "complaint", "incorrect charge"),
        "Treasury": ("fx", "forward", "rollover", "currency", "hedge", "spot deal", "treasury"),
    }

    def __init__(self) -> None:
        """Initialize an inspectable request ledger used only by POC tests."""

        self.calls: list[LLMRequest] = []

    def classify(self, email_text: str, policy_context: tuple[str, ...]) -> Prediction:
        """Infer CTD using expanded intent terms while retaining call evidence."""

        self.calls.append(LLMRequest(email_text=email_text, policy_context=policy_context))
        lowered = email_text.lower()
        scores = {
            department: sum(1 for term in terms if term in lowered)
            for department, terms in self._SEMANTIC_TERMS.items()
        }
        ranking = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        department, top_score = ranking[0]
        second_score = ranking[1][1]
        if top_score == 0:
            return Prediction(UNKNOWN_CTD, 0.45, self.MODEL_VERSION, "RAG found no supported intent evidence.")
        if top_score == second_score:
            return Prediction(UNKNOWN_CTD, 0.55, self.MODEL_VERSION, "RAG evidence remained ambiguous.")
        confidence = min(0.94, 0.68 + 0.08 * top_score + 0.03 * (top_score - second_score))
        return Prediction(
            CTD_BY_DEPARTMENT[department],
            round(confidence, 4),
            self.MODEL_VERSION,
            f"RAG matched {top_score} semantic {department} signals using {len(policy_context)} policy chunks.",
        )


@dataclass(frozen=True, slots=True)
class RAGResult:
    """Return the prediction, retrieval lineage, and any safety flags."""

    prediction: Prediction
    retrieved_document_ids: tuple[str, ...]
    risk_flags: tuple[str, ...]
    llm_invoked: bool


class RAGClassificationAgent:
    """Retrieve routing policy and invoke an LLM only through guardrails."""

    def __init__(
        self,
        vector_store: MockVectorStore,
        llm_client: LLMClient,
        guardrail: ComplianceGuardrailAgent,
        config: AgentConfig | None = None,
    ) -> None:
        """Inject policy retrieval, model, guardrail, and result-limit settings."""

        self.vector_store = vector_store
        self.llm_client = llm_client
        self.guardrail = guardrail
        self.config = config or AgentConfig()

    def classify(self, model_text: str) -> RAGResult:
        """Run pre-checks, retrieval, structured inference, and post-checks."""

        inbound = self.guardrail.inspect_inbound(model_text)
        self.guardrail.assert_pii_free(inbound.safe_text)
        if inbound.requires_human_review:
            return RAGResult(
                prediction=Prediction(
                    UNKNOWN_CTD,
                    0.0,
                    "compliance-guardrail-v1",
                    "LLM invocation blocked by a compliance guardrail.",
                ),
                retrieved_document_ids=(),
                risk_flags=inbound.risk_flags,
                llm_invoked=False,
            )

        retrieved = self.vector_store.search(inbound.safe_text, limit=self.config.rag_result_limit)
        context = tuple(result.document.text for result in retrieved)
        prediction = self.llm_client.classify(inbound.safe_text, context)

        output_text = " | ".join(
            (
                prediction.label.category,
                prediction.label.request_type,
                prediction.label.department,
                prediction.rationale,
            )
        )
        outbound = self.guardrail.inspect_outbound(output_text)
        self.guardrail.assert_pii_free(outbound.safe_text)
        if outbound.requires_human_review or prediction.label.department not in CTD_BY_DEPARTMENT:
            prediction = Prediction(
                UNKNOWN_CTD,
                0.0,
                "compliance-guardrail-v1",
                "Model output failed post-generation compliance validation.",
            )
        return RAGResult(
            prediction=prediction,
            retrieved_document_ids=tuple(result.document.document_id for result in retrieved),
            risk_flags=(*inbound.risk_flags, *outbound.risk_flags),
            llm_invoked=True,
        )


def seed_routing_policies(vector_store: MockVectorStore) -> None:
    """Populate the vector store with the POC's approved routing policy chunks."""

    policies = (
        PolicyDocument(
            "policy-loans",
            "Mortgage, borrowing, refinancing, EMI, repayment, and loan application requests route to Loans.",
            {"department": "Loans", "version": "1"},
        ),
        PolicyDocument(
            "policy-credit",
            "Credit card limits, card upgrades, credit lines, and card servicing requests route to Credit.",
            {"department": "Credit", "version": "1"},
        ),
        PolicyDocument(
            "policy-complaints",
            "Customer dissatisfaction, grievances, incorrect charges, and service complaints route to Complaints.",
            {"department": "Complaints", "version": "1"},
        ),
        PolicyDocument(
            "policy-treasury",
            "FX, currency forwards, hedge, spot deal, rollover, and treasury instructions route to Treasury.",
            {"department": "Treasury", "version": "1"},
        ),
    )
    for policy in policies:
        vector_store.add(policy)

