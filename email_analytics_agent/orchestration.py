"""LangGraph manager agent for safe CTD classification and case routing."""

from __future__ import annotations

from dataclasses import replace
from typing import TypedDict
from uuid import uuid4

from langgraph.graph import END, StateGraph

from .audit import ImmutableAuditLog
from .classification import CTDClassifier
from .config import AgentConfig
from .guardrails import ComplianceGuardrailAgent, GuardrailResult
from .models import (
    EmailMessage,
    Prediction,
    PreprocessingResult,
    ProcessResult,
    QueueItem,
)
from .preprocessing import EmailPreprocessor
from .rag import RAGClassificationAgent, RAGResult
from .routing import MockQueueRouter, SummarizerRoutingAgent
from .storage import ClaimAction, EmailDocumentStore
from .taxonomy import UNKNOWN_CTD


class WorkflowState(TypedDict, total=False):
    """Define data passed between LangGraph nodes for one email run."""

    run_id: str
    email: EmailMessage
    preprocessed: PreprocessingResult
    inbound_guardrail: GuardrailResult
    prediction: Prediction
    rag_result: RAGResult
    risk_flags: tuple[str, ...]
    summary: str
    target_queue: str
    decision: str
    decision_path: tuple[str, ...]
    confidence_scores: dict[str, float]
    downstream_id: str
    storage_key: str
    result: ProcessResult


class ManagerAgent:
    """Coordinate classifiers, guardrails, RAG, summarization, and routing.

    The graph gives safety decisions explicit edges: high-confidence trained-ML
    results bypass RAG, low-confidence results enter RAG, and risky or still
    ambiguous cases terminate in the RM review queue.
    """

    def __init__(
        self,
        preprocessor: EmailPreprocessor,
        ml_classifier: CTDClassifier,
        rag_agent: RAGClassificationAgent,
        guardrail: ComplianceGuardrailAgent,
        summarizer: SummarizerRoutingAgent,
        queues: MockQueueRouter,
        document_store: EmailDocumentStore,
        config: AgentConfig | None = None,
        audit_log: ImmutableAuditLog | None = None,
    ) -> None:
        """Inject all layer adapters and compile the LangGraph workflow."""

        self.preprocessor = preprocessor
        self.ml_classifier = ml_classifier
        self.rag_agent = rag_agent
        self.guardrail = guardrail
        self.summarizer = summarizer
        self.queues = queues
        self.document_store = document_store
        self.config = config or AgentConfig()
        self.audit_log = audit_log or ImmutableAuditLog()
        self.graph = self._build_graph()

    @staticmethod
    def _append_path(state: WorkflowState, node_name: str) -> tuple[str, ...]:
        """Create an immutable decision path extended with one graph node."""

        return (*state.get("decision_path", ()), node_name)

    def _build_graph(self):
        """Construct and compile the confidence-gated LangGraph state machine."""

        workflow = StateGraph(WorkflowState)
        workflow.add_node("preprocess", self._preprocess_node)
        workflow.add_node("ml_classify", self._ml_classify_node)
        workflow.add_node("rag_classify", self._rag_classify_node)
        workflow.add_node("summarize", self._summarize_node)
        workflow.add_node("auto_route", self._auto_route_node)
        workflow.add_node("rm_review", self._rm_review_node)
        workflow.add_node("finalize", self._finalize_node)

        workflow.set_entry_point("preprocess")
        workflow.add_conditional_edges(
            "preprocess",
            self._after_preprocess,
            {"ml_classify": "ml_classify", "rm_review": "rm_review"},
        )
        workflow.add_conditional_edges(
            "ml_classify",
            self._after_ml,
            {"rag_classify": "rag_classify", "summarize": "summarize", "rm_review": "rm_review"},
        )
        workflow.add_conditional_edges(
            "rag_classify",
            self._after_rag,
            {"summarize": "summarize", "rm_review": "rm_review"},
        )
        workflow.add_edge("summarize", "auto_route")
        workflow.add_edge("auto_route", "finalize")
        workflow.add_edge("rm_review", "finalize")
        workflow.add_edge("finalize", END)
        return workflow.compile()

    def _preprocess_node(self, state: WorkflowState) -> WorkflowState:
        """Redact raw email fields and perform the inbound compliance scan."""

        preprocessed = self.preprocessor.process(state["email"])
        inspection = self.guardrail.inspect_inbound(preprocessed.model_text)
        self.guardrail.assert_pii_free(inspection.safe_text)
        return {
            "preprocessed": preprocessed,
            "inbound_guardrail": inspection,
            "risk_flags": inspection.risk_flags,
            "decision_path": self._append_path(state, "preprocess"),
        }

    @staticmethod
    def _after_preprocess(state: WorkflowState) -> str:
        """Bypass every model when inbound controls require human review."""

        inspection = state["inbound_guardrail"]
        return "rm_review" if inspection.requires_human_review else "ml_classify"

    def _ml_classify_node(self, state: WorkflowState) -> WorkflowState:
        """Run the persisted scikit-learn classifier on PII-safe text."""

        prediction = self.ml_classifier.predict(state["preprocessed"].model_text)
        return {
            "prediction": prediction,
            "confidence_scores": {"ml": prediction.confidence},
            "decision_path": self._append_path(state, "ml_classify"),
        }

    def _after_ml(self, state: WorkflowState) -> str:
        """Choose review, direct route, or RAG from risk and trained-ML confidence."""

        if state.get("risk_flags"):
            return "rm_review"
        prediction = state["prediction"]
        if (
            prediction.confidence >= self.config.ml_confidence_threshold
            and prediction.label.department != "RM Review"
        ):
            return "summarize"
        return "rag_classify"

    def _rag_classify_node(self, state: WorkflowState) -> WorkflowState:
        """Invoke retrieval-augmented classification through both guardrails."""

        rag_result = self.rag_agent.classify(state["preprocessed"].model_text)
        return {
            "rag_result": rag_result,
            "prediction": rag_result.prediction,
            "confidence_scores": {
                **state.get("confidence_scores", {}),
                "rag": rag_result.prediction.confidence,
            },
            "risk_flags": (*state.get("risk_flags", ()), *rag_result.risk_flags),
            "decision_path": self._append_path(state, "rag_classify"),
        }

    def _after_rag(self, state: WorkflowState) -> str:
        """Route only compliant, resolved RAG results above the LLM threshold."""

        result = state["rag_result"]
        if state.get("risk_flags") or not result.llm_invoked:
            return "rm_review"
        if (
            result.prediction.confidence >= self.config.llm_confidence_threshold
            and result.prediction.label.department != "RM Review"
        ):
            return "summarize"
        return "rm_review"

    def _summarize_node(self, state: WorkflowState) -> WorkflowState:
        """Generate and post-check the redacted receiving-team case summary."""

        summary = self.summarizer.summarize(state["preprocessed"], state["prediction"])
        output_check = self.guardrail.inspect_outbound(summary)
        self.guardrail.assert_pii_free(output_check.safe_text)
        return {
            "summary": output_check.safe_text,
            "risk_flags": (*state.get("risk_flags", ()), *output_check.risk_flags),
            "decision_path": self._append_path(state, "summarize"),
        }

    def _auto_route_node(self, state: WorkflowState) -> WorkflowState:
        """Enqueue a safe case in its CTD destination after successful checks."""

        target_queue = self.summarizer.target_queue(state["prediction"])
        if target_queue == "rm-review" or state.get("risk_flags"):
            return self._rm_review_node(state)
        item = QueueItem(
            run_id=state["run_id"],
            message_id=state["email"].message_id,
            target_queue=target_queue,
            summary=state["summary"],
            prediction=state["prediction"],
            risk_flags=state.get("risk_flags", ()),
            source_idempotency_key=str(
                state["email"].metadata.get("idempotency_key")
                or state["email"].message_id
            ),
        )
        downstream_id = self.queues.enqueue(item)
        return {
            "decision": "auto_route",
            "target_queue": target_queue,
            "downstream_id": downstream_id,
            "decision_path": self._append_path(state, "auto_route"),
        }

    def _rm_review_node(self, state: WorkflowState) -> WorkflowState:
        """Create a safe summary and enqueue unresolved or risky cases for an RM."""

        prediction = state.get("prediction") or Prediction(
            UNKNOWN_CTD,
            0.0,
            "compliance-guardrail-v1",
            "Classification bypassed because inbound guardrails require human review.",
        )
        summary = state.get("summary") or self.summarizer.summarize(
            state["preprocessed"], prediction
        )
        output_check = self.guardrail.inspect_outbound(summary)
        self.guardrail.assert_pii_free(output_check.safe_text)
        flags = (*state.get("risk_flags", ()), *output_check.risk_flags)
        item = QueueItem(
            run_id=state["run_id"],
            message_id=state["email"].message_id,
            target_queue="rm-review",
            summary=output_check.safe_text,
            prediction=prediction,
            risk_flags=flags,
            source_idempotency_key=str(
                state["email"].metadata.get("idempotency_key")
                or state["email"].message_id
            ),
        )
        downstream_id = self.queues.enqueue(item)
        return {
            "decision": "human_review",
            "target_queue": "rm-review",
            "summary": output_check.safe_text,
            "prediction": prediction,
            "downstream_id": downstream_id,
            "risk_flags": flags,
            "decision_path": self._append_path(state, "rm_review"),
        }

    def _finalize_node(self, state: WorkflowState) -> WorkflowState:
        """Persist the CTD result and construct the immutable public outcome."""

        path = self._append_path(state, "finalize")
        result = ProcessResult(
            run_id=state["run_id"],
            message_id=state["email"].message_id,
            decision=state["decision"],
            prediction=state["prediction"],
            target_queue=state["target_queue"],
            summary=state["summary"],
            decision_path=path,
            downstream_id=state["downstream_id"],
            storage_key=state["storage_key"],
            risk_flags=state.get("risk_flags", ()),
            confidence_scores=state.get("confidence_scores", {}),
        )
        self.document_store.save_outcome(result)
        try:
            self.audit_log.record_decision(result)
            self.document_store.mark_audited(result.storage_key, result.run_id)
        except Exception as error:
            try:
                self.document_store.mark_audit_failed(
                    result.storage_key,
                    result.run_id,
                    type(error).__name__,
                )
            except Exception:
                pass
            raise
        return {"decision_path": path, "result": result}

    def process_email(self, email: EmailMessage) -> ProcessResult:
        """Claim one email, skip completed work, or execute its saved workflow."""

        claim = self.document_store.claim_email(email, str(uuid4()))
        if claim.action is ClaimAction.SKIP:
            outcome = self.document_store.load_outcome(claim.idempotency_key)
            if outcome is None:
                raise RuntimeError("completed email has no persisted outcome")
            return replace(outcome, was_skipped=True)
        if claim.action is ClaimAction.AUDIT_ONLY:
            outcome = self.document_store.load_outcome(claim.idempotency_key)
            if outcome is None:
                raise RuntimeError("audit-only email has no persisted outcome")
            return self._repair_audit(outcome)
        if claim.action is ClaimAction.IN_PROGRESS:
            raise RuntimeError(f"email is already being processed: {email.message_id}")
        if claim.action not in (ClaimAction.NEW, ClaimAction.RESUME):
            raise RuntimeError(f"unsupported email claim action: {claim.action}")
        return self.process_saved_email(email, claim.run_id, claim.idempotency_key)

    def process_saved_email(
        self,
        email: EmailMessage,
        run_id: str,
        storage_key: str | None = None,
    ) -> ProcessResult:
        """Execute LangGraph for an email already claimed and stored by ingestion."""

        resolved_storage_key = storage_key or str(
            email.metadata.get("idempotency_key") or email.message_id
        )
        self.document_store.mark_processing(resolved_storage_key, run_id)
        try:
            final_state = self.graph.invoke(
                {
                    "run_id": run_id,
                    "email": email,
                    "storage_key": resolved_storage_key,
                    "risk_flags": (),
                    "decision_path": (),
                    "confidence_scores": {},
                }
            )
            return final_state["result"]
        except Exception as error:
            if self.document_store.load_outcome(resolved_storage_key) is None:
                try:
                    self.document_store.mark_retry_pending(
                        resolved_storage_key,
                        run_id,
                        "workflow",
                        type(error).__name__,
                    )
                except Exception:
                    pass
            raise

    def _repair_audit(self, outcome: ProcessResult) -> ProcessResult:
        """Retry only the idempotent audit append for a persisted outcome."""

        try:
            self.audit_log.record_decision(outcome)
            self.document_store.mark_audited(
                outcome.storage_key or outcome.message_id,
                outcome.run_id,
            )
        except Exception as error:
            try:
                self.document_store.mark_audit_failed(
                    outcome.storage_key or outcome.message_id,
                    outcome.run_id,
                    type(error).__name__,
                )
            except Exception:
                pass
            raise
        return replace(outcome, was_skipped=True)

