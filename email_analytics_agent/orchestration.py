"""LangGraph manager agent for safe CTD classification and case routing."""

from __future__ import annotations

from typing import TypedDict
from uuid import uuid4

from langgraph.graph import END, StateGraph

from .audit import ImmutableAuditLog
from .classification import BaselineCTDClassifier
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
from .storage import MockMongoStore


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
    result: ProcessResult


class ManagerAgent:
    """Coordinate classifiers, guardrails, RAG, summarization, and routing.

    The graph gives safety decisions explicit edges: high-confidence baseline
    results bypass RAG, low-confidence results enter RAG, and risky or still
    ambiguous cases terminate in the RM review queue.
    """

    def __init__(
        self,
        preprocessor: EmailPreprocessor,
        baseline_classifier: BaselineCTDClassifier,
        rag_agent: RAGClassificationAgent,
        guardrail: ComplianceGuardrailAgent,
        summarizer: SummarizerRoutingAgent,
        queues: MockQueueRouter,
        document_store: MockMongoStore,
        config: AgentConfig | None = None,
        audit_log: ImmutableAuditLog | None = None,
    ) -> None:
        """Inject all layer adapters and compile the LangGraph workflow."""

        self.preprocessor = preprocessor
        self.baseline_classifier = baseline_classifier
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
        workflow.add_edge("preprocess", "ml_classify")
        workflow.add_conditional_edges(
            "ml_classify",
            self._after_baseline,
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

    def _ml_classify_node(self, state: WorkflowState) -> WorkflowState:
        """Run the inexpensive baseline model on PII-safe text."""

        prediction = self.baseline_classifier.predict(state["preprocessed"].model_text)
        return {
            "prediction": prediction,
            "confidence_scores": {"baseline": prediction.confidence},
            "decision_path": self._append_path(state, "ml_classify"),
        }

    def _after_baseline(self, state: WorkflowState) -> str:
        """Choose review, direct route, or RAG from risk and ML confidence."""

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
        )
        self.queues.enqueue(item)
        return {
            "decision": "auto_route",
            "target_queue": target_queue,
            "decision_path": self._append_path(state, "auto_route"),
        }

    def _rm_review_node(self, state: WorkflowState) -> WorkflowState:
        """Create a safe summary and enqueue unresolved or risky cases for an RM."""

        summary = state.get("summary") or self.summarizer.summarize(
            state["preprocessed"], state["prediction"]
        )
        output_check = self.guardrail.inspect_outbound(summary)
        self.guardrail.assert_pii_free(output_check.safe_text)
        flags = (*state.get("risk_flags", ()), *output_check.risk_flags)
        item = QueueItem(
            run_id=state["run_id"],
            message_id=state["email"].message_id,
            target_queue="rm-review",
            summary=output_check.safe_text,
            prediction=state["prediction"],
            risk_flags=flags,
        )
        self.queues.enqueue(item)
        return {
            "decision": "human_review",
            "target_queue": "rm-review",
            "summary": output_check.safe_text,
            "risk_flags": flags,
            "decision_path": self._append_path(state, "rm_review"),
        }

    def _finalize_node(self, state: WorkflowState) -> WorkflowState:
        """Persist the CTD result and construct the immutable public outcome."""

        self.document_store.save_classification(state["email"].message_id, state["prediction"])
        path = self._append_path(state, "finalize")
        result = ProcessResult(
            run_id=state["run_id"],
            message_id=state["email"].message_id,
            decision=state["decision"],
            prediction=state["prediction"],
            target_queue=state["target_queue"],
            summary=state["summary"],
            decision_path=path,
            risk_flags=state.get("risk_flags", ()),
            confidence_scores=state.get("confidence_scores", {}),
        )
        self.audit_log.record_decision(result)
        return {"decision_path": path, "result": result}

    def process_email(self, email: EmailMessage) -> ProcessResult:
        """Persist one raw email, execute the graph, and return its final result."""

        self.document_store.save_email(email)
        final_state = self.graph.invoke(
            {
                "run_id": str(uuid4()),
                "email": email,
                "risk_flags": (),
                "decision_path": (),
                "confidence_scores": {},
            }
        )
        return final_state["result"]

