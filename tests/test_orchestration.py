"""Phase-four tests for LangGraph confidence edges and routing outcomes."""

from email_analytics_agent.classification import BaselineCTDClassifier
from email_analytics_agent.config import AgentConfig
from email_analytics_agent.guardrails import ComplianceGuardrailAgent
from email_analytics_agent.models import EmailMessage
from email_analytics_agent.orchestration import ManagerAgent
from email_analytics_agent.preprocessing import EmailPreprocessor, PIIRedactor
from email_analytics_agent.rag import LocalLLMClient, RAGClassificationAgent, seed_routing_policies
from email_analytics_agent.routing import MockQueueRouter, SummarizerRoutingAgent
from email_analytics_agent.storage import MockMongoStore, MockVectorStore


def _email(message_id: str, subject: str, body: str) -> EmailMessage:
    """Create a test email with a stable synthetic receipt timestamp."""

    return EmailMessage(message_id, "customer@example.test", subject, body, "2026-01-01T00:00:00Z")


def _build_manager() -> tuple[ManagerAgent, LocalLLMClient, MockQueueRouter, MockMongoStore]:
    """Construct isolated orchestration dependencies for one test."""

    config = AgentConfig()
    redactor = PIIRedactor()
    guardrail = ComplianceGuardrailAgent(redactor)
    vectors = MockVectorStore()
    seed_routing_policies(vectors)
    llm = LocalLLMClient()
    rag = RAGClassificationAgent(vectors, llm, guardrail, config)
    queues = MockQueueRouter()
    mongo = MockMongoStore()
    manager = ManagerAgent(
        EmailPreprocessor(redactor),
        BaselineCTDClassifier(),
        rag,
        guardrail,
        SummarizerRoutingAgent(config),
        queues,
        mongo,
        config,
    )
    return manager, llm, queues, mongo


def test_high_confidence_email_bypasses_llm_and_auto_routes() -> None:
    """Verify the high-confidence conditional edge never invokes RAG/LLM."""

    manager, llm, queues, mongo = _build_manager()
    result = manager.process_email(
        _email(
            "HIGH-1",
            "Home loan mortgage application",
            "Please review my loan EMI disbursement for account 991100221234.",
        )
    )

    assert result.decision == "auto_route"
    assert result.target_queue == "loans-operations"
    assert result.prediction.label.department == "Loans"
    assert result.decision_path == ("preprocess", "ml_classify", "summarize", "auto_route", "finalize")
    assert llm.calls == []
    assert queues.count("loans-operations") == 1
    assert "991100221234" not in queues.items("loans-operations")[0].summary
    assert mongo.get("HIGH-1")["classification"]["label"]["department"] == "Loans"


def test_low_confidence_email_triggers_rag_and_auto_routes() -> None:
    """Verify semantic fallback runs only after the baseline score is low."""

    manager, llm, queues, _ = _build_manager()
    result = manager.process_email(
        _email(
            "LOW-1",
            "Rollover request",
            "Please roll over our USD INR forward and preserve the currency hedge for 991100221234.",
        )
    )

    assert result.decision == "auto_route"
    assert result.target_queue == "treasury-operations"
    assert result.prediction.source == "local-rag-llm-v1"
    assert result.decision_path == (
        "preprocess",
        "ml_classify",
        "rag_classify",
        "summarize",
        "auto_route",
        "finalize",
    )
    assert len(llm.calls) == 1
    assert "991100221234" not in llm.calls[0].email_text
    assert queues.count("treasury-operations") == 1


def test_risky_email_routes_to_rm_review_without_llm() -> None:
    """Verify compliance risk overrides even a clear baseline CTD classification."""

    manager, llm, queues, _ = _build_manager()
    result = manager.process_email(
        _email(
            "RISK-1",
            "Unauthorized fraud on home loan",
            "This mortgage EMI appears to be an unauthorized fraud on account 991100221234.",
        )
    )

    assert result.decision == "human_review"
    assert result.target_queue == "rm-review"
    assert "suspected_fraud" in result.risk_flags
    assert result.decision_path == ("preprocess", "ml_classify", "rm_review", "finalize")
    assert llm.calls == []
    assert queues.count("rm-review") == 1


def test_ambiguous_rag_result_routes_to_rm_review() -> None:
    """Verify tied fallback evidence cannot be auto-routed."""

    manager, llm, queues, _ = _build_manager()
    result = manager.process_email(
        _email("AMB-1", "Need advice", "Should I refinance or request a card upgrade?")
    )

    assert result.decision == "human_review"
    assert result.prediction.label.department == "RM Review"
    assert result.decision_path == (
        "preprocess",
        "ml_classify",
        "rag_classify",
        "rm_review",
        "finalize",
    )
    assert len(llm.calls) == 1
    assert queues.count("rm-review") == 1

