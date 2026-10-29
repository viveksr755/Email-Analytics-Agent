"""Configuration values for the deterministic proof-of-concept workflow."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Control confidence thresholds and safety limits for one agent instance.

    Attributes:
        ml_confidence_threshold: Minimum baseline score that bypasses RAG.
        llm_confidence_threshold: Minimum safe RAG score for automatic routing.
        summary_word_limit: Maximum words included in a receiving-team summary.
        rag_result_limit: Maximum routing-policy chunks supplied to the local LLM.
    """

    ml_confidence_threshold: float = 0.80
    llm_confidence_threshold: float = 0.75
    summary_word_limit: int = 36
    rag_result_limit: int = 3

    def __post_init__(self) -> None:
        """Reject invalid thresholds early instead of failing during routing."""

        for field_name in ("ml_confidence_threshold", "llm_confidence_threshold"):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1")
        if self.summary_word_limit < 1 or self.rag_result_limit < 1:
            raise ValueError("summary_word_limit and rag_result_limit must be positive")

