"""Compliance guardrails applied before and after every model interaction."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .preprocessing import PIIRedactor


@dataclass(frozen=True, slots=True)
class GuardrailResult:
    """Return safe text and explainable policy flags from a guardrail check."""

    safe_text: str
    risk_flags: tuple[str, ...]
    prompt_injection_detected: bool

    @property
    def requires_human_review(self) -> bool:
        """Escalate any injection attempt or defined high-risk banking signal."""

        return self.prompt_injection_detected or bool(self.risk_flags)


class ComplianceGuardrailAgent:
    """Enforce PII, prompt-injection, and banking-risk rules around models."""

    _INJECTION_PATTERNS = (
        re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions?", re.IGNORECASE),
        re.compile(r"(?:reveal|print|show)\s+(?:the\s+)?system\s+prompt", re.IGNORECASE),
        re.compile(r"(?:bypass|disable|override)\s+(?:the\s+)?(?:guardrails?|safety|policy)", re.IGNORECASE),
        re.compile(r"you\s+are\s+now\s+(?:an?|the)", re.IGNORECASE),
        re.compile(r"<\s*(?:script|system|assistant)\b", re.IGNORECASE),
    )
    _RISK_PATTERNS = {
        "suspected_fraud": re.compile(r"\b(?:fraud|scam|unauthori[sz]ed)\b", re.IGNORECASE),
        "legal_or_regulatory": re.compile(r"\b(?:legal action|ombudsman|regulator|court order)\b", re.IGNORECASE),
        "financial_crime": re.compile(r"\b(?:money laundering|sanctioned|terrorist financing)\b", re.IGNORECASE),
        "vulnerable_customer": re.compile(r"\b(?:deceased|bereavement|incapacitated)\b", re.IGNORECASE),
    }

    def __init__(self, redactor: PIIRedactor | None = None) -> None:
        """Use one canonical redactor for both pre- and post-model controls."""

        self.redactor = redactor or PIIRedactor()

    def inspect_inbound(self, text: str) -> GuardrailResult:
        """Re-redact model input, detect injections, and identify risky cases."""

        safe_text = self.redactor.redact(text)
        injection = any(pattern.search(safe_text) for pattern in self._INJECTION_PATTERNS)
        risk_flags = tuple(
            name for name, pattern in self._RISK_PATTERNS.items() if pattern.search(safe_text)
        )
        if injection:
            risk_flags = (*risk_flags, "prompt_injection")
        return GuardrailResult(
            safe_text=safe_text,
            risk_flags=risk_flags,
            prompt_injection_detected=injection,
        )

    def inspect_outbound(self, text: str) -> GuardrailResult:
        """Redact unexpected output PII and reject instruction-like content."""

        safe_text = self.redactor.redact(text)
        injection = any(pattern.search(safe_text) for pattern in self._INJECTION_PATTERNS)
        flags = ("unsafe_model_output",) if injection else ()
        return GuardrailResult(
            safe_text=safe_text,
            risk_flags=flags,
            prompt_injection_detected=injection,
        )

    def assert_pii_free(self, text: str) -> None:
        """Raise a clear boundary error if raw PII would reach a model."""

        if self.redactor.contains_pii(text):
            raise ValueError("raw PII detected at protected model boundary")

