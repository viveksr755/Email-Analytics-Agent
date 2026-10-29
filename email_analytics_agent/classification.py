"""Explainable local baseline classifier for common banking email intents."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .models import Prediction
from .taxonomy import CTD_BY_DEPARTMENT, UNKNOWN_CTD


@dataclass(frozen=True, slots=True)
class KeywordRule:
    """Associate a case-insensitive phrase with an integer evidence weight."""

    phrase: str
    weight: int = 1


class BaselineCTDClassifier:
    """Return deterministic CTD predictions and calibrated mock confidence.

    This intentionally small model stands in for a trained scikit-learn or
    transformer classifier. Its confidence behavior, model version, and
    decision evidence remain deterministic so orchestration tests do not rely
    on a remote model or mutable training artifact.
    """

    MODEL_VERSION = "baseline-keyword-v1"
    _RULES: dict[str, tuple[KeywordRule, ...]] = {
        "Loans": (
            KeywordRule("home loan", 2),
            KeywordRule("loan"),
            KeywordRule("mortgage", 2),
            KeywordRule("emi"),
            KeywordRule("foreclosure", 2),
            KeywordRule("disbursement"),
        ),
        "Credit": (
            KeywordRule("credit card", 2),
            KeywordRule("card limit", 2),
            KeywordRule("limit increase"),
            KeywordRule("credit request"),
            KeywordRule("cash limit"),
        ),
        "Complaints": (
            KeywordRule("complaint", 2),
            KeywordRule("grievance", 2),
            KeywordRule("failed transfer"),
            KeywordRule("incorrect fee"),
            KeywordRule("service issue"),
        ),
        "Treasury": (
            KeywordRule("treasury", 2),
            KeywordRule("foreign exchange", 2),
            KeywordRule("settlement"),
            KeywordRule("remittance"),
            KeywordRule("trade"),
        ),
    }

    @classmethod
    def _score(cls, text: str) -> Counter[str]:
        """Count weighted phrase occurrences for every supported department."""

        lowered = text.lower()
        return Counter(
            {
                department: sum(lowered.count(rule.phrase) * rule.weight for rule in rules)
                for department, rules in cls._RULES.items()
            }
        )

    def predict(self, redacted_text: str) -> Prediction:
        """Classify PII-safe text and return confidence suitable for gating."""

        scores = self._score(redacted_text)
        ranking = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        top_department, top_score = ranking[0]
        second_score = ranking[1][1]

        if top_score == 0:
            return Prediction(
                UNKNOWN_CTD,
                0.25,
                self.MODEL_VERSION,
                "No baseline taxonomy terms matched.",
            )
        if top_score == second_score:
            return Prediction(
                UNKNOWN_CTD,
                0.45,
                self.MODEL_VERSION,
                "Evidence was tied across multiple CTD departments.",
            )

        confidence = min(0.96, 0.48 + 0.08 * top_score + 0.04 * (top_score - second_score))
        return Prediction(
            CTD_BY_DEPARTMENT[top_department],
            round(confidence, 4),
            self.MODEL_VERSION,
            f"Matched {top_score} weighted {top_department} taxonomy points; runner-up scored {second_score}.",
        )

