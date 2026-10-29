"""PII redaction, non-reversible entity extraction, and language detection."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import EmailMessage, ExtractedEntity, PreprocessingResult


@dataclass(frozen=True, slots=True)
class _PatternSpec:
    """Bind a PII entity name, safe replacement token, and compiled pattern."""

    entity_type: str
    replacement: str
    pattern: re.Pattern[str]


@dataclass(frozen=True, slots=True)
class _DetectedMatch:
    """Store an internal raw match only for the duration of one redaction call."""

    entity_type: str
    replacement: str
    value: str
    start: int
    end: int


class PIIRedactor:
    """Detect and tokenize common Indian retail-banking identifiers.

    The POC intentionally uses deterministic regular expressions instead of a
    Presidio service. Patterns are applied by priority and overlapping matches
    are discarded, preventing an account number from being partially treated
    as a phone number or Aadhaar identifier.
    """

    def __init__(self) -> None:
        """Compile priority-ordered PII patterns used by all safety boundaries."""

        self._patterns = (
            _PatternSpec(
                "EMAIL_ADDRESS",
                "[EMAIL_ADDRESS]",
                re.compile(
                    r"(?<![\w.+-])[\w.+-]+@[A-Z0-9-]+(?:\.[A-Z0-9-]+)+(?![\w-])",
                    re.IGNORECASE,
                ),
            ),
            _PatternSpec(
                "PAN",
                "[PAN]",
                re.compile(r"(?<![A-Z0-9])[A-Z]{5}[0-9]{4}[A-Z](?![A-Z0-9])", re.IGNORECASE),
            ),
            _PatternSpec(
                "CUSTOMER_ID",
                "[CUSTOMER_ID]",
                re.compile(r"(?<![A-Z0-9])CUST(?:OMER)?[- ]?[0-9]{5,12}(?![A-Z0-9])", re.IGNORECASE),
            ),
            _PatternSpec(
                "AADHAAR",
                "[AADHAAR]",
                re.compile(r"(?<!\d)[2-9]\d{3}[ -]\d{4}[ -]\d{4}(?!\d)"),
            ),
            _PatternSpec(
                "PHONE_NUMBER",
                "[PHONE_NUMBER]",
                re.compile(r"(?<!\d)(?:\+91[ -]?)?[6-9]\d{4}[ -]?\d{5}(?!\d)"),
            ),
            _PatternSpec(
                "ACCOUNT_NUMBER",
                "[ACCOUNT_NUMBER]",
                re.compile(r"(?<!\d)\d{10,18}(?!\d)"),
            ),
        )

    def _detect(self, text: str) -> tuple[_DetectedMatch, ...]:
        """Find prioritized, non-overlapping matches in source-text order."""

        occupied: list[tuple[int, int]] = []
        matches: list[_DetectedMatch] = []
        for spec in self._patterns:
            for match in spec.pattern.finditer(text):
                span = match.span()
                if any(span[0] < end and start < span[1] for start, end in occupied):
                    continue
                occupied.append(span)
                matches.append(
                    _DetectedMatch(
                        entity_type=spec.entity_type,
                        replacement=spec.replacement,
                        value=match.group(0),
                        start=span[0],
                        end=span[1],
                    )
                )
        return tuple(sorted(matches, key=lambda item: item.start))

    @staticmethod
    def _masked_value(value: str) -> str:
        """Create diagnostic metadata that cannot reconstruct the original PII."""

        alphanumeric = "".join(character for character in value if character.isalnum())
        suffix = alphanumeric[-2:] if len(alphanumeric) >= 2 else "**"
        return f"***{suffix}"

    def redact(self, text: str) -> str:
        """Replace all detected PII values with typed, model-safe tokens."""

        redacted = text
        for match in reversed(self._detect(text)):
            redacted = redacted[: match.start] + match.replacement + redacted[match.end :]
        return redacted

    def extract_entities(self, text: str) -> tuple[ExtractedEntity, ...]:
        """Return entity types and masked hints without retaining raw values."""

        return tuple(
            ExtractedEntity(
                entity_type=match.entity_type,
                masked_value=self._masked_value(match.value),
                start=match.start,
                end=match.end,
            )
            for match in self._detect(text)
        )

    def contains_pii(self, text: str) -> bool:
        """Report whether un-tokenized PII remains in a text value."""

        return bool(self._detect(text))


class LanguageDetector:
    """Provide a lightweight local language detector suitable for the POC."""

    _SPANISH_MARKERS = {"hola", "préstamo", "queja", "gracias", "cuenta"}

    def detect(self, text: str) -> str:
        """Return an ISO-like language code using deterministic script hints."""

        if re.search(r"[\u0900-\u097F]", text):
            return "hi"
        tokens = set(re.findall(r"[a-záéíóúñ]+", text.lower()))
        if tokens & self._SPANISH_MARKERS:
            return "es"
        return "en"


class EmailPreprocessor:
    """Convert raw email fields into the only representation models may use."""

    def __init__(
        self,
        redactor: PIIRedactor | None = None,
        language_detector: LanguageDetector | None = None,
    ) -> None:
        """Inject reusable safety services or create deterministic defaults."""

        self.redactor = redactor or PIIRedactor()
        self.language_detector = language_detector or LanguageDetector()

    def process(self, email: EmailMessage) -> PreprocessingResult:
        """Extract metadata first, then redact subject and body before return."""

        joined = f"{email.subject}\n{email.body}"
        entities = self.redactor.extract_entities(joined)
        return PreprocessingResult(
            redacted_subject=self.redactor.redact(email.subject),
            redacted_body=self.redactor.redact(email.body),
            entities=entities,
            language=self.language_detector.detect(joined),
        )

