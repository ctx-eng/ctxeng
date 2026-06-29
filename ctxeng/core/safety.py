from __future__ import annotations

import re
from dataclasses import dataclass, field

SUSPICIOUS_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?(previous|above|prior\s+)?\s*instructions?", re.IGNORECASE),
    re.compile(r"system\s*(prompt|instruction|message)", re.IGNORECASE),
    re.compile(r"you\s+are\s+(now|not\s+)?\s*(an?\s+)?(free|unbounded|ungoverned)", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?(previous|prior)\s+(instructions?|context|rules)", re.IGNORECASE),
    re.compile(r"act\s+as\s+if\s+you\s+are\s+not\s+(an?\s+)?(ai|llm|assistant)", re.IGNORECASE),
    re.compile(r"output\s+the\s+(above|previous)\s+prompt", re.IGNORECASE),
    re.compile(r"reveal\s+(your\s+)?(system\s+)?prompt", re.IGNORECASE),
    re.compile(r"<\|im_start\|>|<\|im_end\|>", re.IGNORECASE),
    re.compile(r"\[\/?INST\]", re.IGNORECASE),
    re.compile(r"\[\/?SYSTEM\]", re.IGNORECASE),
]

POISONING_PATTERNS: list[re.Pattern] = [
    re.compile(r"this\s+is\s+(an?\s+)?(important\s+)?instruction", re.IGNORECASE),
    re.compile(r"when\s+asked\s+about\s+\w+\s+(always|never)\s+(say|respond|answer)", re.IGNORECASE),
    re.compile(r"remember.*new\s+rules?", re.IGNORECASE),
    re.compile(r"overwrite\s+(your\s+)?(previous|default)\s+(behavior|knowledge)", re.IGNORECASE),
    re.compile(r"from\s+now\s+on.*(ignore|forget|disregard)", re.IGNORECASE),
]


@dataclass
class ValidationResult:
    passed: bool
    matched_patterns: list[str] = field(default_factory=list)
    reason: str = ""


class InputValidator:
    max_length: int = 10000

    def validate(self, text: str) -> ValidationResult:
        if not text.strip():
            return ValidationResult(passed=False, reason="empty input")

        if len(text) > self.max_length:
            return ValidationResult(
                passed=False,
                reason=f"input exceeds max length ({len(text)} > {self.max_length})",
            )

        matched: list[str] = []
        for pat in SUSPICIOUS_PATTERNS:
            if pat.search(text):
                matched.append(pat.pattern)

        if matched:
            return ValidationResult(
                passed=False,
                matched_patterns=matched,
                reason="suspicious pattern detected (potential prompt injection)",
            )

        return ValidationResult(passed=True)


class ContextPoisoningFilter:
    def check(self, text: str) -> ValidationResult:
        matched: list[str] = []
        for pat in POISONING_PATTERNS:
            if pat.search(text):
                matched.append(pat.pattern)

        if matched:
            return ValidationResult(
                passed=False,
                matched_patterns=matched,
                reason="context poisoning pattern detected",
            )

        return ValidationResult(passed=True)

    def filter_memories(self, items: list, text_attr: str = "text") -> list:
        clean: list = []
        for item in items:
            text = getattr(item, text_attr, str(item))
            result = self.check(text)
            if result.passed:
                clean.append(item)
        return clean
