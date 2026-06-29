from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable
from typing import Optional


def _split_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if s.strip()]


def _tf_scores(sentences: list[str]) -> list[float]:
    word_freq: Counter = Counter()
    for sent in sentences:
        word_freq.update(sent.lower().split())

    if not word_freq:
        return [0.0] * len(sentences)

    scores = []
    for i, sent in enumerate(sentences):
        words = sent.lower().split()
        if not words:
            scores.append(0.0)
            continue
        score = sum(math.log(word_freq[w] + 1) for w in words) / len(words)
        position_bonus = 1.0 / (i + 1)
        scores.append(score + position_bonus)
    return scores


class ContextSummarizer:
    def __init__(
        self,
        max_sentences: int = 3,
        keep_recent: int = 2,
        summarize_fn: Callable[[str], str] | None = None,
    ) -> None:
        self.max_sentences = max_sentences
        self.keep_recent = keep_recent
        self.summarize_fn = summarize_fn

    def summarize(self, text: str) -> str:
        if not text.strip():
            return ""

        if self.summarize_fn:
            return self.summarize_fn(text)

        sentences = _split_sentences(text)
        if len(sentences) <= self.max_sentences:
            return text

        scores = _tf_scores(sentences)
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )
        selected_indices = set(i for i, _ in ranked[: self.max_sentences])
        selected = [sentences[i] for i in sorted(selected_indices)]
        return " ".join(selected)

    def sliding_window(
        self,
        turns: list[str],
        window: Optional[int] = None,
    ) -> str:
        window = window or self.keep_recent
        if len(turns) <= window:
            return " ".join(turns)

        recent = turns[-window:]
        older = turns[:-window]
        older_text = " ".join(older)
        summary = self.summarize(older_text)
        parts = [f"[summary] {summary}"] if summary else []
        parts.extend(recent)
        return " ".join(parts)
