from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable


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
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        selected_indices = set(i for i, _ in ranked[: self.max_sentences])
        selected = [sentences[i] for i in sorted(selected_indices)]
        return " ".join(selected)

    def sliding_window(
        self,
        turns: list[str],
        window: int | None = None,
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


class MapReduceSummarizer:
    def __init__(
        self,
        chunk_size: int = 5,
        summarizer: ContextSummarizer | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.summarizer = summarizer or ContextSummarizer()

    def summarize(self, texts: list[str]) -> str:
        if not texts:
            return ""

        chunks = [texts[i : i + self.chunk_size] for i in range(0, len(texts), self.chunk_size)]

        map_results: list[str] = []
        for chunk in chunks:
            combined = " ".join(chunk)
            summary = self.summarizer.summarize(combined)
            if summary:
                map_results.append(summary)

        if not map_results:
            return ""

        if len(map_results) == 1:
            return map_results[0]

        combined_summaries = " ".join(map_results)
        return self.summarizer.summarize(combined_summaries)


class LLMCompressor:
    def __init__(
        self,
        compress_fn: Callable[[str], str] | None = None,
        chunk_size: int = 2000,
    ) -> None:
        self.compress_fn = compress_fn
        self.chunk_size = chunk_size

    def compress(self, text: str) -> str:
        if not text.strip():
            return ""

        if self.compress_fn:
            return self.compress_fn(text)

        words = text.split()
        if len(words) <= self.chunk_size:
            return text

        chunks = [
            " ".join(words[i : i + self.chunk_size]) for i in range(0, len(words), self.chunk_size)
        ]

        map_results: list[str] = []
        for chunk in chunks:
            if self.compress_fn:
                map_results.append(self.compress_fn(chunk))
            else:
                words_chunk = chunk.split()
                map_results.append(" ".join(words_chunk[: len(words_chunk) // 2]))

        return " ".join(map_results)
