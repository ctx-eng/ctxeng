from __future__ import annotations

from dataclasses import dataclass, field

from ctxeng.llm.base import LLMProvider


@dataclass
class CorrectnessScore:
    score: float
    explanation: str = ""
    errors: list[str] = field(default_factory=list)


class CorrectnessEvaluator:
    def __init__(self, provider: LLMProvider | None = None) -> None:
        self._provider = provider

    def evaluate(
        self,
        question: str,
        answer: str,
        reference: str | None = None,
    ) -> CorrectnessScore:
        if self._provider is not None:
            return self._llm_judge(question, answer, reference)
        return self._embedding_fallback(question, answer, reference)

    def _llm_judge(
        self,
        question: str,
        answer: str,
        reference: str | None = None,
    ) -> CorrectnessScore:
        ref_text = reference or "No reference provided"
        prompt = (
            f"Evaluate the following answer on a scale of 0.0 to 1.0.\n"
            f"Return only a number between 0 and 1 and a brief explanation.\n\n"
            f"Question: {question}\n"
            f"Reference answer: {ref_text}\n"
            f"Answer to evaluate: {answer}\n\n"
            f"Scoring:\n"
            f"- 1.0 = perfectly correct and complete\n"
            f"- 0.7-0.9 = mostly correct, minor omissions\n"
            f"- 0.4-0.6 = partially correct, significant gaps\n"
            f"- 0.1-0.3 = mostly incorrect\n"
            f"- 0.0 = completely wrong or refuses to answer\n\n"
            f"Score:"
        )
        try:
            from ctxeng.llm.base import LLMMessage

            resp = self._provider.generate(
                [
                    LLMMessage(
                        role="system", content="You are a strict but fair answer evaluator."
                    ),
                    LLMMessage(role="user", content=prompt),
                ]
            )
            text = resp.content.strip()
            import re

            match = re.search(r"([01]\.\d+|1\.0|0\.0)", text)
            score = float(match.group(1)) if match else 0.5
            score = max(0.0, min(1.0, score))
            explanation = text[:200] if text else ""
            return CorrectnessScore(score=score, explanation=explanation)
        except Exception as e:
            return CorrectnessScore(score=0.0, explanation=f"LLM judge error: {e}")

    def _embedding_fallback(
        self,
        question: str,
        answer: str,
        reference: str | None = None,
    ) -> CorrectnessScore:
        if not reference:
            return CorrectnessScore(score=0.0, explanation="No reference answer for comparison")
        common = set(answer.lower().split()) & set(reference.lower().split())
        union = set(answer.lower().split()) | set(reference.lower().split())
        if not union:
            return CorrectnessScore(score=0.0, explanation="Empty answer or reference")
        score = len(common) / len(union)
        return CorrectnessScore(
            score=round(score, 4),
            explanation=f"Token overlap: {len(common)}/{len(union)}",
        )
