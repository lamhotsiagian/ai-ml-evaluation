"""LLM-as-a-Judge: structured, position-debiased, self-consistent grading.

The judge is itself a model, so it is itself a system under test. Four defences
are built in because each corresponds to a documented, reproducible bias:

* **Structured output** -- the verdict is a validated Pydantic object, so a
  malformed grade is an error row rather than a silently mis-parsed 0.
* **Position swapping** -- pairwise judges prefer whichever candidate is shown
  first (Zheng et al., 2023). Every comparison is run in both orders.
* **Self-consistency** -- ``n_samples > 1`` at non-zero temperature turns a
  single noisy verdict into a majority vote plus an agreement number you can
  monitor.
* **Reference-anchored prompting** -- when a gold answer exists it is supplied,
  which cuts the judge's reliance on its own priors.
"""

from __future__ import annotations

import asyncio
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from evalcore.config import Settings, get_settings
from evalcore.judge.rubric import (
    Criterion,
    JudgeVerdict,
    PairwiseVerdict,
    Rubric,
)
from evalcore.llm import CacheKey, ResponseCache, build_chat_model, default_cache, render_messages

_SYSTEM_PROMPT = """You are a strict, calibrated evaluation judge for a production AI system.

Rules you must follow:
1. Grade against the rubric only. Do not reward fluency, length, or confident tone.
2. Quote verbatim evidence from the response for every criterion score. If you cannot
   quote evidence, the score for that criterion is at most 2.
3. Write the per-criterion reasoning BEFORE choosing the overall score.
4. Be willing to give 1s and 5s. A judge that never uses the ends of the scale
   carries almost no information.
5. Judge the response that is present. Do not speculate about what the system
   "probably meant"."""

_USER_TEMPLATE = """{rubric}

## Task given to the system
{task}
{reference_block}{context_block}
## Response under evaluation
{response}

Return your verdict using the required schema. Score every criterion listed in the rubric."""


@dataclass
class JudgeResult:
    """A verdict plus the reliability metadata needed to trust it."""

    verdict: JudgeVerdict
    weighted_score: float
    normalised_score: float
    n_samples: int = 1
    sample_agreement: float = 1.0
    sample_scores: list[float] = field(default_factory=list)
    model: str = ""
    cached: bool = False

    def as_row(self) -> dict[str, Any]:
        return {
            "overall_score": self.verdict.overall_score,
            "weighted_score": round(self.weighted_score, 3),
            "normalised": round(self.normalised_score, 4),
            "verdict": self.verdict.verdict,
            "confidence": self.verdict.confidence,
            "agreement": round(self.sample_agreement, 3),
            "failure_modes": ", ".join(self.verdict.failure_modes),
            "model": self.model,
        }


class RubricJudge:
    """Single-response judge driven by a versioned rubric."""

    def __init__(
        self,
        rubric: Rubric,
        *,
        model: str | None = None,
        temperature: float | None = None,
        n_samples: int = 1,
        settings: Settings | None = None,
        cache: ResponseCache | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.rubric = rubric
        self.n_samples = max(1, n_samples)
        # Self-consistency needs sampling variance; at temperature 0 the extra
        # calls are pure cost and always agree, which is a fake agreement number.
        resolved_temperature = (
            temperature if temperature is not None
            else (0.0 if self.n_samples == 1 else 0.7)
        )
        self.model_name = model or self.settings.judge_model
        self._llm = build_chat_model(
            role="judge", model=self.model_name, temperature=resolved_temperature,
            max_output_tokens=1400, settings=self.settings,
        ).with_structured_output(JudgeVerdict)
        self._cache = cache if cache is not None else default_cache(self.settings)
        self._temperature = resolved_temperature

    # -- prompt ------------------------------------------------------------
    def _build_messages(self, task: str, response: str, *, reference: str | None,
                        contexts: Sequence[str] | None) -> list[Any]:
        reference_block = f"\n## Reference answer (gold)\n{reference}\n" if reference else ""
        context_block = ""
        if contexts:
            joined = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
            context_block = f"\n## Context supplied to the system\n{joined}\n"
        user = _USER_TEMPLATE.format(
            rubric=self.rubric.render(),
            task=task,
            reference_block=reference_block,
            context_block=context_block,
            response=response,
        )
        return [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user)]

    # -- scoring -----------------------------------------------------------
    async def ajudge(
        self,
        task: str,
        response: str,
        *,
        reference: str | None = None,
        contexts: Sequence[str] | None = None,
    ) -> JudgeResult:
        """Grade one response, optionally with self-consistency sampling."""
        messages = self._build_messages(task, response, reference=reference, contexts=contexts)
        cache_key = CacheKey(
            model=self.model_name,
            temperature=self._temperature,
            prompt=render_messages(messages),
            extra=f"{self.rubric.name}:{self.rubric.version}:n={self.n_samples}",
        )
        cached_payload = self._cache.get(cache_key)
        if cached_payload:
            verdict = JudgeVerdict.model_validate_json(cached_payload)
            return self._finalise(verdict, [verdict.overall_score], cached=True)

        verdicts: list[JudgeVerdict] = await asyncio.gather(
            *(self._llm.ainvoke(messages) for _ in range(self.n_samples))
        )
        consensus = self._consensus(verdicts)
        self._cache.put(cache_key, consensus.model_dump_json())
        return self._finalise(consensus, [v.overall_score for v in verdicts])

    def judge(self, task: str, response: str, *, reference: str | None = None,
              contexts: Sequence[str] | None = None) -> JudgeResult:
        return asyncio.run(self.ajudge(task, response, reference=reference, contexts=contexts))

    def _consensus(self, verdicts: list[JudgeVerdict]) -> JudgeVerdict:
        """Median score, majority verdict, union of observed failure modes.

        Median rather than mean: a single outlier sample (the judge occasionally
        misreads a rubric) should not drag the consensus, and on an ordinal 1-5
        scale the median is the defensible central tendency anyway.
        """
        if len(verdicts) == 1:
            return verdicts[0]
        scores = [v.overall_score for v in verdicts]
        median_score = statistics.median(scores)
        chosen = min(verdicts, key=lambda v: abs(v.overall_score - median_score))
        majority = Counter(v.verdict for v in verdicts).most_common(1)[0][0]
        failure_modes = sorted({mode for v in verdicts for mode in v.failure_modes})
        return JudgeVerdict(
            criteria=chosen.criteria,
            overall_score=median_score,
            verdict=majority,  # type: ignore[arg-type]
            confidence=float(statistics.mean(v.confidence for v in verdicts)),
            failure_modes=failure_modes,
        )

    def _finalise(self, verdict: JudgeVerdict, samples: list[float], *, cached: bool = False) -> JudgeResult:
        agreement = 1.0
        if len(samples) > 1:
            modal = Counter(samples).most_common(1)[0][1]
            agreement = modal / len(samples)
        return JudgeResult(
            verdict=verdict,
            weighted_score=self.rubric.weighted_score(verdict.criteria),
            normalised_score=verdict.normalised(),
            n_samples=len(samples),
            sample_agreement=agreement,
            sample_scores=samples,
            model=self.model_name,
            cached=cached,
        )


class PairwiseJudge:
    """Head-to-head judge with mandatory position swapping.

    The swap is not optional. Running only one order gives a number that is
    partly a measurement of the judge's positional preference; running both and
    reporting disagreement as a tie gives a number that is a measurement of the
    responses. The disagreement rate is itself a useful diagnostic -- above
    roughly 20% the rubric is too vague to separate the candidates.
    """

    _SYSTEM = (
        "You compare two candidate responses to the same task. Decide which better "
        "satisfies the rubric. Ignore length and formatting unless the rubric mentions "
        "them; longer is not better. If they are genuinely equivalent, answer 'tie'."
    )

    def __init__(self, rubric: Rubric, *, model: str | None = None,
                 settings: Settings | None = None, cache: ResponseCache | None = None) -> None:
        self.settings = settings or get_settings()
        self.rubric = rubric
        self.model_name = model or self.settings.judge_model
        self._llm = build_chat_model(
            role="judge", model=self.model_name, temperature=0.0,
            max_output_tokens=900, settings=self.settings,
        ).with_structured_output(PairwiseVerdict)
        self._cache = cache if cache is not None else default_cache(self.settings)

    def _messages(self, task: str, first: str, second: str) -> list[Any]:
        user = (
            f"{self.rubric.render()}\n\n## Task\n{task}\n\n"
            f"## Response A\n{first}\n\n## Response B\n{second}\n\n"
            "Which response better satisfies the rubric?"
        )
        return [SystemMessage(content=self._SYSTEM), HumanMessage(content=user)]

    async def acompare(self, task: str, response_a: str, response_b: str) -> dict[str, Any]:
        """Compare A and B in both presentation orders.

        Returns a dict with the debiased winner, both raw verdicts, and a
        ``position_bias`` flag that is True when the two orders disagreed.
        """
        forward, reverse = await asyncio.gather(
            self._one(task, response_a, response_b),
            self._one(task, response_b, response_a),
        )
        # In the reverse pass, "A" refers to the original response B.
        reverse_winner = {"A": "B", "B": "A", "tie": "tie"}[reverse.winner]
        biased = forward.winner != reverse_winner
        winner = "tie" if biased else forward.winner
        return {
            "winner": winner,
            "position_bias": biased,
            "forward_winner": forward.winner,
            "reverse_winner": reverse_winner,
            "margin": forward.margin,
            "reasoning": forward.reasoning,
            "deciding_criterion": forward.deciding_criterion,
        }

    async def _one(self, task: str, first: str, second: str) -> PairwiseVerdict:
        messages = self._messages(task, first, second)
        key = CacheKey(self.model_name, 0.0, render_messages(messages),
                       f"pairwise:{self.rubric.name}:{self.rubric.version}")
        cached = self._cache.get(key)
        if cached:
            return PairwiseVerdict.model_validate_json(cached)
        verdict: PairwiseVerdict = await self._llm.ainvoke(messages)
        self._cache.put(key, verdict.model_dump_json())
        return verdict

    def compare(self, task: str, response_a: str, response_b: str) -> dict[str, Any]:
        return asyncio.run(self.acompare(task, response_a, response_b))


def bradley_terry_scores(
    comparisons: Sequence[tuple[str, str, str]], *, iterations: int = 200, prior: float = 1.0
) -> dict[str, float]:
    """Fit Bradley-Terry strengths from pairwise outcomes.

    Args:
        comparisons: ``(system_a, system_b, winner)`` triples where ``winner`` is
            one of the two system names or ``"tie"``.
        iterations: MM-algorithm iterations.
        prior: Pseudo-count added to every pair; keeps the fit finite when a
            system wins or loses every one of its comparisons, which happens
            constantly in small arenas.

    Returns:
        Normalised strengths summing to 1. Convert to an Elo-style rating with
        ``400 * log10(strength / reference)`` if a leaderboard needs one.
    """
    systems = sorted({name for a, b, _ in comparisons for name in (a, b)})
    if not systems:
        return {}
    index = {name: i for i, name in enumerate(systems)}
    n = len(systems)

    wins = [[prior] * n for _ in range(n)]
    for a, b, winner in comparisons:
        i, j = index[a], index[b]
        if winner == a:
            wins[i][j] += 1
        elif winner == b:
            wins[j][i] += 1
        else:  # a tie counts as half a win each -- the standard BT treatment
            wins[i][j] += 0.5
            wins[j][i] += 0.5

    strength = [1.0] * n
    for _ in range(iterations):
        updated = []
        for i in range(n):
            numerator = sum(wins[i][j] for j in range(n) if j != i)
            denominator = sum(
                (wins[i][j] + wins[j][i]) / (strength[i] + strength[j])
                for j in range(n) if j != i
            )
            updated.append(numerator / denominator if denominator > 0 else strength[i])
        total = sum(updated) or 1.0
        strength = [value / total * n for value in updated]

    total = sum(strength)
    return {name: strength[index[name]] / total for name in systems}


def default_criteria() -> list[Criterion]:
    """Criteria used by the UI when a user builds a rubric interactively."""
    return [
        Criterion(
            name="Correctness",
            question="Is the response factually correct with respect to the task and reference?",
            anchors={1: "Materially wrong.", 3: "Partially correct.", 5: "Fully correct."},
            weight=2.0,
        ),
        Criterion(
            name="Relevance",
            question="Does the response address what was actually asked?",
            anchors={1: "Off topic.", 3: "Partially on topic.", 5: "Directly on topic."},
        ),
    ]
