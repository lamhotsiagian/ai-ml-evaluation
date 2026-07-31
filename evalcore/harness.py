"""A framework-agnostic evaluation harness: metrics, suites, assertions, CI.

Chapter 7 compares OpenAI Evals, DeepEval, Promptfoo, LangSmith, Phoenix, Inspect
and the rest. They differ in surface area but share one core abstraction, which
is implemented here: a **metric** is a pure function from a case and an output to
a score; a **suite** binds metrics to a dataset and a target; an **assertion**
turns aggregate scores into a build decision.

Owning this abstraction is what makes framework choice reversible. Teams that
write their evaluation logic directly against a vendor's decorator API discover
the migration cost the first time they need a feature the vendor lacks.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence

from evalcore.datasets import EvalCase, GoldenDataset
from evalcore.runner.runner import CaseResult, EvaluationRunner, RunResult
from evalcore.stats import Interval, bootstrap_interval, wilson_interval

TargetFn = Callable[[EvalCase], Awaitable[str]] | Callable[[EvalCase], str]
MetricFn = Callable[[EvalCase, str], float | Awaitable[float]]


@dataclass
class Metric:
    """A named scorer with a direction and an optional per-item threshold."""

    name: str
    fn: MetricFn
    higher_is_better: bool = True
    threshold: float | None = None
    binary: bool = False
    description: str = ""

    async def score(self, case: EvalCase, output: str) -> float:
        result = self.fn(case, output)
        if inspect.isawaitable(result):
            result = await result
        return float(result)

    def passed(self, value: float) -> bool:
        if self.threshold is None:
            return True
        return value >= self.threshold if self.higher_is_better else value <= self.threshold


@dataclass
class Assertion:
    """A build-blocking condition on an aggregate metric."""

    metric: str
    minimum: float | None = None
    maximum: float | None = None
    max_regression: float | None = None
    require_ci_clear: bool = False
    description: str = ""

    def check(self, value: float, *, baseline: float | None = None,
              interval: Interval | None = None) -> tuple[bool, str]:
        """Evaluate the assertion, returning (passed, human-readable reason).

        ``require_ci_clear`` is the difference between a gate that blocks real
        regressions and one that blocks noise: it demands the *confidence
        interval* clear the floor, not just the point estimate. On a 100-case
        suite the two differ constantly.
        """
        if self.minimum is not None:
            reference = interval.low if (self.require_ci_clear and interval) else value
            if reference < self.minimum:
                return False, (f"{self.metric}={value:.4f}"
                               + (f" (CI low {interval.low:.4f})" if interval and self.require_ci_clear else "")
                               + f" below minimum {self.minimum:.4f}")
        if self.maximum is not None and value > self.maximum:
            return False, f"{self.metric}={value:.4f} above maximum {self.maximum:.4f}"
        if self.max_regression is not None and baseline is not None:
            drop = baseline - value
            if drop > self.max_regression:
                return False, (f"{self.metric} regressed {drop:.4f} from baseline {baseline:.4f}; "
                               f"budget is {self.max_regression:.4f}")
        return True, f"{self.metric}={value:.4f} OK"


@dataclass
class SuiteReport:
    """Aggregated result of one suite execution."""

    suite: str
    run: RunResult
    aggregates: dict[str, Interval] = field(default_factory=dict)
    per_metric_pass_rate: dict[str, float] = field(default_factory=dict)
    assertion_results: list[tuple[str, bool, str]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(ok for _, ok, _ in self.assertion_results)

    def failures(self) -> list[str]:
        return [reason for _, ok, reason in self.assertion_results if not ok]

    def as_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "metric": name,
                "value": round(interval.estimate, 4),
                "ci_low": round(interval.low, 4),
                "ci_high": round(interval.high, 4),
                "n": interval.n,
                "pass_rate": round(self.per_metric_pass_rate.get(name, float("nan")), 4),
            }
            for name, interval in sorted(self.aggregates.items())
        ]

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        parts = [f"{name}={interval.estimate:.4f}" for name, interval in sorted(self.aggregates.items())]
        return f"[{status}] {self.suite}: " + ", ".join(parts)


class EvaluationSuite:
    """Bind a dataset, a target system, metrics and assertions into one runnable unit."""

    def __init__(
        self,
        name: str,
        dataset: GoldenDataset,
        target: TargetFn,
        metrics: Sequence[Metric],
        *,
        assertions: Sequence[Assertion] = (),
        runner: EvaluationRunner | None = None,
    ) -> None:
        if not metrics:
            raise ValueError("a suite needs at least one metric")
        self.name = name
        self.dataset = dataset
        self.target = target
        self.metrics = list(metrics)
        self.assertions = list(assertions)
        self.runner = runner or EvaluationRunner()

    async def _evaluate_case(self, case: EvalCase) -> dict[str, Any]:
        result = self.target(case)
        output = await result if inspect.isawaitable(result) else result
        output = str(output)
        scores = {metric.name: await metric.score(case, output) for metric in self.metrics}
        return {"scores": scores, "output": output, "metadata": {"difficulty": case.difficulty}}

    def run(
        self,
        *,
        baseline: dict[str, float] | None = None,
        progress: Callable[[int, int, CaseResult], None] | None = None,
    ) -> SuiteReport:
        run = self.runner.run(
            self.name, self.dataset.cases, self._evaluate_case,
            dataset_hash=self.dataset.content_hash(), progress=progress,
        )
        aggregates: dict[str, Interval] = {}
        pass_rates: dict[str, float] = {}

        for metric in self.metrics:
            values = run.scores(metric.name, include_errors=True, error_value=0.0)
            if not values:
                continue
            aggregates[metric.name] = (
                wilson_interval(int(sum(values)), len(values)) if metric.binary
                else bootstrap_interval(values, n_resamples=4_000)
            )
            if metric.threshold is not None:
                pass_rates[metric.name] = sum(metric.passed(v) for v in values) / len(values)

        assertion_results = []
        for assertion in self.assertions:
            interval = aggregates.get(assertion.metric)
            if interval is None:
                assertion_results.append((assertion.metric, False,
                                          f"metric '{assertion.metric}' not produced by this suite"))
                continue
            ok, reason = assertion.check(
                interval.estimate,
                baseline=(baseline or {}).get(assertion.metric),
                interval=interval,
            )
            assertion_results.append((assertion.metric, ok, reason))

        return SuiteReport(self.name, run, aggregates, pass_rates, assertion_results)


# ---------------------------------------------------------------------------
# Reusable metric constructors
# ---------------------------------------------------------------------------
def exact_match_metric(*, normalise: bool = True) -> Metric:
    from evalcore.judge.validators import exact_match

    def _score(case: EvalCase, output: str) -> float:
        if case.expected_output is None:
            return float("nan")
        return exact_match(output, case.expected_output, normalise=normalise).score

    return Metric("exact_match", _score, binary=True, threshold=1.0,
                  description="Normalised string equality against the expected output")


def contains_metric(name: str = "contains_expected") -> Metric:
    def _score(case: EvalCase, output: str) -> float:
        if not case.expected_output:
            return float("nan")
        return float(case.expected_output.lower().strip() in output.lower())

    return Metric(name, _score, binary=True, threshold=1.0,
                  description="Expected answer appears somewhere in the output")


def json_valid_metric(schema: dict[str, Any] | None = None) -> Metric:
    from evalcore.judge.validators import validate_json

    def _score(_: EvalCase, output: str) -> float:
        return validate_json(output, schema).score

    return Metric("json_valid", _score, binary=True, threshold=1.0,
                  description="Output parses as JSON and satisfies the schema")


def latency_metric(budget_ms: float) -> Metric:
    """Latency as a first-class metric so a speed regression fails the build too."""

    def _score(case: EvalCase, _: str) -> float:
        observed = float(case.metadata.get("latency_ms", 0.0))
        return float(observed <= budget_ms)

    return Metric("within_latency_budget", _score, binary=True, threshold=1.0,
                  description=f"Response produced within {budget_ms:.0f}ms")


def judge_metric(judge, *, name: str = "judge_score") -> Metric:
    """Wrap a :class:`~evalcore.judge.judge.RubricJudge` as a suite metric."""

    async def _score(case: EvalCase, output: str) -> float:
        result = await judge.ajudge(
            case.input, output,
            reference=case.expected_output,
            contexts=case.contexts or None,
        )
        return result.normalised_score

    return Metric(name, _score, threshold=0.6,
                  description=f"Rubric '{judge.rubric.name}' normalised to [0, 1]")
