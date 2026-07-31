"""Cost and latency evaluation.

Quality is one axis of a three-axis decision. A configuration that gains two
quality points for 4x the cost and 900ms of added p95 latency is usually the
wrong choice, and a report that omits cost and latency cannot say so. This
module makes the trade-off explicit and computes the frontier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# USD per 1M tokens. Prices change constantly; they live in one dict so a
# refresh is a one-line edit rather than a hunt through the code base. Verify
# against the provider's current pricing page before quoting a number.
PRICE_TABLE: dict[str, tuple[float, float]] = {
    # model: (input per 1M tokens, output per 1M tokens)
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.0-flash-lite": (0.075, 0.30),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.00),
}


@dataclass
class CostBreakdown:
    model: str
    n_requests: int
    input_tokens: int
    output_tokens: int
    input_cost_usd: float
    output_cost_usd: float

    @property
    def total_usd(self) -> float:
        return self.input_cost_usd + self.output_cost_usd

    @property
    def cost_per_request(self) -> float:
        return self.total_usd / self.n_requests if self.n_requests else 0.0

    def as_row(self) -> dict[str, float | str]:
        return {
            "model": self.model,
            "requests": self.n_requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_usd": round(self.total_usd, 5),
            "usd_per_1k_requests": round(self.cost_per_request * 1000, 4),
        }


def estimate_cost(
    model: str, input_tokens: int, output_tokens: int, *, n_requests: int = 1
) -> CostBreakdown:
    """Cost for a workload. Unknown models fall back to the most expensive entry.

    Falling back *upward* is deliberate: an underestimate hides a budget
    overrun until the invoice arrives, whereas an overestimate merely prompts
    someone to add the model to the price table.
    """
    input_price, output_price = PRICE_TABLE.get(model, max(PRICE_TABLE.values()))
    return CostBreakdown(
        model=model,
        n_requests=n_requests,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost_usd=input_tokens / 1_000_000 * input_price,
        output_cost_usd=output_tokens / 1_000_000 * output_price,
    )


def approximate_tokens(text: str) -> int:
    """Rough token count: ~4 characters per token for English.

    Adequate for cost *comparison* between configurations, which is what this
    module is for. Use the provider's token counter for billing reconciliation.
    """
    return max(1, len(text) // 4)


@dataclass
class LatencyReport:
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    max_ms: float
    n: int
    slo_ms: float = 0.0
    slo_violation_rate: float = 0.0

    def as_row(self) -> dict[str, float]:
        return {
            "p50_ms": round(self.p50_ms, 1), "p90_ms": round(self.p90_ms, 1),
            "p95_ms": round(self.p95_ms, 1), "p99_ms": round(self.p99_ms, 1),
            "mean_ms": round(self.mean_ms, 1), "max_ms": round(self.max_ms, 1),
            "slo_violation_rate": round(self.slo_violation_rate, 4), "n": self.n,
        }


def evaluate_latency(latencies_ms: Sequence[float], *, slo_ms: float = 3000.0) -> LatencyReport:
    """Tail-focused latency report.

    Percentiles rather than the mean, because LLM latency distributions are
    heavily right skewed: a mean of 900ms routinely hides a p99 of 8 seconds,
    and the p99 is what a user on a bad day experiences.
    """
    values = np.asarray(latencies_ms, dtype=float)
    if values.size == 0:
        return LatencyReport(0, 0, 0, 0, 0, 0, 0, slo_ms, 0.0)
    return LatencyReport(
        p50_ms=float(np.percentile(values, 50)),
        p90_ms=float(np.percentile(values, 90)),
        p95_ms=float(np.percentile(values, 95)),
        p99_ms=float(np.percentile(values, 99)),
        mean_ms=float(values.mean()),
        max_ms=float(values.max()),
        n=int(values.size),
        slo_ms=slo_ms,
        slo_violation_rate=float((values > slo_ms).mean()),
    )


@dataclass
class Configuration:
    """One point in the quality/cost/latency space."""

    name: str
    quality: float
    cost_per_1k_usd: float
    latency_p95_ms: float
    metadata: dict[str, float] = field(default_factory=dict)


def pareto_frontier(configurations: Sequence[Configuration]) -> list[Configuration]:
    """Configurations not dominated on all three axes simultaneously.

    A configuration is dominated when another is at least as good on quality,
    cost and latency, and strictly better on one. Everything dominated can be
    discarded without argument; the frontier is the genuinely contested set, and
    choosing within it is a product decision, not an engineering one.
    """
    frontier: list[Configuration] = []
    for candidate in configurations:
        dominated = any(
            other is not candidate
            and other.quality >= candidate.quality
            and other.cost_per_1k_usd <= candidate.cost_per_1k_usd
            and other.latency_p95_ms <= candidate.latency_p95_ms
            and (
                other.quality > candidate.quality
                or other.cost_per_1k_usd < candidate.cost_per_1k_usd
                or other.latency_p95_ms < candidate.latency_p95_ms
            )
            for other in configurations
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda c: -c.quality)


def quality_per_dollar(configuration: Configuration) -> float:
    """Quality points per dollar per 1k requests -- the blunt efficiency ratio."""
    return configuration.quality / configuration.cost_per_1k_usd if configuration.cost_per_1k_usd else float("inf")


def evaluation_run_cost(
    n_cases: int,
    *,
    judge_model: str,
    mean_prompt_tokens: int = 1200,
    mean_output_tokens: int = 350,
    n_judge_samples: int = 1,
    cache_hit_rate: float = 0.0,
) -> dict[str, float]:
    """Cost of the *evaluation suite itself*.

    Evaluation cost is a real line item and the reason continuous evaluation
    gets cancelled six weeks in. Two levers dominate: the cache hit rate, and
    self-consistency sampling, which multiplies cost linearly for a sub-linear
    gain in reliability.
    """
    effective_calls = n_cases * n_judge_samples * (1 - cache_hit_rate)
    breakdown = estimate_cost(
        judge_model,
        input_tokens=int(effective_calls * mean_prompt_tokens),
        output_tokens=int(effective_calls * mean_output_tokens),
        n_requests=int(effective_calls),
    )
    return {
        "n_cases": n_cases,
        "effective_llm_calls": round(effective_calls, 1),
        "total_usd": round(breakdown.total_usd, 4),
        "usd_per_case": round(breakdown.total_usd / n_cases, 6) if n_cases else 0.0,
        "monthly_usd_if_daily": round(breakdown.total_usd * 30, 2),
        "cache_hit_rate": cache_hit_rate,
        "n_judge_samples": n_judge_samples,
    }
