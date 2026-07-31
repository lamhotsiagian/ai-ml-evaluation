"""Online evaluation: A/B tests, sequential monitoring, shadow and canary logic.

Offline evaluation tells you whether a change is likely to be better. Online
evaluation tells you whether it *is*, on real traffic, against the metric the
business is paid on. This module implements the three mechanisms that make that
safe: fixed-horizon A/B analysis, an always-valid sequential test for continuous
monitoring, and a canary controller with automatic rollback.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
from scipy import stats as sps

from evalcore.stats import Interval, required_n_for_proportion


# ---------------------------------------------------------------------------
# Deterministic assignment
# ---------------------------------------------------------------------------
def assign_variant(unit_id: str, experiment: str, *, weights: dict[str, float] | None = None) -> str:
    """Hash-based, sticky assignment of a unit to a variant.

    Hashing on (experiment, unit) rather than storing assignments gives three
    properties for free: the same user always sees the same variant, two
    concurrent experiments are independent, and no assignment database has to be
    on the request path.
    """
    weights = weights or {"control": 0.5, "treatment": 0.5}
    total = sum(weights.values())
    digest = hashlib.sha256(f"{experiment}:{unit_id}".encode("utf-8")).hexdigest()
    position = (int(digest[:16], 16) / float(1 << 64)) * total
    cumulative = 0.0
    for variant, weight in sorted(weights.items()):
        cumulative += weight
        if position < cumulative:
            return variant
    return sorted(weights)[-1]


# ---------------------------------------------------------------------------
# Fixed-horizon A/B analysis
# ---------------------------------------------------------------------------
@dataclass
class ABResult:
    control_rate: float
    treatment_rate: float
    absolute_lift: float
    relative_lift: float
    p_value: float
    lift_interval: Interval
    n_control: int
    n_treatment: int
    decision: Literal["ship", "rollback", "inconclusive"]
    reason: str

    def as_row(self) -> dict[str, float | str]:
        return {
            "control": round(self.control_rate, 4),
            "treatment": round(self.treatment_rate, 4),
            "abs_lift": round(self.absolute_lift, 4),
            "rel_lift": round(self.relative_lift, 4),
            "p_value": round(self.p_value, 5),
            "ci": f"[{self.lift_interval.low:+.4f}, {self.lift_interval.high:+.4f}]",
            "n": self.n_control + self.n_treatment,
            "decision": self.decision,
        }


def analyse_ab_test(
    control_outcomes: Sequence[float],
    treatment_outcomes: Sequence[float],
    *,
    alpha: float = 0.05,
    practical_threshold: float = 0.0,
    guardrail_violated: bool = False,
) -> ABResult:
    """Fixed-horizon two-proportion analysis with a practical-significance gate.

    ``practical_threshold`` is the minimum lift worth the operational cost of
    shipping. It is what stops a team from shipping a statistically significant
    +0.2% that doubles latency; the confidence interval must clear the threshold
    entirely, not merely exclude zero.
    """
    control = np.asarray(control_outcomes, dtype=float)
    treatment = np.asarray(treatment_outcomes, dtype=float)
    if control.size == 0 or treatment.size == 0:
        raise ValueError("both arms need observations")

    p_control, p_treatment = float(control.mean()), float(treatment.mean())
    lift = p_treatment - p_control
    relative = lift / p_control if p_control > 0 else float("inf")

    pooled = (control.sum() + treatment.sum()) / (control.size + treatment.size)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / control.size + 1 / treatment.size))
    z = lift / se_pooled if se_pooled > 0 else 0.0
    p_value = float(2 * (1 - sps.norm.cdf(abs(z))))

    se_unpooled = math.sqrt(
        p_control * (1 - p_control) / control.size + p_treatment * (1 - p_treatment) / treatment.size
    )
    margin = float(sps.norm.ppf(1 - alpha / 2)) * se_unpooled
    interval = Interval(lift, lift - margin, lift + margin, 1 - alpha, "normal-approx",
                        control.size + treatment.size)

    if guardrail_violated:
        decision, reason = "rollback", "guardrail metric violated; primary metric is irrelevant"
    elif p_value >= alpha:
        decision, reason = "inconclusive", f"p={p_value:.3f} does not clear alpha={alpha}"
    elif lift < 0:
        decision, reason = "rollback", "treatment is significantly worse"
    elif interval.low <= practical_threshold:
        decision = "inconclusive"
        reason = (f"significant but the CI lower bound {interval.low:+.4f} does not clear the "
                  f"practical threshold {practical_threshold:+.4f}")
    else:
        decision, reason = "ship", "significant and practically meaningful"

    return ABResult(p_control, p_treatment, lift, relative, p_value, interval,
                    control.size, treatment.size, decision, reason)


def sample_size_for_experiment(
    baseline_rate: float, minimum_detectable_effect: float, *,
    alpha: float = 0.05, power: float = 0.8, daily_traffic: int | None = None,
) -> dict[str, float]:
    """Sample size and expected runtime -- computed before the experiment starts.

    Running an experiment without this calculation is how teams end up peeking
    at an underpowered test and shipping noise.
    """
    analysis = required_n_for_proportion(baseline_rate, minimum_detectable_effect,
                                         alpha=alpha, power=power)
    payload: dict[str, float] = {
        "n_per_arm": analysis.n_required,
        "n_total": analysis.n_required * 2,
        "baseline_rate": baseline_rate,
        "mde": minimum_detectable_effect,
        "power": power,
        "alpha": alpha,
    }
    if daily_traffic:
        payload["days_required"] = math.ceil(analysis.n_required * 2 / daily_traffic)
    return payload


# ---------------------------------------------------------------------------
# Always-valid sequential testing
# ---------------------------------------------------------------------------
@dataclass
class SequentialState:
    """Running state of a mixture sequential probability ratio test."""

    n_control: int = 0
    n_treatment: int = 0
    sum_control: float = 0.0
    sum_treatment: float = 0.0
    log_likelihood_ratio: float = 0.0
    boundary: float = math.log(1 / 0.05)
    history: list[float] = field(default_factory=list)

    @property
    def decision(self) -> Literal["continue", "reject_null", "accept_null"]:
        if self.log_likelihood_ratio >= self.boundary:
            return "reject_null"
        if self.log_likelihood_ratio <= -self.boundary:
            return "accept_null"
        return "continue"


class SequentialTest:
    """Always-valid test that permits continuous monitoring without alpha inflation.

    A fixed-horizon p-value is only valid if you look once, at the pre-registered
    sample size. Teams look every day, and looking daily at alpha = 0.05 pushes
    the true false-positive rate above 20%. The mSPRT keeps the type-I error at
    alpha no matter how often you peek, which is what makes automated rollback
    safe to wire up.
    """

    def __init__(self, *, alpha: float = 0.05, tau: float = 0.05) -> None:
        self.alpha = alpha
        self.tau = tau  # prior standard deviation on the effect under H1
        self.state = SequentialState(boundary=math.log(1 / alpha))

    def update(self, control_batch: Sequence[float], treatment_batch: Sequence[float]) -> SequentialState:
        control = np.asarray(control_batch, dtype=float)
        treatment = np.asarray(treatment_batch, dtype=float)
        self.state.n_control += control.size
        self.state.n_treatment += treatment.size
        self.state.sum_control += float(control.sum())
        self.state.sum_treatment += float(treatment.sum())

        n_c, n_t = self.state.n_control, self.state.n_treatment
        if n_c < 2 or n_t < 2:
            return self.state

        p_c = self.state.sum_control / n_c
        p_t = self.state.sum_treatment / n_t
        pooled = (self.state.sum_control + self.state.sum_treatment) / (n_c + n_t)
        variance = max(pooled * (1 - pooled), 1e-9) * (1 / n_c + 1 / n_t)
        delta = p_t - p_c

        # mSPRT with a normal mixing distribution (Johari et al., 2017).
        ratio = math.sqrt(variance / (variance + self.tau**2)) * math.exp(
            (self.tau**2 * delta**2) / (2 * variance * (variance + self.tau**2))
        )
        self.state.log_likelihood_ratio = math.log(max(ratio, 1e-12))
        self.state.history.append(self.state.log_likelihood_ratio)
        return self.state


# ---------------------------------------------------------------------------
# Canary rollout controller
# ---------------------------------------------------------------------------
@dataclass
class CanaryDecision:
    action: Literal["promote", "hold", "rollback"]
    traffic_percent: float
    reason: str
    metrics: dict[str, float] = field(default_factory=dict)


class CanaryController:
    """Staged rollout with automatic rollback on guardrail breach.

    The ladder (1% -> 5% -> 25% -> 50% -> 100%) exists so that a catastrophic
    change is seen by 1% of traffic, not 50%. The controller only promotes when
    the current stage has accumulated enough observations to detect the
    regression it is watching for -- promoting on 40 requests is theatre.
    """

    LADDER = (1.0, 5.0, 25.0, 50.0, 100.0)

    def __init__(
        self,
        *,
        error_rate_budget: float = 0.02,
        latency_p95_budget_ms: float = 3000.0,
        quality_drop_budget: float = 0.03,
        min_observations: int = 200,
    ) -> None:
        self.error_rate_budget = error_rate_budget
        self.latency_p95_budget_ms = latency_p95_budget_ms
        self.quality_drop_budget = quality_drop_budget
        self.min_observations = min_observations
        self.stage = 0

    def evaluate(
        self,
        *,
        n_observations: int,
        error_rate: float,
        latency_p95_ms: float,
        quality_score: float,
        baseline_quality: float,
    ) -> CanaryDecision:
        metrics = {
            "n": float(n_observations), "error_rate": error_rate,
            "latency_p95_ms": latency_p95_ms, "quality": quality_score,
            "quality_drop": baseline_quality - quality_score,
        }
        traffic = self.LADDER[self.stage]

        breaches: list[str] = []
        if error_rate > self.error_rate_budget:
            breaches.append(f"error rate {error_rate:.3f} > budget {self.error_rate_budget:.3f}")
        if latency_p95_ms > self.latency_p95_budget_ms:
            breaches.append(f"p95 latency {latency_p95_ms:.0f}ms > budget {self.latency_p95_budget_ms:.0f}ms")
        if (baseline_quality - quality_score) > self.quality_drop_budget:
            breaches.append(f"quality dropped {baseline_quality - quality_score:.3f}")

        if breaches:
            self.stage = 0
            return CanaryDecision("rollback", 0.0, "; ".join(breaches), metrics)
        if n_observations < self.min_observations:
            return CanaryDecision("hold", traffic,
                                  f"only {n_observations} observations; need {self.min_observations}",
                                  metrics)
        if self.stage >= len(self.LADDER) - 1:
            return CanaryDecision("promote", 100.0, "fully rolled out", metrics)
        self.stage += 1
        return CanaryDecision("promote", self.LADDER[self.stage],
                              f"all guardrails within budget at {traffic:.0f}%", metrics)


def shadow_comparison(
    production_outputs: Sequence[str],
    shadow_outputs: Sequence[str],
    *,
    scorer=None,
) -> dict[str, float]:
    """Compare a shadow deployment with live production on identical traffic.

    Shadow mode is the highest-signal, lowest-risk evaluation available: the
    candidate sees exactly the production distribution and no user sees its
    output. The disagreement rate is the number to watch, because agreement on
    easy traffic tells you nothing and every disagreement is a reviewable case.
    """
    if len(production_outputs) != len(shadow_outputs):
        raise ValueError("shadow comparison requires paired outputs")
    if not production_outputs:
        return {"n": 0, "disagreement_rate": float("nan")}

    if scorer is None:
        disagreements = sum(
            1 for live, shadow in zip(production_outputs, shadow_outputs)
            if live.strip() != shadow.strip()
        )
        return {
            "n": len(production_outputs),
            "disagreement_rate": disagreements / len(production_outputs),
            "n_disagreements": disagreements,
        }

    live_scores = np.asarray([scorer(o) for o in production_outputs], dtype=float)
    shadow_scores = np.asarray([scorer(o) for o in shadow_outputs], dtype=float)
    return {
        "n": len(production_outputs),
        "production_mean": float(live_scores.mean()),
        "shadow_mean": float(shadow_scores.mean()),
        "delta": float(shadow_scores.mean() - live_scores.mean()),
        "win_rate": float((shadow_scores > live_scores).mean()),
        "loss_rate": float((shadow_scores < live_scores).mean()),
    }
