"""Statistical machinery for trustworthy evaluation decisions.

An evaluation score without an interval is a rumour. This module supplies the
four things a release decision actually needs:

* an **interval** around every reported metric (BCa bootstrap by default),
* a **paired test** for "is B better than A on the same items",
* an **effect size** so a statistically detectable change can be judged
  practically meaningful, and
* **power / minimum-detectable-effect** analysis so a suite is sized before it
  is run rather than explained away afterwards.

All functions are pure NumPy/SciPy and are unit tested; nothing here calls a
model, so the statistics can be validated independently of any LLM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

import numpy as np
from scipy import stats as sps

ArrayLike = Sequence[float] | np.ndarray


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Interval:
    """A point estimate with a confidence interval."""

    estimate: float
    low: float
    high: float
    confidence: float = 0.95
    method: str = "bootstrap-bca"
    n: int = 0

    @property
    def half_width(self) -> float:
        return (self.high - self.low) / 2.0

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def as_text(self, digits: int = 4) -> str:
        return (
            f"{self.estimate:.{digits}f} "
            f"[{self.low:.{digits}f}, {self.high:.{digits}f}] "
            f"({self.confidence:.0%} CI, n={self.n})"
        )


def bootstrap_interval(
    values: ArrayLike,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    method: Literal["bca", "percentile", "basic"] = "bca",
    seed: int = 1337,
) -> Interval:
    """Bootstrap a confidence interval for any statistic of a single sample.

    BCa (bias-corrected and accelerated) is the default because evaluation
    metrics are usually bounded and skewed near the edges: a plain percentile
    interval on a model scoring 0.97 accuracy will happily report an upper bound
    above 1.0, which is not a defensible number to put in a release document.

    Args:
        values: Per-item scores (0/1 for accuracy-style metrics is fine).
        statistic: Function mapping a resample to a scalar.
        confidence: Two-sided confidence level.
        n_resamples: Bootstrap replicates. 10k is the practical floor for a
            stable 95% interval; go to 100k for a publication figure.
        method: SciPy bootstrap method.
        seed: RNG seed; evaluation must be reproducible.

    Returns:
        An :class:`Interval`. Degenerate samples (n < 2, or zero variance)
        return a zero-width interval rather than raising, because a suite with
        one row should degrade to "no information", not crash a dashboard.
    """
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), confidence, method, 0)

    point = float(statistic(array))
    if array.size < 2 or np.allclose(array, array[0]):
        return Interval(point, point, point, confidence, f"{method}-degenerate", array.size)

    result = sps.bootstrap(
        (array,),
        statistic,
        confidence_level=confidence,
        n_resamples=n_resamples,
        method=method,
        random_state=np.random.default_rng(seed),
        vectorized=False,
    )
    return Interval(
        estimate=point,
        low=float(result.confidence_interval.low),
        high=float(result.confidence_interval.high),
        confidence=confidence,
        method=f"bootstrap-{method}",
        n=int(array.size),
    )


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion.

    Preferred over the Wald (normal approximation) interval for pass/fail
    evaluation suites: Wald collapses to zero width at 0% and 100% pass rates,
    which is exactly where small evaluation suites tend to sit.
    """
    if trials <= 0:
        return Interval(float("nan"), float("nan"), float("nan"), confidence, "wilson", 0)
    z = float(sps.norm.ppf(1 - (1 - confidence) / 2))
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    margin = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    # Clamp, then snap away floating-point dust at the boundaries so a 0/n suite
    # reports a lower bound of exactly 0.0 rather than 7e-18.
    low = min(max(0.0, centre - margin), 1.0)
    high = min(max(0.0, centre + margin), 1.0)
    low = 0.0 if low < 1e-12 else low
    high = 1.0 if high > 1 - 1e-12 else high
    return Interval(p, low, high, confidence, "wilson", trials)


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------
def cohens_d(a: ArrayLike, b: ArrayLike, *, paired: bool = False) -> float:
    """Standardised mean difference (b - a).

    Rough field conventions: 0.2 small, 0.5 medium, 0.8 large. Report it next to
    every p-value; with 5,000 evaluation rows almost any difference is
    "significant" and only the effect size tells you whether to ship.
    """
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if paired:
        diff = y - x
        sd = diff.std(ddof=1)
        return 0.0 if sd == 0 else float(diff.mean() / sd)
    nx, ny = x.size, y.size
    pooled_var = ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
    return 0.0 if pooled_var == 0 else float((y.mean() - x.mean()) / math.sqrt(pooled_var))


def cliffs_delta(a: ArrayLike, b: ArrayLike) -> float:
    """Non-parametric effect size in [-1, 1]; robust to the skew of judge scores.

    Interpreted as P(b > a) - P(a > b). Use this instead of Cohen's d when the
    metric is an ordinal rubric score (1-5), where the distance between 4 and 5
    is not the same quantity as the distance between 1 and 2.
    """
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.size == 0 or y.size == 0:
        return 0.0
    comparisons = np.sign(y[:, None] - x[None, :])
    return float(comparisons.mean())


# ---------------------------------------------------------------------------
# Hypothesis tests
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TestResult:
    """The complete output of a comparison, not just a p-value."""

    name: str
    statistic: float
    p_value: float
    effect_size: float
    effect_name: str
    delta: float
    delta_interval: Interval | None = None
    n: int = 0
    detail: dict[str, float] = field(default_factory=dict)

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha

    def as_text(self) -> str:
        return (
            f"{self.name}: delta={self.delta:+.4f}, p={self.p_value:.4g}, "
            f"{self.effect_name}={self.effect_size:+.3f}, n={self.n}"
        )


def paired_bootstrap_test(
    baseline: ArrayLike,
    candidate: ArrayLike,
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 1337,
) -> TestResult:
    """Paired bootstrap on the per-item difference. The default A/B test here.

    Pairing matters enormously. Evaluation items differ wildly in difficulty; an
    unpaired test spends most of its power on item variance that both systems
    share. Running both systems on the *same* items and bootstrapping the
    difference removes that variance entirely, which typically cuts the required
    sample size by an order of magnitude.
    """
    x, y = np.asarray(baseline, dtype=float), np.asarray(candidate, dtype=float)
    if x.shape != y.shape:
        raise ValueError("paired test requires equal-length, item-aligned score vectors")
    diff = y - x
    observed = float(diff.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_resamples, diff.size))
    resampled = diff[idx].mean(axis=1)

    # Two-sided p-value from the null-centred bootstrap distribution.
    centred = resampled - resampled.mean()
    p_value = float((np.abs(centred) >= abs(observed)).mean())
    p_value = max(p_value, 1.0 / n_resamples)  # never report p = 0

    alpha = 1 - confidence
    low, high = np.quantile(resampled, [alpha / 2, 1 - alpha / 2])
    return TestResult(
        name="paired bootstrap",
        statistic=observed,
        p_value=p_value,
        effect_size=cohens_d(x, y, paired=True),
        effect_name="Cohen's d (paired)",
        delta=observed,
        delta_interval=Interval(observed, float(low), float(high), confidence, "bootstrap", diff.size),
        n=int(diff.size),
    )


def mcnemar_test(baseline: ArrayLike, candidate: ArrayLike, *, exact: bool = True) -> TestResult:
    """McNemar's test for paired binary outcomes (pass/fail per item).

    The right test when both systems are graded pass/fail on the same suite. It
    looks only at the discordant pairs -- items one system passed and the other
    failed -- because concordant items carry no evidence about which is better.
    """
    x = np.asarray(baseline).astype(int)
    y = np.asarray(candidate).astype(int)
    if x.shape != y.shape:
        raise ValueError("McNemar requires item-aligned binary vectors")

    b = int(np.sum((x == 1) & (y == 0)))  # baseline passed, candidate failed
    c = int(np.sum((x == 0) & (y == 1)))  # candidate fixed a baseline failure
    n_disc = b + c

    if n_disc == 0:
        return TestResult("McNemar", 0.0, 1.0, 0.0, "odds ratio", 0.0, None, int(x.size),
                          {"b": 0.0, "c": 0.0, "discordant": 0.0})

    if exact or n_disc < 25:
        p_value = float(sps.binomtest(c, n_disc, 0.5).pvalue)
        statistic = float(c)
        name = "McNemar (exact binomial)"
    else:
        statistic = float((abs(b - c) - 1) ** 2 / n_disc)  # continuity corrected
        p_value = float(sps.chi2.sf(statistic, df=1))
        name = "McNemar (chi-square, corrected)"

    odds_ratio = float(c / b) if b else float("inf")
    return TestResult(
        name=name,
        statistic=statistic,
        p_value=p_value,
        effect_size=odds_ratio,
        effect_name="odds ratio (fixed/broken)",
        delta=float(y.mean() - x.mean()),
        delta_interval=None,
        n=int(x.size),
        detail={"b_broken": float(b), "c_fixed": float(c), "discordant": float(n_disc)},
    )


def welch_t_test(baseline: ArrayLike, candidate: ArrayLike) -> TestResult:
    """Welch's t-test for unpaired samples with unequal variance.

    Use only when pairing is genuinely impossible -- for example, comparing two
    production traffic slices that saw different requests.
    """
    x, y = np.asarray(baseline, dtype=float), np.asarray(candidate, dtype=float)
    result = sps.ttest_ind(y, x, equal_var=False)
    return TestResult(
        name="Welch t-test",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        effect_size=cohens_d(x, y),
        effect_name="Cohen's d",
        delta=float(y.mean() - x.mean()),
        n=int(x.size + y.size),
    )


def mann_whitney(baseline: ArrayLike, candidate: ArrayLike) -> TestResult:
    """Rank-based unpaired test; the right default for 1-5 rubric scores."""
    x, y = np.asarray(baseline, dtype=float), np.asarray(candidate, dtype=float)
    result = sps.mannwhitneyu(y, x, alternative="two-sided")
    return TestResult(
        name="Mann-Whitney U",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        effect_size=cliffs_delta(x, y),
        effect_name="Cliff's delta",
        delta=float(np.median(y) - np.median(x)),
        n=int(x.size + y.size),
    )


# ---------------------------------------------------------------------------
# Multiple comparisons
# ---------------------------------------------------------------------------
def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    """Return a boolean mask of hypotheses rejected under FDR control.

    Evaluation dashboards routinely test 40 metrics across 12 slices. At
    alpha = 0.05 that is 24 false alarms per clean release. Controlling the
    false discovery rate is not optional once a dashboard has more than a
    handful of panels.
    """
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    thresholds = alpha * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresholds
    rejected = np.zeros(n, dtype=bool)
    if passed.any():
        cutoff = np.max(np.nonzero(passed)[0])
        rejected[order[: cutoff + 1]] = True
    return rejected


# ---------------------------------------------------------------------------
# Power and sample sizing
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PowerAnalysis:
    """Answer to "how many evaluation items do I need?"."""

    n_required: int
    effect: float
    alpha: float
    power: float
    test: str

    def as_text(self) -> str:
        return (
            f"{self.test}: n>={self.n_required} per arm to detect {self.effect:+.3f} "
            f"at alpha={self.alpha}, power={self.power:.0%}"
        )


def required_n_for_proportion(
    baseline_rate: float,
    minimum_detectable_effect: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    paired_correlation: float = 0.0,
) -> PowerAnalysis:
    """Sample size for detecting a shift in a pass rate.

    ``paired_correlation`` is the correlation between the two systems' per-item
    outcomes. Set it honestly: for two checkpoints of the same model on the same
    suite it is routinely 0.7-0.9, and the variance-reduction factor
    ``(1 - rho)`` is the single largest lever on evaluation cost.
    """
    if not 0 < baseline_rate < 1:
        raise ValueError("baseline_rate must be strictly between 0 and 1")
    p1 = baseline_rate
    p2 = min(max(baseline_rate + minimum_detectable_effect, 1e-6), 1 - 1e-6)
    p_bar = (p1 + p2) / 2

    z_alpha = float(sps.norm.ppf(1 - alpha / 2))
    z_beta = float(sps.norm.ppf(power))
    numerator = (z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
                 + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    n = numerator / (p2 - p1) ** 2
    n *= max(1e-6, 1.0 - paired_correlation)
    return PowerAnalysis(int(math.ceil(n)), minimum_detectable_effect, alpha, power,
                         "two-proportion z-test")


def minimum_detectable_effect(
    n: int, baseline_rate: float, *, alpha: float = 0.05, power: float = 0.8
) -> float:
    """The smallest true improvement a suite of size ``n`` can reliably detect.

    Run this *before* the evaluation. If the MDE is 6 points and the team is
    arguing about a 1-point regression, the suite cannot settle the argument and
    no amount of dashboard staring will change that.
    """
    if n <= 0:
        return float("nan")
    z_alpha = float(sps.norm.ppf(1 - alpha / 2))
    z_beta = float(sps.norm.ppf(power))
    se = math.sqrt(2 * baseline_rate * (1 - baseline_rate) / n)
    return float((z_alpha + z_beta) * se)


# ---------------------------------------------------------------------------
# Inter-rater agreement (judge calibration, Chapter 3)
# ---------------------------------------------------------------------------
def cohens_kappa(rater_a: Sequence, rater_b: Sequence) -> float:
    """Chance-corrected agreement between two raters over categorical labels.

    Raw agreement flatters a judge on any imbalanced rubric: a judge that always
    says "pass" scores 90% agreement on a suite that is 90% passes while
    carrying zero information. Kappa removes that baseline.
    """
    a, b = list(rater_a), list(rater_b)
    if len(a) != len(b) or not a:
        raise ValueError("kappa requires two equal-length, non-empty label sequences")
    labels = sorted({*a, *b}, key=str)
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=float)
    for left, right in zip(a, b):
        matrix[index[left], index[right]] += 1
    total = matrix.sum()
    observed = np.trace(matrix) / total
    expected = float((matrix.sum(axis=0) @ matrix.sum(axis=1)) / total**2)
    return 1.0 if expected == 1 else float((observed - expected) / (1 - expected))


def krippendorff_alpha_nominal(ratings: np.ndarray) -> float:
    """Krippendorff's alpha for nominal data with missing values (NaN allowed).

    Args:
        ratings: 2-D array shaped (n_raters, n_items); NaN marks "not rated".

    Unlike kappa this handles three or more judges and incomplete rating
    designs, which is the realistic situation once a human panel is involved.
    """
    data = np.asarray(ratings, dtype=float)
    observed_disagreement, pairs = 0.0, 0
    value_counts: dict[float, float] = {}

    for item in range(data.shape[1]):
        column = data[:, item]
        column = column[~np.isnan(column)]
        m = column.size
        if m < 2:
            continue
        for value in column:
            value_counts[value] = value_counts.get(value, 0.0) + 1
        for i in range(m):
            for j in range(m):
                if i != j and column[i] != column[j]:
                    observed_disagreement += 1 / (m - 1)
        pairs += m

    if pairs == 0:
        return float("nan")
    do = observed_disagreement / pairs
    total = sum(value_counts.values())
    de = 1.0 - sum((count / total) ** 2 for count in value_counts.values())
    de *= total / max(total - 1, 1)
    return 1.0 if de == 0 else float(1 - do / de)
