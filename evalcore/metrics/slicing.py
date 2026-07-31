"""Slice evaluation, fairness and automatic error-cluster discovery.

Aggregate metrics are a compression of the truth, and the compression always
discards the failure you were hired to find. This module makes the slice the
unit of evaluation: every metric is computed per slice, differences are tested
with multiplicity control, and unlabelled failure clusters are surfaced so a new
slice can be *discovered* rather than guessed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Hashable, Sequence

import numpy as np

from evalcore.stats import Interval, benjamini_hochberg, bootstrap_interval, wilson_interval


@dataclass
class SliceResult:
    name: str
    n: int
    score: Interval
    delta_vs_overall: float
    p_value: float = 1.0
    flagged: bool = False

    def as_row(self) -> dict[str, float | str | bool]:
        return {
            "slice": self.name,
            "n": self.n,
            "score": round(self.score.estimate, 4),
            "ci_low": round(self.score.low, 4),
            "ci_high": round(self.score.high, 4),
            "delta_vs_overall": round(self.delta_vs_overall, 4),
            "p_value": round(self.p_value, 5),
            "flagged": self.flagged,
        }


@dataclass
class SliceReport:
    overall: Interval
    slices: list[SliceResult] = field(default_factory=list)
    min_slice_size: int = 20
    alpha: float = 0.05

    @property
    def worst(self) -> SliceResult | None:
        eligible = [s for s in self.slices if s.n >= self.min_slice_size]
        return min(eligible, key=lambda s: s.score.estimate) if eligible else None

    @property
    def flagged(self) -> list[SliceResult]:
        return [s for s in self.slices if s.flagged]

    def summary(self) -> str:
        worst = self.worst
        head = f"overall={self.overall.estimate:.4f} over {self.overall.n} items"
        if worst is None:
            return head
        return (f"{head}; worst slice '{worst.name}' = {worst.score.estimate:.4f} "
                f"({worst.delta_vs_overall:+.4f}, n={worst.n}); {len(self.flagged)} flagged")


def evaluate_slices(
    scores: Sequence[float],
    slice_tags: Sequence[Sequence[str]],
    *,
    min_slice_size: int = 20,
    alpha: float = 0.05,
    binary: bool = True,
) -> SliceReport:
    """Per-slice metrics with FDR-controlled flagging.

    A row may carry several tags (``["mobile", "spanish", "long_query"]``) and
    contributes to each slice; slices therefore overlap, and their p-values are
    correlated. Benjamini-Hochberg is applied across slices so a 40-slice
    dashboard does not manufacture two false alarms per clean release.

    Args:
        scores: Per-item scores (0/1 when ``binary``).
        slice_tags: Per-item tag lists, aligned with ``scores``.
        min_slice_size: Slices smaller than this are reported but never flagged;
            below roughly 20 items the interval is too wide to act on.
        binary: Use a Wilson interval (pass/fail) instead of a bootstrap.
    """
    array = np.asarray(scores, dtype=float)
    if array.size != len(slice_tags):
        raise ValueError("scores and slice_tags must be aligned")

    overall = (wilson_interval(int(array.sum()), array.size) if binary
               else bootstrap_interval(array, n_resamples=5_000))

    buckets: dict[str, list[int]] = defaultdict(list)
    for index, tags in enumerate(slice_tags):
        for tag in tags:
            buckets[tag].append(index)

    results: list[SliceResult] = []
    p_values: list[float] = []
    for name, indices in sorted(buckets.items()):
        subset = array[np.asarray(indices, dtype=int)]
        interval = (wilson_interval(int(subset.sum()), subset.size) if binary
                    else bootstrap_interval(subset, n_resamples=2_000))
        rest = np.delete(array, indices)
        p_value = _two_sample_p(subset, rest)
        p_values.append(p_value)
        results.append(SliceResult(
            name=name,
            n=int(subset.size),
            score=interval,
            delta_vs_overall=float(interval.estimate - overall.estimate),
            p_value=p_value,
        ))

    if results:
        rejected = benjamini_hochberg(p_values, alpha=alpha)
        for result, reject in zip(results, rejected):
            result.flagged = bool(reject and result.n >= min_slice_size
                                  and result.delta_vs_overall < 0)

    return SliceReport(overall=overall, slices=results, min_slice_size=min_slice_size, alpha=alpha)


def evaluate_slices_by_metric(
    y_true: Sequence[int],
    y_score: Sequence[float],
    slice_tags: Sequence[Sequence[str]],
    *,
    metric: Callable[[np.ndarray, np.ndarray], float],
    min_slice_size: int = 30,
    alpha: float = 0.05,
    n_resamples: int = 400,
    seed: int = 1337,
) -> SliceReport:
    """Per-slice evaluation for metrics that cannot be averaged over items.

    :func:`evaluate_slices` averages a per-item score, which is correct for
    accuracy or pass/fail but wrong for AUC, F1, precision and every other
    metric defined over a *set* of items. Those must be recomputed within each
    slice, which is what this function does.

    The distinction is not academic. On an imbalanced task, per-item accuracy is
    pinned near ``1 - prevalence`` in every slice regardless of model skill, so
    an accuracy-sliced dashboard reports a flat line over a model that is close
    to random on one segment. Recomputing AUC per slice exposes it immediately.

    Args:
        y_true: Binary labels.
        y_score: Model scores.
        slice_tags: Per-item tag lists, aligned with ``y_true``.
        metric: Callable ``(y_true, y_score) -> float`` evaluated per slice.
        min_slice_size: Slices smaller than this are reported but never flagged.
        alpha: FDR level applied across slices.
        n_resamples: Bootstrap replicates for each slice's interval.
        seed: RNG seed.
    """
    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    if truth.size != len(slice_tags):
        raise ValueError("y_true and slice_tags must be aligned")

    overall_value = float(metric(truth, scores))
    overall = _bootstrap_metric(truth, scores, metric, n_resamples=n_resamples, seed=seed)

    buckets: dict[str, list[int]] = defaultdict(list)
    for index, tags in enumerate(slice_tags):
        for tag in tags:
            buckets[tag].append(index)

    results: list[SliceResult] = []
    p_values: list[float] = []
    rng = np.random.default_rng(seed)

    for name, indices in sorted(buckets.items()):
        idx = np.asarray(indices, dtype=int)
        if len(np.unique(truth[idx])) < 2:
            continue  # metric undefined for a single-class slice
        interval = _bootstrap_metric(truth[idx], scores[idx], metric,
                                     n_resamples=n_resamples, seed=seed)
        # Two-stage testing. A slice whose bootstrap interval already contains
        # the overall value cannot clear a permutation test, so skip it and
        # spend the permutation budget on the candidates that might. This keeps
        # the whole report interactive while giving the plausible slices enough
        # replicates to resolve a small p-value.
        if interval.contains(overall_value):
            p_value = 1.0
        else:
            p_value = _permutation_p_metric(truth, scores, idx, metric, rng,
                                            n_permutations=2_000)
        p_values.append(p_value)
        results.append(SliceResult(
            name=name,
            n=int(idx.size),
            score=interval,
            delta_vs_overall=float(interval.estimate - overall_value),
            p_value=p_value,
        ))

    if results:
        rejected = benjamini_hochberg(p_values, alpha=alpha)
        for result, reject in zip(results, rejected):
            result.flagged = bool(reject and result.n >= min_slice_size
                                  and result.delta_vs_overall < 0)

    return SliceReport(overall=overall, slices=results,
                       min_slice_size=min_slice_size, alpha=alpha)


def _bootstrap_metric(
    truth: np.ndarray, scores: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *, n_resamples: int, seed: int,
) -> Interval:
    """Bootstrap a set-level metric by resampling items and recomputing it."""
    point = float(metric(truth, scores))
    if truth.size < 8:
        return Interval(point, point, point, 0.95, "bootstrap-small", int(truth.size))

    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, truth.size, truth.size)
        if len(np.unique(truth[idx])) < 2:
            continue
        values.append(float(metric(truth[idx], scores[idx])))
    if not values:
        return Interval(point, point, point, 0.95, "bootstrap-degenerate", int(truth.size))
    low, high = np.quantile(values, [0.025, 0.975])
    return Interval(point, float(low), float(high), 0.95, "bootstrap-percentile", int(truth.size))


def _permutation_p_metric(
    truth: np.ndarray, scores: np.ndarray, indices: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    rng: np.random.Generator, n_permutations: int = 400,
) -> float:
    """Permutation p-value for "this slice's metric differs from the rest".

    Slice membership is shuffled while class balance is preserved by sampling a
    same-sized index set, so the null is "a random subset of this size", not "a
    random relabelling of the outcome".
    """
    mask = np.zeros(truth.size, dtype=bool)
    mask[indices] = True
    rest = ~mask
    if len(np.unique(truth[rest])) < 2 or len(np.unique(truth[mask])) < 2:
        return 1.0

    observed = abs(float(metric(truth[mask], scores[mask]))
                   - float(metric(truth[rest], scores[rest])))
    count = 0
    all_idx = np.arange(truth.size)
    for _ in range(n_permutations):
        sampled = rng.choice(all_idx, size=indices.size, replace=False)
        other = np.setdiff1d(all_idx, sampled, assume_unique=False)
        if len(np.unique(truth[sampled])) < 2 or len(np.unique(truth[other])) < 2:
            continue
        delta = abs(float(metric(truth[sampled], scores[sampled]))
                    - float(metric(truth[other], scores[other])))
        count += int(delta >= observed)
    return max((count + 1) / (n_permutations + 1), 1.0 / n_permutations)


def _two_sample_p(subset: np.ndarray, rest: np.ndarray) -> float:
    """Permutation p-value for "this slice differs from everything else"."""
    if subset.size == 0 or rest.size == 0:
        return 1.0
    observed = abs(subset.mean() - rest.mean())
    combined = np.concatenate([subset, rest])
    rng = np.random.default_rng(1337)
    n_permutations = 2_000
    count = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        if abs(combined[: subset.size].mean() - combined[subset.size :].mean()) >= observed:
            count += 1
    return max((count + 1) / (n_permutations + 1), 1.0 / n_permutations)


# ---------------------------------------------------------------------------
# Fairness
# ---------------------------------------------------------------------------
@dataclass
class FairnessReport:
    group_rates: dict[str, float]
    demographic_parity_difference: float
    disparate_impact_ratio: float
    equal_opportunity_difference: float
    equalised_odds_difference: float
    reference_group: str

    def as_row(self) -> dict[str, float | str]:
        return {
            "reference_group": self.reference_group,
            "demographic_parity_diff": round(self.demographic_parity_difference, 4),
            "disparate_impact_ratio": round(self.disparate_impact_ratio, 4),
            "equal_opportunity_diff": round(self.equal_opportunity_difference, 4),
            "equalised_odds_diff": round(self.equalised_odds_difference, 4),
        }


def evaluate_fairness(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    groups: Sequence[Hashable],
) -> FairnessReport:
    """Group fairness metrics for a binary decision.

    Four definitions are reported because they are mathematically incompatible
    and a team must choose which one it is optimising:

    * **Demographic parity** -- equal positive rates. Ignores ground truth, so it
      is the right notion when the label itself is suspected of encoding bias.
    * **Disparate impact ratio** -- the same idea as a ratio; the US EEOC's
      "four-fifths rule" flags values below 0.8.
    * **Equal opportunity** -- equal true positive rates. The usual choice when
      a false negative is the harm (missed loan, missed diagnosis).
    * **Equalised odds** -- equal TPR *and* FPR; the strictest, and generally
      unattainable simultaneously with calibration when base rates differ
      (Kleinberg, Mullainathan & Raghavan, 2017).
    """
    truth = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    group_array = np.asarray(list(groups))

    rates: dict[str, float] = {}
    tprs: dict[str, float] = {}
    fprs: dict[str, float] = {}
    for group in np.unique(group_array):
        mask = group_array == group
        key = str(group)
        rates[key] = float(predicted[mask].mean())
        positives, negatives = mask & (truth == 1), mask & (truth == 0)
        tprs[key] = float(predicted[positives].mean()) if positives.any() else float("nan")
        fprs[key] = float(predicted[negatives].mean()) if negatives.any() else float("nan")

    reference = max(rates, key=rates.get) if rates else ""
    rate_values = list(rates.values())
    tpr_values = [v for v in tprs.values() if not np.isnan(v)]
    fpr_values = [v for v in fprs.values() if not np.isnan(v)]

    max_rate = max(rate_values) if rate_values else 0.0
    min_rate = min(rate_values) if rate_values else 0.0

    return FairnessReport(
        group_rates={k: round(v, 4) for k, v in rates.items()},
        demographic_parity_difference=float(max_rate - min_rate),
        disparate_impact_ratio=float(min_rate / max_rate) if max_rate > 0 else 1.0,
        equal_opportunity_difference=float(max(tpr_values) - min(tpr_values)) if tpr_values else 0.0,
        equalised_odds_difference=float(
            max(
                (max(tpr_values) - min(tpr_values)) if tpr_values else 0.0,
                (max(fpr_values) - min(fpr_values)) if fpr_values else 0.0,
            )
        ),
        reference_group=reference,
    )


# ---------------------------------------------------------------------------
# Automatic error-cluster discovery
# ---------------------------------------------------------------------------
def discover_error_clusters(
    texts: Sequence[str],
    scores: Sequence[float],
    *,
    min_support: int = 5,
    max_terms: int = 15,
    failure_threshold: float = 0.5,
) -> list[dict[str, float | str]]:
    """Find lexical patterns over-represented among failures.

    Predefined slices only catch failures somebody already imagined. This finds
    the ones nobody did: for every token appearing in at least ``min_support``
    items, compare the failure rate with and without it, and rank by lift. It is
    intentionally simple (no embeddings, no clustering) because a term-level
    signal is directly readable by a human and converts straight into a new
    named slice.
    """
    array = np.asarray(scores, dtype=float)
    failures = (array < failure_threshold).astype(float)
    base_rate = float(failures.mean()) if failures.size else 0.0
    if base_rate == 0.0:
        return []

    postings: dict[str, list[int]] = defaultdict(list)
    for index, text in enumerate(texts):
        for token in {t for t in text.lower().split() if len(t) > 3}:
            postings[token].append(index)

    candidates: list[dict[str, float | str]] = []
    for token, indices in postings.items():
        if len(indices) < min_support:
            continue
        subset = failures[np.asarray(indices, dtype=int)]
        rate = float(subset.mean())
        if rate <= base_rate:
            continue
        candidates.append({
            "pattern": token,
            "support": len(indices),
            "failure_rate": round(rate, 4),
            "base_rate": round(base_rate, 4),
            "lift": round(rate / base_rate, 3),
        })

    candidates.sort(key=lambda row: (-float(row["lift"]), -int(row["support"])))
    return candidates[:max_terms]


def metamorphic_consistency(
    original_scores: Sequence[float],
    transformed_scores: Sequence[float],
    *,
    tolerance: float = 0.05,
    relation: Callable[[float, float], bool] | None = None,
) -> dict[str, float]:
    """Consistency rate under a meaning-preserving transformation.

    Metamorphic testing needs no labels, which is what makes it usable on
    production traffic: paraphrase the input, change an irrelevant name, or
    reorder independent clauses, and the score must not move. A consistency rate
    of 0.72 under paraphrase means 28% of your evaluation is measuring surface
    form, not capability.
    """
    original = np.asarray(original_scores, dtype=float)
    transformed = np.asarray(transformed_scores, dtype=float)
    if original.shape != transformed.shape:
        raise ValueError("metamorphic pairs must be aligned")
    check = relation or (lambda a, b: abs(a - b) <= tolerance)
    consistent = np.asarray([check(a, b) for a, b in zip(original, transformed)], dtype=float)
    return {
        "consistency_rate": float(consistent.mean()) if consistent.size else float("nan"),
        "mean_absolute_shift": float(np.mean(np.abs(transformed - original))),
        "max_shift": float(np.max(np.abs(transformed - original))) if original.size else 0.0,
        "n_pairs": int(original.size),
    }
