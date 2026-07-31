"""Judge calibration: proving the judge agrees with humans before trusting it.

An uncalibrated judge is a random number generator with good manners. This
module implements the meta-evaluation loop -- score the judge against a human
panel, measure chance-corrected agreement, quantify the biases, and produce a
go/no-go decision -- so that "we use an LLM judge" becomes a claim with evidence
attached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from evalcore.stats import Interval, cohens_kappa, krippendorff_alpha_nominal, wilson_interval


@dataclass
class JudgeCalibrationReport:
    """Meta-evaluation of a judge against human labels."""

    n: int
    raw_agreement: Interval
    kappa: float
    spearman: float
    mean_bias: float
    human_positive_rate: float
    judge_positive_rate: float
    false_pass_rate: float
    false_fail_rate: float
    per_slice_kappa: dict[str, float] = field(default_factory=dict)

    @property
    def deployable(self) -> bool:
        """Ship gate for a judge.

        Thresholds follow the conventional reading of kappa (Landis & Koch,
        1977): below 0.6 agreement is "moderate", which is not enough to gate a
        release. The false-pass constraint is separate and stricter, because a
        judge that waves through bad output is worse than no judge -- it
        launders a failure into a green dashboard.
        """
        return self.kappa >= 0.6 and self.false_pass_rate <= 0.10

    def verdict_text(self) -> str:
        status = "DEPLOYABLE" if self.deployable else "NOT DEPLOYABLE"
        return (
            f"{status}: kappa={self.kappa:.3f}, raw agreement={self.raw_agreement.estimate:.3f} "
            f"{self.raw_agreement.as_text(3)}, false-pass={self.false_pass_rate:.3f}, "
            f"bias={self.mean_bias:+.3f} (judge minus human), n={self.n}"
        )


def calibrate_judge(
    human_scores: Sequence[float],
    judge_scores: Sequence[float],
    *,
    pass_threshold: float = 3.0,
    slice_tags: Sequence[Sequence[str]] | None = None,
) -> JudgeCalibrationReport:
    """Compare judge scores with human scores on the same items.

    Args:
        human_scores: Human panel scores (1-5, or the panel median).
        judge_scores: Judge scores on the same items and scale.
        pass_threshold: Score at or above which an item counts as a pass. Kappa
            is computed on this binarised decision because that is the decision
            the judge is actually automating.
        slice_tags: Optional per-item tags, so agreement can be reported per
            slice -- judges routinely agree with humans on English and disagree
            badly on code or on a second language.
    """
    human = np.asarray(human_scores, dtype=float)
    judge = np.asarray(judge_scores, dtype=float)
    if human.shape != judge.shape:
        raise ValueError("human and judge scores must be item-aligned")
    if human.size == 0:
        raise ValueError("calibration requires at least one labelled item")

    human_pass = (human >= pass_threshold).astype(int)
    judge_pass = (judge >= pass_threshold).astype(int)
    agreement = (human_pass == judge_pass).astype(float)

    false_pass = int(np.sum((human_pass == 0) & (judge_pass == 1)))
    false_fail = int(np.sum((human_pass == 1) & (judge_pass == 0)))
    n_human_fail = int(np.sum(human_pass == 0)) or 1
    n_human_pass = int(np.sum(human_pass == 1)) or 1

    per_slice: dict[str, float] = {}
    if slice_tags is not None:
        buckets: dict[str, list[int]] = {}
        for index, tags in enumerate(slice_tags):
            for tag in tags:
                buckets.setdefault(tag, []).append(index)
        for tag, indices in sorted(buckets.items()):
            if len(indices) < 10:
                continue
            try:
                per_slice[tag] = cohens_kappa(
                    human_pass[indices].tolist(), judge_pass[indices].tolist()
                )
            except ValueError:
                continue

    return JudgeCalibrationReport(
        n=int(human.size),
        raw_agreement=wilson_interval(int(agreement.sum()), agreement.size),
        kappa=cohens_kappa(human_pass.tolist(), judge_pass.tolist()),
        spearman=_spearman(human, judge),
        mean_bias=float(np.mean(judge - human)),
        human_positive_rate=float(human_pass.mean()),
        judge_positive_rate=float(judge_pass.mean()),
        false_pass_rate=false_pass / n_human_fail,
        false_fail_rate=false_fail / n_human_pass,
        per_slice_kappa=per_slice,
    )


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation without importing SciPy for one call."""
    if a.size < 2:
        return float("nan")
    rank_a = np.argsort(np.argsort(a)).astype(float)
    rank_b = np.argsort(np.argsort(b)).astype(float)
    if rank_a.std() == 0 or rank_b.std() == 0:
        return 0.0
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


@dataclass
class BiasProbeReport:
    """Quantified judge biases, each measured with a controlled manipulation."""

    position_bias: float
    verbosity_bias: float
    self_preference: float
    formatting_bias: float
    n_probes: int

    def worst(self) -> tuple[str, float]:
        pairs = {
            "position": self.position_bias,
            "verbosity": self.verbosity_bias,
            "self-preference": self.self_preference,
            "formatting": self.formatting_bias,
        }
        name = max(pairs, key=lambda k: abs(pairs[k]))
        return name, pairs[name]

    def as_row(self) -> dict[str, float]:
        return {
            "position_bias": round(self.position_bias, 4),
            "verbosity_bias": round(self.verbosity_bias, 4),
            "self_preference": round(self.self_preference, 4),
            "formatting_bias": round(self.formatting_bias, 4),
            "n_probes": self.n_probes,
        }


def measure_position_bias(forward_winners: Sequence[str], reverse_winners: Sequence[str]) -> float:
    """Fraction of comparisons whose winner flips when the order is swapped.

    0.0 is a positionally unbiased judge. Values above ~0.15 mean a meaningful
    share of your pairwise results are an artefact of presentation order, and
    single-order evaluation must be abandoned.
    """
    if not forward_winners:
        return 0.0
    flips = sum(
        1 for forward, reverse in zip(forward_winners, reverse_winners)
        if forward != reverse and "tie" not in (forward, reverse)
    )
    return flips / len(forward_winners)


def measure_verbosity_bias(
    short_scores: Sequence[float], padded_scores: Sequence[float]
) -> float:
    """Mean score lift from padding a response with no added information.

    The probe: take a correct short answer, append a paragraph that restates it
    without new content, and re-judge. Any positive lift is the judge paying for
    length. Left unmeasured, this bias rewards models for producing longer
    answers, and the next fine-tune dutifully learns to be verbose.
    """
    short = np.asarray(short_scores, dtype=float)
    padded = np.asarray(padded_scores, dtype=float)
    if short.size == 0 or short.shape != padded.shape:
        return 0.0
    return float(np.mean(padded - short))


def measure_self_preference(
    own_family_scores: Sequence[float], other_family_scores: Sequence[float]
) -> float:
    """Score gap on responses from the judge's own model family, quality held equal.

    Requires the two response sets to be matched on human-rated quality;
    otherwise this measures a real quality difference rather than a bias.
    """
    own = np.asarray(own_family_scores, dtype=float)
    other = np.asarray(other_family_scores, dtype=float)
    if own.size == 0 or other.size == 0:
        return 0.0
    return float(own.mean() - other.mean())


def panel_agreement(rating_matrix: np.ndarray) -> float:
    """Krippendorff's alpha across a multi-rater panel (NaN = not rated).

    Report this next to judge-human kappa. If three humans only reach alpha 0.45
    with each other, a judge scoring kappa 0.55 against their median is already
    at the ceiling of the task, and the honest fix is a clearer rubric, not a
    better judge.
    """
    return krippendorff_alpha_nominal(rating_matrix)
