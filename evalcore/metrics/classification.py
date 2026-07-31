"""Classification metrics with intervals, thresholds and operating points.

The functions here wrap scikit-learn rather than reimplementing it -- the value
added is the *decision layer*: every metric ships with a confidence interval,
and the threshold selectors optimise the quantity a product actually cares
about instead of the default 0.5 cut nobody chose deliberately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from evalcore.stats import Interval, bootstrap_interval, wilson_interval


@dataclass
class ClassificationReport:
    """Threshold-dependent and threshold-free metrics in one object."""

    threshold: float
    accuracy: Interval
    precision: float
    recall: float
    f1: float
    mcc: float
    roc_auc: float | None
    pr_auc: float | None
    support: dict[str, int]
    confusion: np.ndarray = field(default_factory=lambda: np.zeros((2, 2), dtype=int))

    def as_row(self) -> dict[str, float | str]:
        return {
            "threshold": round(self.threshold, 4),
            "accuracy": round(self.accuracy.estimate, 4),
            "accuracy_ci": f"[{self.accuracy.low:.3f}, {self.accuracy.high:.3f}]",
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "mcc": round(self.mcc, 4),
            "roc_auc": round(self.roc_auc, 4) if self.roc_auc is not None else float("nan"),
            "pr_auc": round(self.pr_auc, 4) if self.pr_auc is not None else float("nan"),
            "n": sum(self.support.values()),
        }


def evaluate_binary(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    threshold: float = 0.5,
    confidence: float = 0.95,
) -> ClassificationReport:
    """Full binary classification report at a chosen operating threshold.

    Two deliberate choices:

    * **MCC is always reported.** On a 99:1 imbalance, accuracy and even F1 can
      look healthy while the model is close to useless; MCC uses all four
      confusion-matrix cells and collapses to zero for a trivial classifier.
    * **PR-AUC accompanies ROC-AUC.** ROC-AUC is insensitive to prevalence -- it
      barely moves when the positive class goes from 10% to 0.1% -- which makes
      it the wrong headline for rare-event detection. PR-AUC moves.
    """
    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    predictions = (scores >= threshold).astype(int)
    correct = (predictions == truth).astype(float)

    both_classes = len(np.unique(truth)) > 1
    matrix = confusion_matrix(truth, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    return ClassificationReport(
        threshold=threshold,
        accuracy=wilson_interval(int(correct.sum()), correct.size, confidence),
        precision=float(precision_score(truth, predictions, zero_division=0)),
        recall=float(recall_score(truth, predictions, zero_division=0)),
        f1=float(f1_score(truth, predictions, zero_division=0)),
        mcc=float(matthews_corrcoef(truth, predictions)) if both_classes else 0.0,
        roc_auc=float(roc_auc_score(truth, scores)) if both_classes else None,
        pr_auc=float(average_precision_score(truth, scores)) if both_classes else None,
        support={"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        confusion=matrix,
    )


def threshold_for_target_recall(
    y_true: Sequence[int], y_score: Sequence[float], target_recall: float
) -> tuple[float, float]:
    """Cheapest threshold meeting a recall floor; returns (threshold, precision).

    This is how a threshold is chosen in a regulated or safety-critical setting:
    the product fixes the recall the business will accept (e.g. "we must catch
    98% of fraudulent transactions") and evaluation reports the precision cost
    of meeting it. Optimising F1 instead silently trades away recall the
    business never agreed to trade.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns one more point than thresholds.
    feasible = np.nonzero(recall[:-1] >= target_recall)[0]
    if feasible.size == 0:
        return 0.0, float(precision[0])
    best = feasible[np.argmax(precision[:-1][feasible])]
    return float(thresholds[best]), float(precision[best])


def threshold_for_max_f1(y_true: Sequence[int], y_score: Sequence[float]) -> tuple[float, float]:
    """Threshold maximising F1, with the achieved F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)
    best = int(np.argmax(f1[:-1])) if thresholds.size else 0
    return (float(thresholds[best]) if thresholds.size else 0.5), float(f1[best])


def threshold_for_min_cost(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    cost_fp: float,
    cost_fn: float,
) -> tuple[float, float]:
    """Threshold minimising expected cost; returns (threshold, cost per item).

    The most honest formulation of a threshold choice, and the one that survives
    contact with a product manager: state the two error costs explicitly and let
    the arithmetic pick the cut. A 20:1 FN:FP cost ratio produces a very
    different threshold from the 0.5 default, and the gap is pure money.
    """
    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    candidates = np.unique(np.concatenate([[0.0], scores, [1.0]]))
    costs = []
    for cut in candidates:
        predicted = (scores >= cut).astype(int)
        fp = int(np.sum((predicted == 1) & (truth == 0)))
        fn = int(np.sum((predicted == 0) & (truth == 1)))
        costs.append((fp * cost_fp + fn * cost_fn) / truth.size)
    best = int(np.argmin(costs))
    return float(candidates[best]), float(costs[best])


def roc_points(y_true: Sequence[int], y_score: Sequence[float]) -> dict[str, list[float]]:
    """FPR/TPR/threshold triples for plotting an ROC curve in the UI."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    thresholds = np.where(np.isinf(thresholds), 1.0, thresholds)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "threshold": thresholds.tolist()}


def pr_points(y_true: Sequence[int], y_score: Sequence[float]) -> dict[str, list[float]]:
    """Precision/recall pairs for plotting a PR curve in the UI."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "threshold": np.append(thresholds, 1.0).tolist(),
    }


def macro_metrics_multiclass(
    y_true: Sequence[str], y_pred: Sequence[str], *, confidence: float = 0.95
) -> dict[str, float | Interval]:
    """Macro-averaged multiclass metrics with a bootstrap interval on accuracy.

    Macro rather than micro averaging by default: micro-F1 on an imbalanced
    label set is dominated by the majority class and hides exactly the rare-class
    collapse an evaluation is supposed to catch.
    """
    truth = np.asarray(list(y_true))
    predicted = np.asarray(list(y_pred))
    correct = (truth == predicted).astype(float)
    return {
        "accuracy": bootstrap_interval(correct, confidence=confidence, n_resamples=5_000),
        "macro_precision": float(precision_score(truth, predicted, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(truth, predicted, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, predicted, average="weighted", zero_division=0)),
    }
