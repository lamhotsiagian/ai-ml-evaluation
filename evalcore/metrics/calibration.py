"""Calibration measurement and post-hoc recalibration.

A model can rank perfectly (AUC 0.95) and still be badly calibrated: when it
says 0.9 it might be right 60% of the time. Ranking quality is what you need for
sorting; calibration is what you need the moment a probability enters an
arithmetic decision -- expected loss, a routing threshold, or an escalation
policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression


@dataclass
class CalibrationReport:
    ece: float
    mce: float
    brier: float
    log_loss: float
    bins: list[dict[str, float]]
    n_bins: int
    strategy: str

    def as_row(self) -> dict[str, float | str]:
        return {
            "ece": round(self.ece, 4),
            "mce": round(self.mce, 4),
            "brier": round(self.brier, 4),
            "log_loss": round(self.log_loss, 4),
            "bins": self.n_bins,
            "strategy": self.strategy,
        }


def _bin_edges(scores: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    if strategy == "quantile":
        edges = np.quantile(scores, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        return np.unique(edges)
    return np.linspace(0.0, 1.0, n_bins + 1)


def evaluate_calibration(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    *,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "quantile",
) -> CalibrationReport:
    """Expected and maximum calibration error plus proper scoring rules.

    ``strategy="quantile"`` is the default and matters more than it sounds.
    Uniform bins on a model whose scores cluster near 0 and 1 leave the middle
    bins nearly empty, so ECE is computed from a handful of points and swings
    wildly between runs. Equal-mass bins give every bin the same weight of
    evidence.
    """
    truth = np.asarray(y_true, dtype=float)
    probs = np.clip(np.asarray(y_prob, dtype=float), 1e-12, 1 - 1e-12)

    edges = _bin_edges(probs, n_bins, strategy)
    bins: list[dict[str, float]] = []
    ece, mce = 0.0, 0.0
    total = truth.size

    for i in range(len(edges) - 1):
        low, high = edges[i], edges[i + 1]
        mask = (probs >= low) & (probs < high if i < len(edges) - 2 else probs <= high)
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(probs[mask].mean())
        observed = float(truth[mask].mean())
        gap = abs(confidence - observed)
        ece += (count / total) * gap
        mce = max(mce, gap)
        bins.append({
            "bin": i, "low": float(low), "high": float(high), "n": count,
            "mean_confidence": round(confidence, 4),
            "observed_rate": round(observed, 4),
            "gap": round(confidence - observed, 4),
        })

    brier = float(np.mean((probs - truth) ** 2))
    logloss = float(-np.mean(truth * np.log(probs) + (1 - truth) * np.log(1 - probs)))
    return CalibrationReport(ece, mce, brier, logloss, bins, len(bins), strategy)


class TemperatureScaler:
    """Single-parameter recalibration fitted on held-out logits.

    Temperature scaling is the right first attempt because it has exactly one
    parameter: it cannot overfit a small validation set, and being monotonic it
    leaves AUC and every ranking metric untouched. If it does not fix the
    miscalibration, the problem is shape, not sharpness -- move to isotonic.
    """

    def __init__(self) -> None:
        self.temperature: float = 1.0
        self._fitted = False

    def fit(self, logits: Sequence[float], y_true: Sequence[int]) -> "TemperatureScaler":
        z = np.asarray(logits, dtype=float)
        y = np.asarray(y_true, dtype=float)

        def negative_log_likelihood(temperature: float) -> float:
            if temperature <= 0:
                return 1e12
            p = 1.0 / (1.0 + np.exp(-z / temperature))
            p = np.clip(p, 1e-12, 1 - 1e-12)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

        result = minimize_scalar(negative_log_likelihood, bounds=(0.01, 100.0), method="bounded")
        self.temperature = float(result.x)
        self._fitted = True
        return self

    def transform(self, logits: Sequence[float]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TemperatureScaler.fit must be called before transform")
        z = np.asarray(logits, dtype=float)
        return 1.0 / (1.0 + np.exp(-z / self.temperature))


class IsotonicCalibrator:
    """Non-parametric monotone recalibration.

    More expressive than temperature scaling and able to fix S-shaped
    miscalibration, at the cost of needing several hundred held-out points and
    producing a step function that can be brittle in the tails. Fit it on a
    *separate* calibration split -- fitting on the test set makes ECE look
    perfect and means nothing.
    """

    def __init__(self) -> None:
        self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._fitted = False

    def fit(self, y_prob: Sequence[float], y_true: Sequence[int]) -> "IsotonicCalibrator":
        self._model.fit(np.asarray(y_prob, dtype=float), np.asarray(y_true, dtype=float))
        self._fitted = True
        return self

    def transform(self, y_prob: Sequence[float]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("IsotonicCalibrator.fit must be called before transform")
        return np.asarray(self._model.predict(np.asarray(y_prob, dtype=float)), dtype=float)


def reliability_curve(
    y_true: Sequence[int], y_prob: Sequence[float], *, n_bins: int = 10
) -> dict[str, list[float]]:
    """Points for a reliability diagram (the plot the UI renders)."""
    report = evaluate_calibration(y_true, y_prob, n_bins=n_bins)
    return {
        "mean_confidence": [b["mean_confidence"] for b in report.bins],
        "observed_rate": [b["observed_rate"] for b in report.bins],
        "count": [b["n"] for b in report.bins],
    }
