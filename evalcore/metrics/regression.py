"""Regression metrics, error decomposition and residual diagnostics.

The headline number for a regression model is almost never the interesting
part. What ships or blocks a model is the *shape* of its error: whether it is
biased, whether variance grows with the target, and which slice carries the tail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from evalcore.stats import Interval, bootstrap_interval


@dataclass
class RegressionReport:
    mae: Interval
    rmse: float
    r2: float
    mape: float | None
    smape: float
    median_ae: float
    bias: float
    p90_ae: float
    p99_ae: float
    n: int

    def as_row(self) -> dict[str, float]:
        return {
            "mae": round(self.mae.estimate, 4),
            "mae_low": round(self.mae.low, 4),
            "mae_high": round(self.mae.high, 4),
            "rmse": round(self.rmse, 4),
            "r2": round(self.r2, 4),
            "mape": round(self.mape, 4) if self.mape is not None else float("nan"),
            "smape": round(self.smape, 4),
            "median_ae": round(self.median_ae, 4),
            "bias": round(self.bias, 4),
            "p90_ae": round(self.p90_ae, 4),
            "p99_ae": round(self.p99_ae, 4),
            "n": self.n,
        }


def evaluate_regression(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    confidence: float = 0.95,
    mape_epsilon: float = 1e-8,
) -> RegressionReport:
    """Regression report with tail percentiles and an explicit bias term.

    Design notes:

    * **MAE carries the interval, RMSE does not.** RMSE is a sum of squares and
      its bootstrap distribution is dominated by one or two outliers, so an
      interval on it is unstable and misleading. Report RMSE as a point estimate
      alongside p90/p99 absolute error, which describes the tail honestly.
    * **MAPE is returned as ``None`` when any target is near zero** rather than
      returning an enormous meaningless number. MAPE explodes near zero and
      penalises over-prediction and under-prediction asymmetrically; sMAPE is
      always computed as the safe alternative.
    """
    truth = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if truth.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    errors = predicted - truth
    absolute = np.abs(errors)

    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    near_zero = np.any(np.abs(truth) < mape_epsilon)
    mape = None if near_zero else float(np.mean(absolute / np.abs(truth)))

    denominator = (np.abs(truth) + np.abs(predicted)) / 2
    smape = float(np.mean(np.where(denominator > 0, absolute / np.maximum(denominator, 1e-12), 0.0)))

    return RegressionReport(
        mae=bootstrap_interval(absolute, confidence=confidence, n_resamples=5_000),
        rmse=float(np.sqrt(np.mean(errors**2))),
        r2=r2,
        mape=mape,
        smape=smape,
        median_ae=float(np.median(absolute)),
        bias=float(errors.mean()),
        p90_ae=float(np.quantile(absolute, 0.90)),
        p99_ae=float(np.quantile(absolute, 0.99)),
        n=int(truth.size),
    )


def residual_bins(
    y_true: Sequence[float], y_pred: Sequence[float], *, n_bins: int = 10
) -> list[dict[str, float]]:
    """Error broken down by target quantile -- the heteroscedasticity check.

    A model with flat MAE across bins is well behaved. A model whose MAE triples
    in the top decile is a model that will fail on exactly the high-value cases
    the business cares about, while its aggregate MAE looks fine.
    """
    truth = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if truth.size == 0:
        return []
    edges = np.quantile(truth, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    rows: list[dict[str, float]] = []
    for i in range(len(edges) - 1):
        low, high = edges[i], edges[i + 1]
        mask = (truth >= low) & (truth <= high if i == len(edges) - 2 else truth < high)
        if not mask.any():
            continue
        errors = predicted[mask] - truth[mask]
        rows.append({
            "bin": i,
            "target_low": float(low),
            "target_high": float(high),
            "n": int(mask.sum()),
            "mae": float(np.mean(np.abs(errors))),
            "bias": float(errors.mean()),
            "rmse": float(np.sqrt(np.mean(errors**2))),
        })
    return rows


def quantile_loss(y_true: Sequence[float], y_pred: Sequence[float], quantile: float) -> float:
    """Pinball loss -- the correct metric when a model predicts a quantile.

    Scoring a p90 forecast with MAE rewards the model for predicting the median,
    which is precisely the wrong behaviour for capacity planning or inventory.
    """
    if not 0 < quantile < 1:
        raise ValueError("quantile must be in (0, 1)")
    truth = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    diff = truth - predicted
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))
