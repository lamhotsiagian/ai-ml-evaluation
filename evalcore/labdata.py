"""Reproducible synthetic datasets for the classical-ML labs.

These are *generated*, not fabricated: every array comes from a seeded
scikit-learn generator or an explicit noise model, so any reader running the
same seed gets the same numbers and can verify every metric in the book by
recomputation. Nothing here is a hand-written "example output".

The generators are shaped to expose the failures the chapters discuss: class
imbalance, an over-confident classifier, a slice that underperforms, and
heteroscedastic regression error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split


@dataclass
class ClassificationLabData:
    """A fitted binary classifier and its held-out predictions."""

    y_true: np.ndarray
    y_score: np.ndarray
    logits: np.ndarray
    slice_tags: list[list[str]]
    groups: np.ndarray
    positive_rate: float
    description: str


def make_fraud_like_dataset(
    *,
    n_samples: int = 4000,
    positive_rate: float = 0.06,
    miscalibration: float = 1.8,
    weak_slice_penalty: float = 0.55,
    seed: int = 1337,
) -> ClassificationLabData:
    """An imbalanced, deliberately miscalibrated classifier with a weak slice.

    Three properties are engineered in, each corresponding to a lesson:

    * ``positive_rate`` well below 0.5, so accuracy is useless and PR-AUC and
      MCC do the work;
    * ``miscalibration`` sharpening the logits, so the model ranks well
      (AUC unchanged, since sharpening is monotonic) while its probabilities are
      badly over-confident -- exactly the situation temperature scaling fixes;
    * ``weak_slice_penalty`` degrading the signal for one subgroup, so slice
      evaluation finds a failure the aggregate metric hides.
    """
    rng = np.random.default_rng(seed)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=18,
        n_informative=7,
        n_redundant=4,
        n_clusters_per_class=3,
        weights=[1 - positive_rate, positive_rate],
        flip_y=0.02,
        class_sep=1.1,
        random_state=seed,
    )

    # Three overlapping slice dimensions, assigned independently of the label so
    # any slice gap the lab finds comes from the model, not from the sampler.
    channel = rng.choice(["web", "mobile", "api"], size=n_samples, p=[0.45, 0.40, 0.15])
    region = rng.choice(["emea", "amer", "apac"], size=n_samples, p=[0.4, 0.4, 0.2])
    tenure = rng.choice(["new", "established"], size=n_samples, p=[0.3, 0.7])

    # Two planted degradations, at two different scales.
    #
    # 1. A *marginal* one on channel:mobile (~40% of rows). Large enough that a
    #    one-dimensional slice table has the statistical power to flag it, which
    #    is what makes the slice lab demonstrate something rather than merely
    #    report noise.
    # 2. A stronger *interaction* one on mobile x apac (~8% of rows). Its
    #    marginals are diluted by the segments around them, so it is much harder
    #    to see -- which is precisely the failure that motivates interaction
    #    slices and automatic error-cluster discovery in Chapter 2.
    X = X.copy()

    def _degrade(mask: np.ndarray, retained: float) -> None:
        """Blend the masked rows' features toward noise.

        Every column is degraded, not just the informative ones: scikit-learn's
        generator emits redundant columns that are linear combinations of the
        informative ones, so degrading only the first block leaves the model an
        intact copy of the signal and the planted slice never appears.
        """
        n_rows, n_cols = int(mask.sum()), X.shape[1]
        if n_rows == 0:
            return
        X[mask] = (X[mask] * retained
                   + rng.normal(0, 1 - retained, size=(n_rows, n_cols)))

    _degrade(channel == "mobile", weak_slice_penalty)
    _degrade((channel == "mobile") & (region == "apac"), weak_slice_penalty * 0.5)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(n_samples), test_size=0.35, stratify=y, random_state=seed
    )
    model = LogisticRegression(max_iter=2000, class_weight=None)
    model.fit(X_train, y_train)

    logits = model.decision_function(X_test) * miscalibration
    scores = 1.0 / (1.0 + np.exp(-logits))

    # Each row carries its three marginal tags plus the channel x region
    # interaction, so the UI can show that the marginals stay flat while the
    # interaction is where the model actually fails.
    tags = [
        [
            f"channel:{channel[i]}",
            f"region:{region[i]}",
            f"tenure:{tenure[i]}",
            f"segment:{channel[i]}-{region[i]}",
        ]
        for i in idx_test
    ]
    return ClassificationLabData(
        y_true=y_test,
        y_score=scores,
        logits=logits,
        slice_tags=tags,
        groups=region[idx_test],
        positive_rate=float(y_test.mean()),
        description=(
            f"Synthetic fraud-like task: n={len(y_test)} held-out rows, "
            f"{y_test.mean():.1%} positive, logits sharpened x{miscalibration} to induce "
            "over-confidence, mobile+apac slice degraded."
        ),
    )


@dataclass
class RegressionLabData:
    y_true: np.ndarray
    y_pred: np.ndarray
    description: str


def make_forecast_dataset(
    *, n_samples: int = 3000, heteroscedastic: bool = True, seed: int = 1337
) -> RegressionLabData:
    """A regression task whose error grows with the target.

    Heteroscedasticity is the point. Aggregate MAE looks acceptable while error
    in the top decile is several times larger, which is what the residual-bin
    diagnostic in :mod:`evalcore.metrics.regression` exists to surface.
    """
    rng = np.random.default_rng(seed)
    X, y = make_regression(n_samples=n_samples, n_features=12, n_informative=6,
                           noise=8.0, random_state=seed)
    y = y - y.min() + 50.0  # shift to a positive scale so MAPE is defined

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.35, random_state=seed
    )
    model = LinearRegression().fit(X_train, y_train)
    predictions = model.predict(X_test)

    if heteroscedastic:
        spread = float(np.ptp(y_test))
        scale = 0.04 * (y_test - y_test.min()) / max(spread, 1e-9)
        predictions = predictions + rng.normal(0, 1, y_test.size) * scale * y_test

    return RegressionLabData(
        y_true=y_test,
        y_pred=predictions,
        description=(
            f"Synthetic forecast task: n={len(y_test)} held-out rows"
            + (", error variance grows with the target." if heteroscedastic else ".")
        ),
    )


@dataclass
class RankingLabData:
    results: list[list[str]]
    gold: list[set[str]]
    graded: list[dict[str, float]]
    description: str


def make_ranking_dataset(
    *, n_queries: int = 200, depth: int = 10, quality: float = 0.65, seed: int = 1337
) -> RankingLabData:
    """A retrieval run with controllable quality, for the ranking-metric lab.

    ``quality`` is the probability that a gold document is placed in the ranked
    list at all; position is then drawn from a geometric distribution so good
    documents cluster near the top. Sweeping ``quality`` shows how recall@k,
    MRR and NDCG respond differently to the same underlying change, which is the
    lesson of the ranking section.
    """
    rng = np.random.default_rng(seed)
    results, gold, graded = [], [], []

    for query in range(n_queries):
        n_gold = int(rng.integers(1, 4))
        gold_ids = {f"q{query}-gold-{i}" for i in range(n_gold)}
        distractors = [f"q{query}-noise-{i}" for i in range(depth * 2)]

        ranked: list[str] = []
        for doc_id in sorted(gold_ids):
            if rng.random() < quality:
                position = min(int(rng.geometric(0.45)) - 1, depth - 1)
                ranked.insert(min(position, len(ranked)), doc_id)
        rng.shuffle(distractors)
        for doc_id in distractors:
            if len(ranked) >= depth:
                break
            ranked.append(doc_id)

        results.append(ranked[:depth])
        gold.append(gold_ids)
        graded.append({doc_id: float(rng.integers(2, 4)) for doc_id in gold_ids})

    return RankingLabData(
        results, gold, graded,
        f"Synthetic retrieval run: {n_queries} queries, depth {depth}, gold-placement probability {quality:.2f}.",
    )


def make_drift_windows(
    *, n: int = 2000, shift: float = 0.0, kind: Literal["numeric", "categorical"] = "numeric",
    seed: int = 1337,
):
    """Reference and current monitoring windows with a controllable shift.

    Sweeping ``shift`` from 0 upward and watching where PSI and KS first fire is
    the fastest way to calibrate alert thresholds for a real system -- and to
    discover that KS fires on shifts far too small to matter at production
    volumes.
    """
    rng = np.random.default_rng(seed)
    if kind == "numeric":
        reference = rng.gamma(shape=2.0, scale=1.0, size=n)
        current = rng.gamma(shape=2.0 + shift, scale=1.0 + shift * 0.3, size=n)
        return reference, current

    categories = ["billing", "auth", "integration", "reporting", "other"]
    base = np.array([0.35, 0.25, 0.20, 0.15, 0.05])
    moved = base + np.array([-shift, shift * 0.6, shift * 0.4, 0.0, 0.0])
    moved = np.clip(moved, 0.01, None)
    moved = moved / moved.sum()
    reference = rng.choice(categories, size=n, p=base)
    current = rng.choice(categories, size=n, p=moved)
    return list(reference), list(current)
