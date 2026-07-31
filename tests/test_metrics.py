"""Metric implementations must match hand-computed values."""

from __future__ import annotations

import numpy as np
import pytest

from evalcore.metrics.calibration import TemperatureScaler, evaluate_calibration
from evalcore.metrics.classification import (
    evaluate_binary,
    threshold_for_min_cost,
    threshold_for_target_recall,
)
from evalcore.metrics.ranking import (
    average_precision,
    evaluate_ranking,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    reciprocal_rank_fusion,
)
from evalcore.metrics.regression import evaluate_regression, quantile_loss, residual_bins
from evalcore.metrics.slicing import evaluate_fairness, evaluate_slices, metamorphic_consistency


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def test_binary_metrics_match_hand_computation():
    y_true = [1, 1, 1, 0, 0, 0, 0, 0]
    y_score = [0.9, 0.8, 0.3, 0.7, 0.2, 0.1, 0.05, 0.4]
    report = evaluate_binary(y_true, y_score, threshold=0.5)
    # TP=2 (0.9, 0.8), FN=1 (0.3), FP=1 (0.7), TN=4
    assert report.support == {"tp": 2, "fn": 1, "fp": 1, "tn": 4}
    assert report.precision == pytest.approx(2 / 3)
    assert report.recall == pytest.approx(2 / 3)
    assert report.f1 == pytest.approx(2 / 3)


def test_accuracy_is_useless_under_heavy_imbalance():
    y_true = [0] * 990 + [1] * 10
    always_negative = [0.0] * 1000
    report = evaluate_binary(y_true, always_negative, threshold=0.5)
    assert report.accuracy.estimate == pytest.approx(0.99)
    assert report.recall == 0.0
    assert report.mcc == pytest.approx(0.0)  # MCC exposes what accuracy hides


def test_threshold_for_target_recall_meets_the_floor():
    rng = np.random.default_rng(0)
    y_true = rng.binomial(1, 0.3, 500)
    y_score = np.clip(y_true * 0.4 + rng.random(500) * 0.6, 0, 1)
    threshold, _ = threshold_for_target_recall(y_true, y_score, 0.90)
    achieved = evaluate_binary(y_true, y_score, threshold=threshold).recall
    assert achieved >= 0.90 - 1e-9


def test_cost_optimal_threshold_moves_with_the_cost_ratio():
    """Overlapping score distributions, so the threshold has somewhere to move."""
    rng = np.random.default_rng(1)
    y_true = rng.binomial(1, 0.2, 2000)
    # Positives are shifted but heavily overlapping with negatives.
    y_score = np.clip(rng.normal(0.5 + 0.18 * y_true, 0.20), 0.001, 0.999)
    cheap_fn, _ = threshold_for_min_cost(y_true, y_score, cost_fp=1.0, cost_fn=1.0)
    costly_fn, _ = threshold_for_min_cost(y_true, y_score, cost_fp=1.0, cost_fn=30.0)
    assert costly_fn < cheap_fn  # expensive false negatives push the threshold down


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def test_perfect_calibration_has_near_zero_ece():
    rng = np.random.default_rng(3)
    probs = rng.random(6000)
    outcomes = (rng.random(6000) < probs).astype(int)
    assert evaluate_calibration(outcomes, probs, n_bins=10).ece < 0.03


def test_temperature_scaling_improves_ece_without_touching_ranking():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(4)
    logits = rng.normal(0, 1, 4000)
    y = (rng.random(4000) < 1 / (1 + np.exp(-logits))).astype(int)
    overconfident = 1 / (1 + np.exp(-logits * 2.5))

    before = evaluate_calibration(y, overconfident)
    scaler = TemperatureScaler().fit(logits * 2.5, y)
    after = evaluate_calibration(y, scaler.transform(logits * 2.5))

    assert after.ece < before.ece
    assert roc_auc_score(y, overconfident) == pytest.approx(
        roc_auc_score(y, scaler.transform(logits * 2.5)), abs=1e-9
    )


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
def test_ranking_primitives_match_hand_computation():
    retrieved = ["a", "x", "b", "y", "z"]
    relevant = {"a", "b", "c"}
    assert recall_at_k(retrieved, relevant, 5) == pytest.approx(2 / 3)
    assert precision_at_k(retrieved, relevant, 5) == pytest.approx(2 / 5)
    assert reciprocal_rank(retrieved, relevant) == pytest.approx(1.0)
    # AP = (1/1 + 2/3) / 3
    assert average_precision(retrieved, relevant) == pytest.approx((1.0 + 2 / 3) / 3)


def test_ndcg_ideal_uses_the_full_relevance_map():
    """Computing ideal DCG over retrieved docs only inflates NDCG; guard against it."""
    retrieved = ["low"]
    relevance = {"low": 1.0, "high": 3.0}
    assert ndcg_at_k(retrieved, relevance, 1) < 0.2


def test_ndcg_is_one_for_the_ideal_ranking():
    relevance = {"a": 3.0, "b": 2.0, "c": 1.0}
    assert ndcg_at_k(["a", "b", "c"], relevance, 3) == pytest.approx(1.0)


def test_mrr_is_position_sensitive_where_recall_is_not():
    gold = [{"g"}, {"g"}]
    early = [["g", "x", "y"], ["g", "x", "y"]]
    late = [["x", "y", "g"], ["x", "y", "g"]]
    early_report = evaluate_ranking(early, gold, k=3)
    late_report = evaluate_ranking(late, gold, k=3)
    assert early_report.recall_at_k == late_report.recall_at_k
    assert early_report.mrr > late_report.mrr


def test_rrf_promotes_documents_ranked_well_by_both_retrievers():
    dense = ["a", "b", "c"]
    lexical = ["c", "a", "d"]
    assert reciprocal_rank_fusion([dense, lexical])[0] == "a"


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
def test_regression_report_matches_hand_computation():
    report = evaluate_regression([10.0, 20.0, 30.0], [12.0, 18.0, 33.0])
    # errors = pred - true = [+2, -2, +3]
    assert report.mae.estimate == pytest.approx((2 + 2 + 3) / 3)
    assert report.rmse == pytest.approx(np.sqrt((4 + 4 + 9) / 3))
    assert report.bias == pytest.approx((2 - 2 + 3) / 3)
    assert report.median_ae == pytest.approx(2.0)


def test_mape_is_none_when_a_target_is_zero():
    assert evaluate_regression([0.0, 10.0], [1.0, 11.0]).mape is None


def test_residual_bins_expose_heteroscedasticity():
    rng = np.random.default_rng(9)
    truth = np.linspace(1, 100, 800)
    predictions = truth + rng.normal(0, 1, 800) * truth * 0.1
    bins = residual_bins(truth, predictions, n_bins=5)
    assert bins[-1]["mae"] > bins[0]["mae"] * 3


def test_quantile_loss_prefers_the_requested_quantile():
    rng = np.random.default_rng(2)
    truth = rng.normal(100, 20, 4000)
    p90 = float(np.quantile(truth, 0.9))
    median = float(np.median(truth))
    assert quantile_loss(truth, [p90] * 4000, 0.9) < quantile_loss(truth, [median] * 4000, 0.9)


# ---------------------------------------------------------------------------
# Slices and fairness
# ---------------------------------------------------------------------------
def test_slice_evaluation_finds_a_planted_weak_slice():
    scores = [1.0] * 200 + [0.0] * 60
    tags = [["strong"]] * 200 + [["weak"]] * 60
    report = evaluate_slices(scores, tags, min_slice_size=20)
    worst = report.worst
    assert worst is not None and worst.name == "weak"
    assert worst.flagged


def test_set_level_slice_evaluation_finds_what_accuracy_slicing_misses():
    """AUC must be recomputed per slice; averaging per-item accuracy hides the failure."""
    from sklearn.metrics import roc_auc_score

    from evalcore.labdata import make_fraud_like_dataset
    from evalcore.metrics.slicing import evaluate_slices_by_metric

    data = make_fraud_like_dataset(n_samples=4000, positive_rate=0.06)

    # Per-item accuracy is pinned near 1 - prevalence in every slice, so the
    # planted interaction failure is invisible.
    correct = ((data.y_score >= 0.5).astype(int) == data.y_true).astype(float)
    by_accuracy = evaluate_slices(correct, data.slice_tags, min_slice_size=25)
    assert not by_accuracy.flagged

    # Recomputing AUC inside each slice exposes it.
    by_auc = evaluate_slices_by_metric(
        data.y_true, data.y_score, data.slice_tags,
        metric=roc_auc_score, min_slice_size=30,
    )
    worst = by_auc.worst
    assert worst is not None
    assert worst.name == "segment:mobile-apac"
    assert worst.score.estimate < 0.60          # near-random on the weak segment
    assert by_auc.overall.estimate > 0.75       # healthy overall
    assert worst.flagged


def test_set_level_slice_intervals_are_reported():
    from sklearn.metrics import roc_auc_score

    from evalcore.labdata import make_fraud_like_dataset
    from evalcore.metrics.slicing import evaluate_slices_by_metric

    data = make_fraud_like_dataset(n_samples=1500, positive_rate=0.10)
    report = evaluate_slices_by_metric(
        data.y_true, data.y_score, data.slice_tags,
        metric=roc_auc_score, min_slice_size=30, n_resamples=120,
    )
    assert report.slices
    for row in report.slices:
        assert row.score.low <= row.score.estimate <= row.score.high


def test_fairness_flags_a_four_fifths_violation():
    y_true = [1] * 100 + [0] * 100
    y_pred = [1] * 80 + [0] * 20 + [1] * 20 + [0] * 80
    groups = ["a"] * 100 + ["b"] * 100
    report = evaluate_fairness(y_true, y_pred, groups)
    assert report.disparate_impact_ratio < 0.8
    assert report.demographic_parity_difference > 0.3


def test_metamorphic_consistency_detects_paraphrase_sensitivity():
    original = [1.0] * 100
    transformed = [1.0] * 72 + [0.0] * 28
    result = metamorphic_consistency(original, transformed, tolerance=0.05)
    assert result["consistency_rate"] == pytest.approx(0.72)
