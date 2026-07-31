"""Statistics must be verifiable independently of any model."""

from __future__ import annotations

import math

import numpy as np
import pytest

from evalcore.stats import (
    benjamini_hochberg,
    bootstrap_interval,
    cliffs_delta,
    cohens_d,
    cohens_kappa,
    mcnemar_test,
    minimum_detectable_effect,
    paired_bootstrap_test,
    required_n_for_proportion,
    wilson_interval,
)


def test_wilson_interval_stays_in_bounds_at_the_extremes():
    """The failure mode Wald has and Wilson does not."""
    perfect = wilson_interval(100, 100)
    assert 0.0 <= perfect.low <= perfect.high <= 1.0
    assert perfect.low < 1.0  # a Wald interval would collapse to [1, 1]

    zero = wilson_interval(0, 50)
    assert zero.low == 0.0
    assert zero.high > 0.0


def test_wilson_matches_published_value():
    interval = wilson_interval(10, 20, confidence=0.95)
    assert interval.estimate == pytest.approx(0.5)
    assert interval.low == pytest.approx(0.2993, abs=1e-3)
    assert interval.high == pytest.approx(0.7007, abs=1e-3)


def test_bootstrap_interval_covers_the_truth():
    rng = np.random.default_rng(0)
    covered = 0
    for seed in range(60):
        sample = rng.binomial(1, 0.7, size=200).astype(float)
        interval = bootstrap_interval(sample, n_resamples=1500, seed=seed)
        covered += interval.contains(0.7)
    assert covered >= 51  # ~95% nominal coverage, tolerant to Monte Carlo noise


def test_bootstrap_degenerate_sample_does_not_raise():
    interval = bootstrap_interval([1.0] * 10)
    assert interval.low == interval.high == 1.0
    assert "degenerate" in interval.method


def test_paired_test_is_more_powerful_than_unpaired():
    """The whole argument for running both systems on the same items."""
    rng = np.random.default_rng(7)
    difficulty = rng.random(150)
    baseline = (difficulty < 0.75).astype(float)
    candidate = (difficulty < 0.82).astype(float)  # perfectly correlated improvement

    paired = paired_bootstrap_test(baseline, candidate)
    assert paired.p_value < 0.05
    assert paired.delta > 0
    assert paired.delta_interval is not None and paired.delta_interval.low > 0


def test_mcnemar_ignores_concordant_pairs():
    baseline = np.array([1] * 90 + [0] * 10)
    candidate = np.array([1] * 90 + [0] * 10)
    assert mcnemar_test(baseline, candidate).p_value == 1.0

    # Same marginal totals, but every disagreement favours the candidate.
    baseline = np.array([1] * 80 + [0] * 20)
    candidate = np.array([1] * 80 + [1] * 15 + [0] * 5)
    result = mcnemar_test(baseline, candidate)
    assert result.p_value < 0.001
    assert result.detail["c_fixed"] == 15
    assert result.detail["b_broken"] == 0


def test_effect_sizes_have_expected_sign_and_scale():
    a = np.zeros(100)
    b = np.ones(100)
    assert cliffs_delta(a, b) == pytest.approx(1.0)
    assert cliffs_delta(b, a) == pytest.approx(-1.0)
    assert cohens_d([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(0.0)


def test_benjamini_hochberg_controls_discoveries():
    all_null = np.linspace(0.05, 0.99, 40)
    assert benjamini_hochberg(all_null, alpha=0.05).sum() <= 1

    with_signal = np.concatenate([[1e-8, 1e-7, 1e-6], np.linspace(0.2, 0.99, 37)])
    assert benjamini_hochberg(with_signal, alpha=0.05).sum() >= 3


def test_power_and_mde_are_mutually_consistent():
    baseline, effect = 0.80, 0.05
    n = required_n_for_proportion(baseline, effect).n_required
    mde = minimum_detectable_effect(n, baseline)
    assert mde == pytest.approx(effect, rel=0.10)


def test_pairing_reduces_required_sample_size():
    unpaired = required_n_for_proportion(0.8, 0.05, paired_correlation=0.0).n_required
    paired = required_n_for_proportion(0.8, 0.05, paired_correlation=0.8).n_required
    assert paired < unpaired / 4


def test_kappa_penalises_a_constant_rater():
    """A judge that always says 'pass' has high raw agreement and zero information."""
    human = ["pass"] * 90 + ["fail"] * 10
    always_pass = ["pass"] * 100
    raw_agreement = sum(h == j for h, j in zip(human, always_pass)) / 100
    assert raw_agreement == pytest.approx(0.90)
    assert cohens_kappa(human, always_pass) == pytest.approx(0.0, abs=1e-9)


def test_kappa_is_one_for_perfect_agreement():
    labels = ["a", "b", "a", "c", "b", "c"]
    assert cohens_kappa(labels, labels) == pytest.approx(1.0)


def test_interval_text_is_readable():
    text = wilson_interval(87, 100).as_text()
    assert "0.87" in text and "95% CI" in text and "n=100" in text


def test_required_n_rejects_impossible_baselines():
    with pytest.raises(ValueError):
        required_n_for_proportion(0.0, 0.05)
    with pytest.raises(ValueError):
        required_n_for_proportion(1.0, 0.05)


def test_mcnemar_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        mcnemar_test([1, 0, 1], [1, 0])


def test_bootstrap_of_median_works():
    values = list(range(1, 101))
    interval = bootstrap_interval(values, statistic=np.median, n_resamples=2000)
    assert math.isclose(interval.estimate, 50.5, rel_tol=1e-9)
    assert interval.low < 50.5 < interval.high
