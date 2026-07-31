"""Chapter 1 lab — intervals, power, splits and leakage."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalcore.datasets import DatasetRegistry  # noqa: E402
from evalcore.splits import (  # noqa: E402
    class_balance,
    detect_leakage,
    grouped_split,
    hash_split,
    stratified_split,
    temporal_split,
)
from evalcore.stats import (  # noqa: E402
    benjamini_hochberg,
    bootstrap_interval,
    mcnemar_test,
    minimum_detectable_effect,
    paired_bootstrap_test,
    required_n_for_proportion,
    wilson_interval,
)
from ui.components import dataframe, explain, interval_metric, page_header, settings_sidebar  # noqa: E402

page_header("Evaluation Foundations", 1, "intervals, power, splits, leakage")
settings_sidebar()

tab_power, tab_intervals, tab_compare, tab_splits, tab_leakage = st.tabs(
    ["Power & sample size", "Intervals", "Comparing two systems", "Splits", "Leakage audit"]
)

# ---------------------------------------------------------------------------
with tab_power:
    st.subheader("How big does the evaluation set need to be?")
    explain(
        "Run this before writing a single test case. If the MDE is larger than the "
        "improvement you are arguing about, the suite cannot settle the argument."
    )
    col1, col2, col3 = st.columns(3)
    baseline = col1.slider("Baseline pass rate", 0.05, 0.95, 0.80, 0.01)
    mde = col2.slider("Improvement to detect", 0.01, 0.20, 0.05, 0.01)
    correlation = col3.slider("Paired correlation ρ", 0.0, 0.95, 0.0, 0.05,
                              help="Correlation between the two systems' per-item outcomes. "
                                   "Pairing multiplies the required n by (1-ρ).")

    analysis = required_n_for_proportion(baseline, mde, paired_correlation=correlation)
    unpaired = required_n_for_proportion(baseline, mde, paired_correlation=0.0)

    a, b, c = st.columns(3)
    a.metric("Cases needed per arm", f"{analysis.n_required:,}")
    b.metric("Unpaired equivalent", f"{unpaired.n_required:,}",
             delta=f"{analysis.n_required - unpaired.n_required:+,}")
    c.metric("Saving from pairing", f"{1 - analysis.n_required / unpaired.n_required:.0%}")

    st.markdown("**Minimum detectable effect by suite size**")
    sizes = np.array([25, 50, 100, 200, 400, 800, 1600, 3200])
    mdes = [minimum_detectable_effect(int(n), baseline) for n in sizes]
    figure = go.Figure(go.Scatter(x=sizes, y=mdes, mode="lines+markers", name="MDE"))
    figure.add_hline(y=mde, line_dash="dash", annotation_text=f"target {mde:.0%}")
    figure.update_layout(xaxis_type="log", xaxis_title="cases per arm",
                         yaxis_title="minimum detectable effect", height=340,
                         margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(figure, width="stretch")
    dataframe([{"n_per_arm": int(n), "mde": round(v, 4), "detects_target": bool(v <= mde)}
               for n, v in zip(sizes, mdes)])

# ---------------------------------------------------------------------------
with tab_intervals:
    st.subheader("Every score needs an interval")
    col1, col2 = st.columns(2)
    n = col1.slider("Cases evaluated", 10, 1000, 100, 10)
    passes = col2.slider("Cases passed", 0, n, min(int(n * 0.87), n))

    wilson = wilson_interval(passes, n)
    bootstrap = bootstrap_interval(np.array([1.0] * passes + [0.0] * (n - passes)),
                                   n_resamples=5000)
    left, right = st.columns(2)
    with left:
        interval_metric("Wilson score interval", wilson,
                        help_text="Correct for proportions; stays inside [0,1] at the extremes.")
    with right:
        interval_metric("Bootstrap BCa interval", bootstrap,
                        help_text="Works for any statistic, not only proportions.")

    explain(
        f"With {n} cases the interval is ±{wilson.half_width:.3f}. Any reported improvement "
        f"smaller than {2 * wilson.half_width:.3f} is indistinguishable from noise at this size."
    )

    st.markdown("**Interval width shrinks with the square root of n**")
    grid = np.array([20, 40, 80, 160, 320, 640, 1280])
    widths = [wilson_interval(int(round(g * passes / n)), int(g)).half_width for g in grid]
    st.plotly_chart(
        go.Figure(go.Bar(x=[str(int(g)) for g in grid], y=widths)).update_layout(
            xaxis_title="cases", yaxis_title="CI half-width", height=300,
            margin=dict(l=10, r=10, t=20, b=10)),
        width="stretch",
    )

# ---------------------------------------------------------------------------
with tab_compare:
    st.subheader("Is B actually better than A?")
    explain(
        "Both systems are scored on the SAME items, and the test looks at the per-item "
        "difference. Unpaired comparison wastes most of your power on item difficulty."
    )
    col1, col2, col3 = st.columns(3)
    n_cases = col1.slider("Shared cases", 30, 500, 120, 10, key="cmp_n")
    rate_a = col2.slider("A pass rate", 0.3, 0.99, 0.78, 0.01)
    rate_b = col3.slider("B pass rate", 0.3, 0.99, 0.84, 0.01)
    shared_difficulty = st.slider("Item-difficulty correlation", 0.0, 0.95, 0.7, 0.05)

    rng = np.random.default_rng(1337)
    difficulty = rng.random(n_cases)
    def _draw(rate: float) -> np.ndarray:
        threshold = difficulty * shared_difficulty + rng.random(n_cases) * (1 - shared_difficulty)
        return (threshold < rate).astype(float)

    scores_a, scores_b = _draw(rate_a), _draw(rate_b)

    paired = paired_bootstrap_test(scores_a, scores_b)
    mcnemar = mcnemar_test(scores_a.astype(int), scores_b.astype(int))

    left, right = st.columns(2)
    left.metric("Observed delta", f"{scores_b.mean() - scores_a.mean():+.4f}")
    right.metric("Discordant pairs", int(mcnemar.detail.get("discordant", 0)),
                 help="Only items where the two systems disagree carry evidence.")
    dataframe([
        {"test": paired.name, "p_value": round(paired.p_value, 5),
         "effect": round(paired.effect_size, 3), "effect_name": paired.effect_name,
         "significant@0.05": paired.significant()},
        {"test": mcnemar.name, "p_value": round(mcnemar.p_value, 5),
         "effect": round(mcnemar.effect_size, 3), "effect_name": mcnemar.effect_name,
         "significant@0.05": mcnemar.significant()},
    ])
    if paired.delta_interval:
        st.info(f"Delta CI: {paired.delta_interval.as_text()}")

    st.markdown("**Multiple comparisons**")
    explain("Testing 40 metrics at α=0.05 produces two false alarms per clean release. "
            "Benjamini-Hochberg controls the false discovery rate instead.")
    n_metrics = st.slider("Metrics on the dashboard", 5, 80, 40, 5)
    p_values = rng.random(n_metrics)  # all null by construction
    naive = int((p_values < 0.05).sum())
    controlled = int(benjamini_hochberg(p_values).sum())
    a, b = st.columns(2)
    a.metric("False alarms, naive α=0.05", naive)
    b.metric("False alarms, BH-controlled", controlled)

# ---------------------------------------------------------------------------
with tab_splits:
    st.subheader("Splitting strategies")
    registry = DatasetRegistry()
    dataset = registry.load("rag_qa")
    labels = [case.difficulty for case in dataset]
    groups = [case.slice_tags[0] if case.slice_tags else "none" for case in dataset]

    strategy = st.radio("Strategy", ["hash", "stratified", "grouped", "temporal"], horizontal=True)
    if strategy == "hash":
        split = hash_split([c.case_id for c in dataset])
    elif strategy == "stratified":
        split = stratified_split(labels)
    elif strategy == "grouped":
        split = grouped_split(groups)
    else:
        split = temporal_split(list(range(len(dataset))), embargo=1.0)

    st.write(split.sizes)
    st.write("Test-set difficulty balance:",
             {k: round(v, 3) for k, v in class_balance([labels[i] for i in split.test]).items()})
    explain(
        "Stratified keeps rare classes present in every partition. Grouped keeps every row of a "
        "conversation on one side. Temporal is mandatory whenever inputs drift. Hash is stable "
        "as the dataset grows, so yesterday's test scores stay comparable with tomorrow's."
    )

# ---------------------------------------------------------------------------
with tab_leakage:
    st.subheader("Leakage audit — run this as a gate, not a report")
    default_train = (
        "How much does the Growth plan cost per month?\n"
        "What is the overage rate on Starter?\n"
        "Explain the circuit breaker policy for workspaces."
    )
    default_test = (
        "how much does the growth plan cost per month\n"
        "What is the Growth plan monthly cost per month?\n"
        "How long are raw events retained on Scale?"
    )
    col1, col2 = st.columns(2)
    train_text = col1.text_area("Train items (one per line)", default_train, height=160)
    test_text = col2.text_area("Test items (one per line)", default_test, height=160)
    threshold = st.slider("Near-duplicate Jaccard threshold", 0.3, 0.95, 0.6, 0.05)

    train_rows = [line for line in train_text.splitlines() if line.strip()]
    test_rows = [line for line in test_text.splitlines() if line.strip()]
    report = detect_leakage(train_rows, test_rows, near_duplicate_threshold=threshold, shingle_k=3)

    (st.success if report.clean else st.error)(report.summary())
    if report.exact_duplicates:
        dataframe([{"train_index": i, "test_index": j, "kind": "exact"}
                   for i, j in report.exact_duplicates])
    if report.near_duplicates:
        dataframe([{"train_index": i, "test_index": j, "jaccard": s}
                   for i, j, s in report.near_duplicates])
