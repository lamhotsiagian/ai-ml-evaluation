"""Chapter 8 lab — drift, A/B, sequential testing, canary rollout, cost."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalcore.labdata import make_drift_windows  # noqa: E402
from evalcore.production import (  # noqa: E402
    CanaryController,
    Configuration,
    DriftDashboard,
    SequentialTest,
    analyse_ab_test,
    assign_variant,
    chi_square_drift,
    embedding_drift,
    evaluate_latency,
    evaluation_run_cost,
    ks_drift,
    pareto_frontier,
    population_stability_index,
    sample_size_for_experiment,
)
from ui.components import dataframe, explain, page_header, settings_sidebar, verdict_banner  # noqa: E402

page_header("Production AI Evaluation", 8, "drift, online experiments, canaries, cost")
settings_sidebar()

tabs = st.tabs(["Drift", "A/B test", "Sequential monitoring", "Canary", "Cost & latency"])

# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Four surfaces drift; each needs a different detector")
    shift = st.slider("Injected distribution shift", 0.0, 1.0, 0.3, 0.05)

    numeric_ref, numeric_cur = make_drift_windows(shift=shift, kind="numeric")
    cat_ref, cat_cur = make_drift_windows(shift=shift * 0.3, kind="categorical")

    rng = np.random.default_rng(7)
    ref_emb = rng.normal(0, 1, size=(300, 32))
    cur_emb = rng.normal(shift * 0.4, 1, size=(300, 32))

    dashboard = (DriftDashboard()
                 .add(population_stability_index(numeric_ref, numeric_cur))
                 .add(ks_drift(numeric_ref, numeric_cur))
                 .add(chi_square_drift(cat_ref, cat_cur))
                 .add(embedding_drift(ref_emb, cur_emb)))

    dataframe(dashboard.as_rows())
    verdict_banner(not dashboard.should_page(),
                   "No page: evidence is not correlated across detectors."
                   if not dashboard.should_page()
                   else f"PAGE ON-CALL: worst severity {dashboard.worst_severity}")
    explain("Single-detector alerting generates enough noise that the rota stops reading it. "
            "Paging on one critical or two simultaneous alerts is a far better predictor of a "
            "genuine incident.")

    st.plotly_chart(
        go.Figure([go.Histogram(x=numeric_ref, name="reference", opacity=0.6, nbinsx=40),
                   go.Histogram(x=numeric_cur, name="current", opacity=0.6, nbinsx=40)])
        .update_layout(barmode="overlay", height=320, title="Numeric feature distributions",
                       margin=dict(l=10, r=10, t=40, b=10)),
        width="stretch")

    st.markdown("**Where does each detector first fire?**")
    sweep = []
    for s in np.arange(0.0, 1.01, 0.1):
        ref, cur = make_drift_windows(shift=float(s), kind="numeric")
        sweep.append({"shift": round(float(s), 2),
                      "psi": round(population_stability_index(ref, cur).statistic, 4),
                      "ks_statistic": round(ks_drift(ref, cur).statistic, 4),
                      "ks_p": round(ks_drift(ref, cur).p_value or 1.0, 6)})
    dataframe(sweep)
    explain("KS fires far earlier than PSI. At production volumes that sensitivity is a "
            "liability: it flags shifts too small to affect any user.")

# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("A/B test with a practical-significance gate")
    col1, col2, col3 = st.columns(3)
    baseline_rate = col1.slider("Control rate", 0.1, 0.95, 0.72, 0.01)
    treatment_rate = col2.slider("Treatment rate", 0.1, 0.95, 0.75, 0.01)
    n_per_arm = col3.slider("Users per arm", 200, 20000, 4000, 200)
    practical = st.slider("Minimum lift worth shipping", 0.0, 0.10, 0.02, 0.005)
    guardrail = st.checkbox("Guardrail metric violated (e.g. p95 latency budget blown)")

    rng = np.random.default_rng(11)
    control = (rng.random(n_per_arm) < baseline_rate).astype(float)
    treatment = (rng.random(n_per_arm) < treatment_rate).astype(float)
    result = analyse_ab_test(control, treatment, practical_threshold=practical,
                             guardrail_violated=guardrail)

    verdict_banner(result.decision == "ship", f"{result.decision.upper()} — {result.reason}")
    dataframe([result.as_row()])
    explain("A statistically significant +0.2% that doubles latency is not a shipping decision. "
            "The confidence interval must clear the practical threshold entirely, not merely "
            "exclude zero.")

    st.markdown("**Sizing the experiment before running it**")
    daily = st.number_input("Daily eligible traffic", 100, 1_000_000, 5000, 500)
    sizing = sample_size_for_experiment(baseline_rate, practical or 0.01, daily_traffic=int(daily))
    dataframe([sizing])

    st.markdown("**Deterministic, sticky assignment**")
    explain("Hashing on (experiment, unit) means the same user always sees the same variant, "
            "two concurrent experiments stay independent, and no assignment database sits on "
            "the request path.")
    dataframe([{"user_id": f"user-{i}",
                "variant": assign_variant(f"user-{i}", "judge-rubric-v3")} for i in range(8)])

# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Always-valid sequential testing")
    explain("A fixed-horizon p-value is only valid if you look once. Teams look daily, and "
            "daily peeking at α=0.05 pushes the real false-positive rate above 20%. The mSPRT "
            "holds α however often you peek, which is what makes automated rollback safe.")

    col1, col2 = st.columns(2)
    true_lift = col1.slider("True lift (0 = null is true)", -0.05, 0.10, 0.02, 0.005)
    batches = col2.slider("Daily batches", 5, 60, 25)
    batch_size = st.slider("Observations per arm per batch", 50, 2000, 400, 50)

    rng = np.random.default_rng(23)
    test = SequentialTest(alpha=0.05)
    history, naive_flags = [], 0
    for day in range(1, batches + 1):
        control_batch = (rng.random(batch_size) < 0.70).astype(float)
        treatment_batch = (rng.random(batch_size) < 0.70 + true_lift).astype(float)
        state = test.update(control_batch, treatment_batch)
        fixed = analyse_ab_test(control_batch, treatment_batch)
        naive_flags += int(fixed.p_value < 0.05)
        history.append({"day": day, "llr": round(state.log_likelihood_ratio, 4),
                        "decision": state.decision,
                        "n_per_arm": state.n_control,
                        "naive_p_today": round(fixed.p_value, 4)})

    dataframe(history)
    left, right = st.columns(2)
    left.metric("Sequential decision", history[-1]["decision"])
    right.metric("Times a naive daily p-value fired", naive_flags,
                 help="Under a true null, this is the false-positive count that peeking buys you.")

    st.plotly_chart(
        go.Figure([
            go.Scatter(x=[h["day"] for h in history], y=[h["llr"] for h in history],
                       mode="lines+markers", name="log likelihood ratio"),
        ]).add_hline(y=np.log(1 / 0.05), line_dash="dash", annotation_text="reject null")
        .add_hline(y=-np.log(1 / 0.05), line_dash="dash", annotation_text="accept null")
        .update_layout(height=340, xaxis_title="day", margin=dict(l=10, r=10, t=20, b=10)),
        width="stretch")

# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Canary rollout with automatic rollback")
    controller = CanaryController()
    col1, col2 = st.columns(2)
    error_rate = col1.slider("Observed error rate", 0.0, 0.10, 0.01, 0.005)
    latency_p95 = col2.slider("Observed p95 latency (ms)", 500, 8000, 2200, 100)
    col3, col4 = st.columns(2)
    quality = col3.slider("Candidate quality", 0.5, 1.0, 0.86, 0.01)
    baseline_quality = col4.slider("Baseline quality", 0.5, 1.0, 0.87, 0.01)
    observations = st.slider("Observations at this stage", 50, 5000, 800, 50)

    rows = []
    for _ in range(len(CanaryController.LADDER)):
        decision = controller.evaluate(
            n_observations=observations, error_rate=error_rate,
            latency_p95_ms=float(latency_p95), quality_score=quality,
            baseline_quality=baseline_quality)
        rows.append({"action": decision.action, "traffic_percent": decision.traffic_percent,
                     "reason": decision.reason})
        if decision.action in ("rollback",) or decision.traffic_percent >= 100:
            break
    dataframe(rows)
    explain("The ladder exists so that a catastrophic change is seen by 1% of traffic, not 50%. "
            "The controller refuses to promote before a stage has enough observations to detect "
            "the regression it is watching for.")

# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Cost, latency and the Pareto frontier")
    latencies = np.random.default_rng(5).lognormal(mean=6.6, sigma=0.55, size=3000)
    latency_report = evaluate_latency(latencies, slo_ms=3000)
    dataframe([latency_report.as_row()])
    explain("Percentiles, not the mean. LLM latency is heavily right skewed: a mean of 900ms "
            "routinely hides a p99 of 8 seconds, and the p99 is what a user on a bad day gets.")

    st.markdown("**What does the evaluation suite itself cost?**")
    col1, col2, col3 = st.columns(3)
    n_cases = col1.number_input("Cases in the suite", 10, 20000, 500, 10)
    samples = col2.slider("Self-consistency samples", 1, 5, 1)
    hit_rate = col3.slider("Cache hit rate", 0.0, 0.95, 0.6, 0.05)
    dataframe([evaluation_run_cost(int(n_cases), judge_model="gemini-2.0-flash-lite",
                                   n_judge_samples=samples, cache_hit_rate=hit_rate)])
    explain("Evaluation cost is a real line item and the reason continuous evaluation gets "
            "cancelled six weeks in. Cache hit rate and self-consistency sampling are the two "
            "levers that dominate it.")

    st.markdown("**Pareto frontier over quality, cost and latency**")
    configurations = [
        Configuration("flash-lite, k=3, no rerank", 0.78, 0.35, 900),
        Configuration("flash, k=5, no rerank", 0.84, 0.90, 1400),
        Configuration("flash, k=5, rerank", 0.88, 2.60, 3100),
        Configuration("flash, k=10, rerank", 0.89, 4.80, 4600),
        Configuration("flash-lite, k=5, rerank", 0.83, 1.20, 2400),
        Configuration("flash, k=3, no rerank", 0.82, 0.70, 1150),
    ]
    frontier = {c.name for c in pareto_frontier(configurations)}
    dataframe([{"configuration": c.name, "quality": c.quality,
                "usd_per_1k": c.cost_per_1k_usd, "p95_ms": c.latency_p95_ms,
                "on_frontier": c.name in frontier} for c in configurations])
    st.plotly_chart(
        go.Figure(go.Scatter(
            x=[c.cost_per_1k_usd for c in configurations],
            y=[c.quality for c in configurations],
            mode="markers+text", text=[c.name for c in configurations], textposition="top center",
            marker=dict(size=[c.latency_p95_ms / 120 for c in configurations],
                        color=["#1F6FEB" if c.name in frontier else "#B0B7C3"
                               for c in configurations]),
        )).update_layout(height=420, xaxis_title="USD per 1k requests", yaxis_title="quality",
                         title="Marker size = p95 latency; blue = on the frontier",
                         margin=dict(l=10, r=10, t=50, b=10)),
        width="stretch")
    explain("Everything off the frontier can be discarded without argument. Choosing within it "
            "is a product decision, not an engineering one.")
