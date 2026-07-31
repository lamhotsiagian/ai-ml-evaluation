"""Chapter 2 lab — classification, thresholds, calibration, ranking, slices, fairness."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalcore.labdata import (  # noqa: E402
    make_forecast_dataset,
    make_fraud_like_dataset,
    make_ranking_dataset,
)
from evalcore.metrics.calibration import (  # noqa: E402
    TemperatureScaler,
    evaluate_calibration,
    reliability_curve,
)
from evalcore.metrics.classification import (  # noqa: E402
    evaluate_binary,
    pr_points,
    roc_points,
    threshold_for_max_f1,
    threshold_for_min_cost,
    threshold_for_target_recall,
)
from evalcore.metrics.ranking import evaluate_ranking  # noqa: E402
from evalcore.metrics.regression import evaluate_regression, residual_bins  # noqa: E402
from evalcore.metrics.slicing import (  # noqa: E402
    discover_error_clusters,
    evaluate_fairness,
    evaluate_slices,
    evaluate_slices_by_metric,
)
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from ui.components import dataframe, explain, interval_metric, page_header, settings_sidebar  # noqa: E402

page_header("Classical ML Evaluation", 2, "thresholds, calibration, ranking, slices")
settings_sidebar()


@st.cache_data(show_spinner=False)
def load_classification(n: int, rate: float, miscal: float):
    return make_fraud_like_dataset(n_samples=n, positive_rate=rate, miscalibration=miscal)


with st.sidebar:
    st.subheader("Synthetic task")
    n_samples = st.slider("Rows generated", 1000, 8000, 4000, 500)
    positive_rate = st.slider("Positive class rate", 0.01, 0.40, 0.06, 0.01)
    miscalibration = st.slider("Logit sharpening", 1.0, 3.0, 1.8, 0.1,
                               help="Monotonic, so AUC is unchanged and only calibration degrades.")

data = load_classification(n_samples, positive_rate, miscalibration)
st.caption(data.description)

tabs = st.tabs(["Thresholds", "Calibration", "Slices & fairness", "Ranking", "Regression"])

# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("The threshold is a product decision, not a default")
    threshold = st.slider("Decision threshold", 0.01, 0.99, 0.50, 0.01)
    report = evaluate_binary(data.y_true, data.y_score, threshold=threshold)

    cols = st.columns(5)
    interval_metric("Accuracy", report.accuracy)
    cols[1].metric("Precision", f"{report.precision:.4f}")
    cols[2].metric("Recall", f"{report.recall:.4f}")
    cols[3].metric("F1", f"{report.f1:.4f}")
    cols[4].metric("MCC", f"{report.mcc:.4f}",
                   help="Uses all four confusion cells; collapses to 0 for a trivial classifier.")

    a, b = st.columns(2)
    a.metric("ROC-AUC", f"{report.roc_auc:.4f}" if report.roc_auc else "n/a")
    b.metric("PR-AUC", f"{report.pr_auc:.4f}" if report.pr_auc else "n/a",
             help="Moves with prevalence; ROC-AUC does not. Use PR-AUC for rare events.")

    explain(
        f"At {positive_rate:.0%} positives, a model that always predicts 'negative' scores "
        f"{1 - data.positive_rate:.1%} accuracy and 0.0 MCC. That is why accuracy is never the "
        "headline for an imbalanced task."
    )
    dataframe([report.support])

    st.markdown("**Threshold selection strategies**")
    col1, col2 = st.columns(2)
    target_recall = col1.slider("Recall floor the business requires", 0.50, 0.99, 0.85, 0.01)
    cost_ratio = col2.slider("Cost of a false negative relative to a false positive",
                             1.0, 50.0, 20.0, 1.0)

    t_recall, p_at_recall = threshold_for_target_recall(data.y_true, data.y_score, target_recall)
    t_f1, best_f1 = threshold_for_max_f1(data.y_true, data.y_score)
    t_cost, cost = threshold_for_min_cost(data.y_true, data.y_score, cost_fp=1.0, cost_fn=cost_ratio)

    dataframe([
        {"strategy": f"meet recall ≥ {target_recall:.2f}", "threshold": round(t_recall, 4),
         "precision_at_threshold": round(p_at_recall, 4)},
        {"strategy": "maximise F1", "threshold": round(t_f1, 4),
         "precision_at_threshold": round(best_f1, 4)},
        {"strategy": f"minimise cost (FN={cost_ratio:.0f}×FP)", "threshold": round(t_cost, 4),
         "precision_at_threshold": round(cost, 5)},
    ])
    explain("Three defensible thresholds, three different numbers. Optimising F1 by default "
            "silently trades away recall nobody agreed to trade.")

    roc = roc_points(data.y_true, data.y_score)
    pr = pr_points(data.y_true, data.y_score)
    left, right = st.columns(2)
    left.plotly_chart(
        go.Figure([go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name="ROC"),
                   go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line_dash="dash", name="chance")])
        .update_layout(title="ROC", xaxis_title="FPR", yaxis_title="TPR", height=340,
                       margin=dict(l=10, r=10, t=40, b=10)),
        width="stretch")
    right.plotly_chart(
        go.Figure([go.Scatter(x=pr["recall"], y=pr["precision"], mode="lines", name="PR"),
                   go.Scatter(x=[0, 1], y=[data.positive_rate] * 2, mode="lines",
                              line_dash="dash", name="prevalence")])
        .update_layout(title="Precision–Recall", xaxis_title="Recall", yaxis_title="Precision",
                       height=340, margin=dict(l=10, r=10, t=40, b=10)),
        width="stretch")

# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Ranking well is not the same as being calibrated")
    before = evaluate_calibration(data.y_true, data.y_score)
    split = len(data.y_true) // 2
    scaler = TemperatureScaler().fit(data.logits[:split], data.y_true[:split])
    recalibrated = scaler.transform(data.logits)
    after = evaluate_calibration(data.y_true, recalibrated)

    cols = st.columns(4)
    cols[0].metric("ECE before", f"{before.ece:.4f}")
    cols[1].metric("ECE after", f"{after.ece:.4f}", delta=f"{after.ece - before.ece:+.4f}",
                   delta_color="inverse")
    cols[2].metric("Brier before", f"{before.brier:.4f}")
    cols[3].metric("Fitted temperature", f"{scaler.temperature:.3f}")

    explain(
        "Temperature scaling is monotonic, so ROC-AUC, PR-AUC and every ranking metric are "
        "unchanged — only the probabilities move. That is what makes it safe to apply after "
        "a model is already approved on ranking quality."
    )

    curve_before = reliability_curve(data.y_true, data.y_score)
    curve_after = reliability_curve(data.y_true, recalibrated)
    st.plotly_chart(
        go.Figure([
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line_dash="dash", name="perfect"),
            go.Scatter(x=curve_before["mean_confidence"], y=curve_before["observed_rate"],
                       mode="lines+markers", name="before"),
            go.Scatter(x=curve_after["mean_confidence"], y=curve_after["observed_rate"],
                       mode="lines+markers", name="after temperature scaling"),
        ]).update_layout(title="Reliability diagram", xaxis_title="predicted probability",
                         yaxis_title="observed frequency", height=380,
                         margin=dict(l=10, r=10, t=40, b=10)),
        width="stretch")
    dataframe(before.bins)

# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("The aggregate metric hides the failure")
    threshold = st.slider("Threshold", 0.01, 0.99, 0.50, 0.01, key="slice_threshold")
    predictions = (data.y_score >= threshold).astype(int)
    correct = (predictions == data.y_true).astype(float)

    slice_metric = st.radio(
        "Slice on",
        ["ROC-AUC (recomputed per slice)", "PR-AUC (recomputed per slice)",
         "Accuracy (averaged per item)"],
        horizontal=True,
        help="Accuracy can be averaged over items. AUC cannot -- it is defined over a set "
             "and must be recomputed inside each slice.",
    )

    if slice_metric.startswith("Accuracy"):
        slice_report = evaluate_slices(correct, data.slice_tags, min_slice_size=25)
    else:
        scorer = roc_auc_score if slice_metric.startswith("ROC") else average_precision_score
        with st.spinner("Recomputing the metric inside every slice…"):
            slice_report = evaluate_slices_by_metric(
                data.y_true, data.y_score, data.slice_tags,
                metric=scorer, min_slice_size=30,
            )

    st.info(slice_report.summary())
    dataframe([row.as_row() for row in sorted(slice_report.slices, key=lambda s: s.score.estimate)])

    if slice_metric.startswith("Accuracy"):
        st.warning(
            "Nothing is flagged, and that is the finding. At this prevalence, accuracy is "
            "pinned near 1 − base rate inside every slice regardless of model skill, so an "
            "accuracy-sliced dashboard reports a flat line over a model that is close to "
            "random on one segment. Switch to ROC-AUC and look again."
        )
    else:
        explain(
            "The planted failure lives on the channel × region interaction, not on either "
            "attribute alone: the marginal slices are diluted by the segments around them. "
            "Slices overlap, so their p-values are correlated and Benjamini-Hochberg is "
            "applied across them — a 20-slice dashboard without FDR control manufactures a "
            "false alarm on most clean releases."
        )

    st.markdown("**Fairness — four incompatible definitions**")
    fairness = evaluate_fairness(data.y_true, predictions, data.groups)
    dataframe([fairness.as_row()])
    dataframe([{"group": k, "positive_rate": v} for k, v in fairness.group_rates.items()])
    if fairness.disparate_impact_ratio < 0.8:
        st.error(f"Disparate impact ratio {fairness.disparate_impact_ratio:.3f} is below the "
                 "four-fifths rule threshold of 0.80.")

    st.markdown("**Automatic error-cluster discovery**")
    explain("Predefined slices only catch failures somebody already imagined. This ranks "
            "lexical patterns by failure lift so a new slice can be discovered.")
    texts = [" ".join(tags) for tags in data.slice_tags]
    clusters = discover_error_clusters(texts, correct, min_support=20)
    dataframe(clusters or [{"pattern": "none found above base rate", "lift": 0}])

# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Ranking metrics respond differently to the same change")
    col1, col2 = st.columns(2)
    quality = col1.slider("Gold-placement probability", 0.2, 1.0, 0.65, 0.05)
    k = col2.slider("Cutoff K", 1, 10, 5)
    ranking = make_ranking_dataset(n_queries=250, quality=quality)
    report = evaluate_ranking(ranking.results, ranking.gold, k=k,
                              graded_relevance=ranking.graded)
    dataframe([report.as_row()])

    sweep = []
    for q in np.arange(0.2, 1.01, 0.1):
        run = make_ranking_dataset(n_queries=150, quality=float(q))
        row = evaluate_ranking(run.results, run.gold, k=k, graded_relevance=run.graded)
        sweep.append({"quality": round(float(q), 2), "recall@k": row.recall_at_k,
                      "mrr": row.mrr, "ndcg@k": row.ndcg, "hit_rate": row.hit_rate})
    figure = go.Figure()
    for metric in ("recall@k", "mrr", "ndcg@k", "hit_rate"):
        figure.add_scatter(x=[r["quality"] for r in sweep], y=[r[metric] for r in sweep],
                           mode="lines+markers", name=metric)
    figure.update_layout(xaxis_title="retrieval quality", yaxis_title="metric", height=360,
                         margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(figure, width="stretch")
    explain("Hit rate saturates first; MRR keeps moving because it is position sensitive. "
            "Report the one that matches how the downstream system consumes the results.")

# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Regression: the shape of the error matters more than its size")
    regression = make_forecast_dataset(n_samples=3000)
    report = evaluate_regression(regression.y_true, regression.y_pred)
    interval_metric("MAE", report.mae)
    cols = st.columns(4)
    cols[0].metric("RMSE", f"{report.rmse:.3f}")
    cols[1].metric("R²", f"{report.r2:.4f}")
    cols[2].metric("p99 abs error", f"{report.p99_ae:.3f}")
    cols[3].metric("Bias", f"{report.bias:+.3f}")
    explain("RMSE is reported without an interval on purpose: its bootstrap distribution is "
            "dominated by one or two outliers. The p90/p99 absolute errors describe the tail honestly.")

    bins = residual_bins(regression.y_true, regression.y_pred, n_bins=10)
    dataframe(bins)
    st.plotly_chart(
        go.Figure(go.Bar(x=[f"{b['bin']}" for b in bins], y=[b["mae"] for b in bins]))
        .update_layout(title="MAE by target decile — heteroscedasticity check",
                       xaxis_title="target decile", yaxis_title="MAE", height=320,
                       margin=dict(l=10, r=10, t=40, b=10)),
        width="stretch")
