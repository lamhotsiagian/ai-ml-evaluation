"""Chapter 10 lab — the consolidated evaluation dashboard and release scorecard."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalcore.config import get_settings  # noqa: E402
from evalcore.datasets import DatasetRegistry  # noqa: E402
from evalcore.production import evaluate_latency, evaluation_run_cost  # noqa: E402
from evalcore.runner import ExperimentStore  # noqa: E402
from evalcore.stats import wilson_interval  # noqa: E402
from ui.components import dataframe, explain, page_header, settings_sidebar, verdict_banner  # noqa: E402

page_header("Production Evaluation Dashboard", 10, "the release scorecard")
settings_sidebar()

store = ExperimentStore()
registry = DatasetRegistry()
settings = get_settings()

st.markdown(
    "A release scorecard answers one question: **can this ship?** It aggregates the quality, "
    "safety, cost and reliability gates into a single decision and shows exactly which gate "
    "blocked it. Nothing on this page is advisory — every row is a gate someone agreed to."
)

# ---------------------------------------------------------------------------
st.subheader("Release scorecard")

with st.expander("Gate thresholds (these are the negotiated contract, edit them deliberately)",
                 expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    quality_floor = col1.slider("Quality floor", 0.5, 1.0, 0.80, 0.01)
    faithfulness_floor = col2.slider("Faithfulness floor", 0.5, 1.0, 0.90, 0.01)
    asr_ceiling = col3.slider("Attack success ceiling", 0.0, 0.20, 0.05, 0.01)
    latency_budget = col4.slider("p95 latency budget (ms)", 500, 8000, 3000, 100)

runs = store.list_runs(limit=20)

if runs:
    latest = runs[0]
    st.caption(f"Latest run `{latest.run_id}` — {latest.suite}, "
               f"{latest.n_ok}/{latest.n_total} completed, dataset `{latest.dataset_hash}`")
    quality = wilson_interval(latest.n_ok, latest.n_total)
else:
    st.info("No recorded runs. The scorecard below uses the offline reference values so the "
            "gate logic is still inspectable; run page 7 to populate it with live results.")
    quality = wilson_interval(84, 100)

col1, col2 = st.columns(2)
observed_faithfulness = col1.slider("Observed faithfulness", 0.5, 1.0, 0.93, 0.01)
observed_asr = col2.slider("Observed attack success rate", 0.0, 0.30, 0.02, 0.01)

latencies = np.random.default_rng(3).lognormal(mean=6.7, sigma=0.5, size=2000)
latency = evaluate_latency(latencies, slo_ms=float(latency_budget))

gates = [
    {"gate": "Quality (CI lower bound must clear the floor)",
     "observed": round(quality.estimate, 4),
     "ci_low": round(quality.low, 4),
     "threshold": quality_floor,
     "passes": bool(quality.low >= quality_floor)},
    {"gate": "Faithfulness", "observed": observed_faithfulness, "ci_low": None,
     "threshold": faithfulness_floor, "passes": observed_faithfulness >= faithfulness_floor},
    {"gate": "Attack success rate (lower is better)", "observed": observed_asr, "ci_low": None,
     "threshold": asr_ceiling, "passes": observed_asr <= asr_ceiling},
    {"gate": "p95 latency (ms, lower is better)", "observed": round(latency.p95_ms, 1),
     "ci_low": None, "threshold": float(latency_budget),
     "passes": latency.p95_ms <= latency_budget},
    {"gate": "Run completion rate", "observed": round(1 - (latest.error_rate if runs else 0.0), 4),
     "ci_low": None, "threshold": 0.98,
     "passes": (1 - (latest.error_rate if runs else 0.0)) >= 0.98},
]

blocked = [g["gate"] for g in gates if not g["passes"]]
verdict_banner(not blocked,
               "SHIP — every gate passes."
               if not blocked else f"BLOCKED by {len(blocked)} gate(s): " + "; ".join(blocked))
dataframe(gates)
explain("The quality gate tests the confidence-interval lower bound, not the point estimate. "
        "On a 100-case suite those two differ constantly, and the difference is exactly the "
        "set of releases that ship on noise.")

# ---------------------------------------------------------------------------
st.divider()
left, right = st.columns(2)

with left:
    st.subheader("Latency profile")
    dataframe([latency.as_row()])
    st.plotly_chart(
        go.Figure(go.Histogram(x=latencies, nbinsx=50))
        .add_vline(x=latency.p95_ms, line_dash="dash", annotation_text="p95")
        .add_vline(x=float(latency_budget), line_dash="dot", annotation_text="SLO")
        .update_layout(height=320, xaxis_title="latency (ms)",
                       margin=dict(l=10, r=10, t=20, b=10)),
        width="stretch")

with right:
    st.subheader("Continuous-evaluation cost")
    col1, col2 = st.columns(2)
    suite_size = col1.number_input("Suite size", 50, 20000, 800, 50)
    hit_rate = col2.slider("Cache hit rate", 0.0, 0.95, 0.65, 0.05, key="dash_cache")
    cost = evaluation_run_cost(int(suite_size), judge_model=settings.judge_model,
                               cache_hit_rate=hit_rate)
    dataframe([cost])
    st.metric("Monthly cost if run daily", f"${cost['monthly_usd_if_daily']:.2f}")

# ---------------------------------------------------------------------------
st.divider()
st.subheader("Suite coverage")
explain("A dashboard that reports only aggregate quality cannot show you what is *not* being "
        "measured. Coverage is the honest counterpart to the score.")
coverage_rows = []
for name in registry.available():
    dataset = registry.load(name)
    unanswerable = sum(1 for c in dataset if not c.is_answerable)
    coverage_rows.append({
        "dataset": name,
        "cases": len(dataset),
        "hard_cases": dataset.difficulty_counts().get("hard", 0),
        "unanswerable": unanswerable,
        "unanswerable_pct": round(unanswerable / len(dataset), 3) if len(dataset) else 0.0,
        "distinct_slices": len(dataset.slice_counts()),
        "content_hash": dataset.content_hash(),
    })
dataframe(coverage_rows)
for row in coverage_rows:
    if row["unanswerable_pct"] < 0.15 and row["dataset"] == "rag_qa":
        st.warning(f"`{row['dataset']}` has only {row['unanswerable_pct']:.0%} unanswerable "
                   "questions. Below ~15%, abstention behaviour is effectively unmeasured.")

# ---------------------------------------------------------------------------
st.divider()
st.subheader("Run history")
if runs:
    dataframe([{"run_id": r.run_id, "suite": r.suite, "label": r.label or "",
                "started_at": r.started_at, "n_ok": r.n_ok, "n_total": r.n_total,
                "error_rate": round(r.error_rate, 4), "commit": r.git_commit or ""}
               for r in runs])
    st.plotly_chart(
        go.Figure(go.Scatter(x=[r.started_at for r in reversed(runs)],
                             y=[r.n_ok / max(r.n_total, 1) for r in reversed(runs)],
                             mode="lines+markers", name="completion rate"))
        .update_layout(height=300, yaxis_title="completed fraction", yaxis_range=[0, 1.02],
                       margin=dict(l=10, r=10, t=20, b=10)),
        width="stretch")
else:
    st.caption("Run the harness on page 7 to populate the history.")
