"""Chapter 9 lab — dataset registry, experiment store, runner, regression gate."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalcore.config import get_settings  # noqa: E402
from evalcore.datasets import DatasetRegistry, EvalCase, GoldenDataset  # noqa: E402
from evalcore.llm import default_cache  # noqa: E402
from evalcore.report import MetricGate, run_regression_gate  # noqa: E402
from evalcore.runner import ExperimentStore  # noqa: E402
from ui.components import dataframe, explain, page_header, settings_sidebar, verdict_banner  # noqa: E402

page_header("Evaluation Infrastructure", 9, "registry, versioning, tracking, gates")
settings_sidebar()

registry = DatasetRegistry()
store = ExperimentStore()
settings = get_settings()

tabs = st.tabs(["Dataset registry", "Dataset diff", "Experiment store", "Regression gate", "Cache"])

# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Content-addressed golden datasets")
    explain("An evaluation number is meaningless without the dataset version that produced it. "
            "The content hash lets you decompose 'score went 0.81 → 0.88' into 'the system "
            "changed' versus 'the ruler changed'.")
    rows = []
    for name in registry.available():
        dataset = registry.load(name)
        descriptor = dataset.describe()
        rows.append({"dataset": name, "cases": descriptor.n_cases,
                     "content_hash": descriptor.content_hash,
                     "difficulty": str(descriptor.difficulty_counts),
                     "slices": len(descriptor.slice_counts)})
    dataframe(rows)

    selected = st.selectbox("Inspect", registry.available())
    dataset = registry.load(selected)
    st.write("Slice counts:", dataset.slice_counts())
    dataframe([{"case_id": c.case_id, "difficulty": c.difficulty,
                "answerable": c.is_answerable, "slices": ", ".join(c.slice_tags),
                "input": c.input[:90], "expected": (c.expected_output or "")[:70]}
               for c in dataset.cases[:40]], height=420)

    st.markdown("**The hash is order-independent**")
    reversed_dataset = GoldenDataset(dataset.name, list(reversed(dataset.cases)), dataset.version)
    verdict_banner(reversed_dataset.content_hash() == dataset.content_hash(),
                   f"reversed hash `{reversed_dataset.content_hash()}` "
                   f"{'==' if reversed_dataset.content_hash() == dataset.content_hash() else '!='} "
                   f"original `{dataset.content_hash()}`")
    explain("Re-exporting from a database with a different ORDER BY must not look like a new "
            "dataset version.")

# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("What changed between two dataset versions?")
    explain("Attach this report whenever a metric moves and the dataset also moved. Without it, "
            "the two effects are unattributable.")
    base = registry.load(selected)
    edited_cases = [c.model_copy() for c in base.cases[:-2]]
    if edited_cases:
        edited_cases[0] = edited_cases[0].model_copy(update={"input": edited_cases[0].input + " (clarified)"})
    edited_cases.append(EvalCase(case_id="new-001", input="What is the audit export rate limit?",
                                 expected_output="One export per workspace per hour.",
                                 slice_tags=["security", "audit"], difficulty="easy"))
    edited = GoldenDataset(base.name, edited_cases, "v2")

    diff = registry.diff(base, edited)
    cols = st.columns(3)
    cols[0].metric("Added", len(diff["added"]))
    cols[1].metric("Removed", len(diff["removed"]))
    cols[2].metric("Modified", len(diff["modified"]))
    dataframe([{"change": k, "case_ids": ", ".join(v) or "—"} for k, v in diff.items()])
    st.caption(f"v1 hash `{base.content_hash()}` → v2 hash `{edited.content_hash()}` — "
               "runs across these two versions are not comparable.")

# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Experiment store")
    runs = store.list_runs(limit=50)
    if not runs:
        st.info("No runs recorded yet. Run the harness on page 7 to populate the store.")
    else:
        dataframe([{"run_id": r.run_id, "suite": r.suite, "label": r.label or "",
                    "git_commit": r.git_commit or "", "started_at": r.started_at,
                    "n_ok": r.n_ok, "n_total": r.n_total,
                    "error_rate": round(r.error_rate, 3),
                    "dataset_hash": r.dataset_hash,
                    "fingerprint": r.settings_fingerprint} for r in runs])

        suites = sorted({r.suite for r in runs})
        suite = st.selectbox("Suite", suites)
        metric = st.text_input("Metric to chart", "contains_expected")
        history = store.metric_history(suite, metric)
        if history:
            figure = go.Figure(go.Scatter(
                x=[h["started_at"] for h in history], y=[h["value"] for h in history],
                mode="lines+markers", error_y=dict(
                    type="data", symmetric=False,
                    array=[(h["ci_high"] or h["value"]) - h["value"] for h in history],
                    arrayminus=[h["value"] - (h["ci_low"] or h["value"]) for h in history])))
            figure.update_layout(height=340, yaxis_title=metric,
                                 margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(figure, width="stretch")
            dataframe(history)
        else:
            st.caption(f"No recorded history for `{metric}` in `{suite}`.")

# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("The regression gate")
    explain("Three rules: refuse to compare incomparable runs; test the difference rather than "
            "eyeballing it; and name the cases that broke, not just the fact that something did.")

    runs = store.list_runs(limit=50)
    if len(runs) < 2:
        st.info("Need at least two recorded runs. Run the harness on page 7 twice.")
    else:
        col1, col2 = st.columns(2)
        baseline_run = col1.selectbox("Baseline run", runs, format_func=lambda r: r.run_id)
        candidate_run = col2.selectbox("Candidate run", runs, index=min(1, len(runs) - 1),
                                       format_func=lambda r: r.run_id)
        metric_name = st.text_input("Metric", "contains_expected", key="gate_metric")
        col3, col4 = st.columns(2)
        budget = col3.slider("Regression budget", 0.0, 0.20, 0.02, 0.005)
        floor = col4.slider("Absolute floor", 0.0, 1.0, 0.60, 0.05)

        comparable = baseline_run.comparable_with(candidate_run)
        report = run_regression_gate(
            {metric_name: store.case_scores(baseline_run.run_id, metric_name)},
            {metric_name: store.case_scores(candidate_run.run_id, metric_name)},
            [MetricGate(metric_name, floor=floor, max_regression=budget, binary=True)],
            comparable=comparable,
            baseline_run_id=baseline_run.run_id,
            candidate_run_id=candidate_run.run_id,
            incomparable_reason="dataset hash or settings fingerprint differ between the runs",
            min_cases=5,
        )
        verdict_banner(report.decision == "pass", report.summary())
        if report.findings:
            dataframe([f.as_row() for f in report.findings])
            for finding in report.findings:
                if finding.regressed_cases:
                    st.error("Newly failing: " + ", ".join(finding.regressed_cases[:20]))
                if finding.fixed_cases:
                    st.success("Newly passing: " + ", ".join(finding.fixed_cases[:20]))
        st.code(report.markdown(), language="markdown")
        st.caption(f"Exit code {report.exit_code}")

# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Response cache")
    explain("Judge calls are pure functions of (model, prompt, params). Caching them across the "
            "UI, pytest and CI is what makes per-commit continuous evaluation affordable.")
    cache = default_cache()
    dataframe([cache.stats()])
    st.write("Store path:", str(settings.store_path))
    st.write("Chroma path:", str(settings.chroma_dir))
    if st.button("Clear cache", type="secondary"):
        st.success(f"Deleted {cache.clear()} cached responses.")
