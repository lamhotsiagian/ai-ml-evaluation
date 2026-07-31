"""Chapter 7 lab — the framework-agnostic suite, metrics, assertions and CI output."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalcore.datasets import DatasetRegistry, EvalCase  # noqa: E402
from evalcore.harness import (  # noqa: E402
    Assertion,
    EvaluationSuite,
    Metric,
    contains_metric,
    exact_match_metric,
    json_valid_metric,
    judge_metric,
)
from evalcore.judge import BUILTIN_RUBRICS, RubricJudge  # noqa: E402
from evalcore.llm import build_chat_model  # noqa: E402
from evalcore.runner import ExperimentStore  # noqa: E402
from ui.components import (  # noqa: E402
    dataframe,
    explain,
    page_header,
    require_api_key,
    settings_sidebar,
    verdict_banner,
)

page_header("Evaluation Harness", 7, "metrics, suites, assertions, CI gates")
settings_sidebar()

st.markdown(
    "Frameworks differ in surface area but share one abstraction: a **metric** is a pure "
    "function of (case, output); a **suite** binds metrics to a dataset and a target; an "
    "**assertion** turns aggregates into a build decision. Owning that abstraction is what "
    "makes framework choice reversible."
)

registry = DatasetRegistry()

tabs = st.tabs(["Build a suite", "Assertions & CI", "Framework comparison"])

# ---------------------------------------------------------------------------
with tabs[0]:
    dataset_name = st.selectbox("Dataset", registry.available(), index=0)
    dataset = registry.load(dataset_name)
    limit = st.slider("Cases", 3, len(dataset), min(8, len(dataset)))
    subset = type(dataset)(dataset.name, dataset.cases[:limit], dataset.version)

    st.caption(f"`{subset.name}` — {len(subset)} cases, content hash `{subset.content_hash()}`")

    chosen = st.multiselect(
        "Metrics",
        ["contains_expected", "exact_match", "json_valid", "judge_score", "non_empty"],
        default=["contains_expected", "non_empty"],
    )
    rubric_name = st.selectbox("Judge rubric (used only if judge_score is selected)",
                               list(BUILTIN_RUBRICS))

    system_prompt = st.text_area(
        "System prompt for the target under test",
        "You answer questions about the Orbital billing platform. Be precise and concise. "
        "If you do not know, say so.",
        height=90,
    )

    if st.button("Run suite", type="primary") and require_api_key("The harness"):
        model = build_chat_model(role="generation")

        async def target(case: EvalCase) -> str:
            from langchain_core.messages import HumanMessage, SystemMessage
            reply = await model.ainvoke([SystemMessage(content=system_prompt),
                                         HumanMessage(content=case.input)])
            return str(reply.content)

        metrics: list[Metric] = []
        if "contains_expected" in chosen:
            metrics.append(contains_metric())
        if "exact_match" in chosen:
            metrics.append(exact_match_metric())
        if "json_valid" in chosen:
            metrics.append(json_valid_metric())
        if "non_empty" in chosen:
            metrics.append(Metric("non_empty", lambda _, out: float(bool(out.strip())),
                                  binary=True, threshold=1.0,
                                  description="Guards against silent empty responses"))
        if "judge_score" in chosen:
            metrics.append(judge_metric(RubricJudge(BUILTIN_RUBRICS[rubric_name])))
        if not metrics:
            st.error("Select at least one metric.")
            st.stop()

        suite = EvaluationSuite(
            f"{dataset_name}-harness", subset, target, metrics,
            assertions=[Assertion(m.name, minimum=0.5, require_ci_clear=False) for m in metrics],
        )
        bar = st.progress(0.0)
        report = suite.run(progress=lambda done, total, _: bar.progress(done / total))

        verdict_banner(report.passed, report.summary())
        dataframe(report.as_rows())
        dataframe([report.run.summary()])
        dataframe([{"case_id": r.case_id, "status": r.status,
                    "latency_ms": round(r.latency_ms), "attempts": r.attempts,
                    **{k: round(v, 3) for k, v in r.scores.items()},
                    "error": r.error or "", "output": r.output[:110]}
                   for r in report.run.results])

        explain("Errored rows are scored as failures, not dropped. Dropping them is how a "
                "suite reports 0.92 while a third of it never ran.")

        run_id = ExperimentStore().save_run(
            report.run, label="harness-ui",
            metrics={name: {"value": interval.estimate, "ci_low": interval.low,
                            "ci_high": interval.high, "n": interval.n}
                     for name, interval in report.aggregates.items()},
        )
        st.success(f"Saved as run `{run_id}` — visible on the Infrastructure and Dashboard pages.")

# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Assertions decide; metrics only measure")
    explain("`require_ci_clear` is the difference between a gate that blocks real regressions "
            "and one that blocks noise: it demands the confidence interval clear the floor, "
            "not just the point estimate.")

    col1, col2, col3 = st.columns(3)
    observed = col1.slider("Observed metric", 0.0, 1.0, 0.82, 0.01)
    floor = col2.slider("Assertion minimum", 0.0, 1.0, 0.80, 0.01)
    n_cases = col3.slider("Suite size", 20, 1000, 100, 10)

    from evalcore.stats import wilson_interval

    interval = wilson_interval(int(round(observed * n_cases)), n_cases)
    point_assertion = Assertion("quality", minimum=floor)
    ci_assertion = Assertion("quality", minimum=floor, require_ci_clear=True)

    ok_point, reason_point = point_assertion.check(interval.estimate, interval=interval)
    ok_ci, reason_ci = ci_assertion.check(interval.estimate, interval=interval)

    dataframe([
        {"gate": "point estimate", "passes": ok_point, "reason": reason_point},
        {"gate": "CI must clear floor", "passes": ok_ci, "reason": reason_ci},
    ])
    st.info(f"Interval: {interval.as_text()}")
    if ok_point and not ok_ci:
        st.warning("This is the dangerous zone: the point estimate passes, but the suite is "
                   "too small to be confident the true value clears the floor.")

    st.markdown("**CI comment produced by the gate**")
    from evalcore.report import MetricGate, run_regression_gate

    baseline = {"quality": {f"case-{i}": 1.0 if i < 84 else 0.0 for i in range(100)}}
    candidate = {"quality": {f"case-{i}": 1.0 if i < int(observed * 100) else 0.0 for i in range(100)}}
    gate_report = run_regression_gate(baseline, candidate, [MetricGate("quality", binary=True,
                                                                        max_regression=0.02)])
    st.code(gate_report.markdown(), language="markdown")
    st.caption(f"CI exit code: {gate_report.exit_code} "
               "(0 ships, 1 blocks, 2 means the comparison could not be made)")

# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Choosing a framework")
    dataframe([
        {"Framework": "OpenAI Evals", "Model-agnostic": "partial", "Judge built in": "yes",
         "CI story": "CLI + registry YAML", "Best for": "Benchmark-style eval registries"},
        {"Framework": "DeepEval", "Model-agnostic": "yes", "Judge built in": "yes",
         "CI story": "pytest plugin", "Best for": "Teams that want eval to look like unit tests"},
        {"Framework": "Promptfoo", "Model-agnostic": "yes", "Judge built in": "yes",
         "CI story": "YAML + CLI, strong diffing", "Best for": "Prompt A/B and red-team sweeps"},
        {"Framework": "LangSmith", "Model-agnostic": "yes", "Judge built in": "yes",
         "CI story": "SDK + hosted UI", "Best for": "Tracing-first LangChain/LangGraph stacks"},
        {"Framework": "Arize Phoenix", "Model-agnostic": "yes", "Judge built in": "yes",
         "CI story": "OTel traces, self-hostable", "Best for": "Open-source observability + eval"},
        {"Framework": "Inspect AI", "Model-agnostic": "yes", "Judge built in": "yes",
         "CI story": "Python task API", "Best for": "Rigorous safety and capability evaluations"},
        {"Framework": "MLflow Evaluate", "Model-agnostic": "yes", "Judge built in": "partial",
         "CI story": "Tracking-server native", "Best for": "Shops already standardised on MLflow"},
        {"Framework": "lm-evaluation-harness", "Model-agnostic": "yes", "Judge built in": "no",
         "CI story": "CLI", "Best for": "Reproducible academic benchmark numbers"},
        {"Framework": "evalcore (this book)", "Model-agnostic": "yes", "Judge built in": "yes",
         "CI story": "exit-code gate + markdown", "Best for": "Understanding what the others do"},
    ])
    explain("Adopt a framework for its runner, tracing and UI. Keep your metric definitions, "
            "golden datasets and gates in your own repository — those are the assets, and they "
            "must outlive whichever tool you are using this year.")
