"""Chapter 5 lab — outcome vs trajectory evaluation of a LangGraph agent."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalcore.agents import (  # noqa: E402
    EvaluableAgent,
    aggregate_trajectories,
    build_evaluation_tools,
    diff_trajectories,
    evaluate_trajectory,
)
from evalcore.datasets import DatasetRegistry  # noqa: E402
from evalcore.rag import RagIndex  # noqa: E402
from ui.components import (  # noqa: E402
    dataframe,
    explain,
    page_header,
    require_api_key,
    settings_sidebar,
)

page_header("Agent Evaluation", 5, "trajectories, tool selection, efficiency, safety")
settings_sidebar()

registry = DatasetRegistry()
dataset = registry.load("agent_tasks")

with st.sidebar:
    st.subheader("Agent configuration")
    max_iterations = st.slider("Max iterations", 2, 12, 6)
    enable_search = st.checkbox("Give the agent corpus search", value=True)
    forbidden = st.multiselect("Forbidden tools for this task",
                               ["calculator", "unit_convert", "date_difference", "knowledge_search"])


@st.cache_resource(show_spinner="Building the corpus index for knowledge_search…")
def build_search_index() -> RagIndex:
    index = RagIndex(collection_name="agent_corpus")
    return index.build(index.load_corpus())


def make_agent() -> EvaluableAgent:
    searcher = None
    if enable_search:
        index = build_search_index()
        searcher = lambda query, k: [h.chunk.text for h in index.search(query, k, mode="hybrid")]  # noqa: E731
    return EvaluableAgent(build_evaluation_tools(searcher), max_iterations=max_iterations)


tabs = st.tabs(["Single run", "Suite", "Regression diff"])

# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("One run, fully traced")
    explain("Outcome-only evaluation cannot tell a clean solve from one that wandered through "
            "three wasted tool calls. Every decision below is a recorded, scorable step.")
    case = st.selectbox("Task", dataset.cases,
                        format_func=lambda c: f"{c.case_id} · {c.input[:70]}")
    st.caption(f"Expected tools: `{', '.join(case.expected_tools) or 'none'}` · "
               f"expected answer: `{case.expected_output}`")

    if st.button("Run agent", type="primary") and require_api_key("The agent"):
        with st.spinner("Running…"):
            run = make_agent().run(case.input)

        st.markdown("**Final answer**")
        st.write(run.final_answer)

        cols = st.columns(5)
        cols[0].metric("Stop reason", run.stop_reason)
        cols[1].metric("Tool calls", run.n_tool_calls)
        cols[2].metric("LLM calls", run.n_llm_calls)
        cols[3].metric("Errors", run.n_errors)
        cols[4].metric("Latency", f"{run.total_latency_ms:.0f} ms")

        st.markdown("**Trajectory**")
        dataframe(run.to_records())

        outcome = float(str(case.expected_output).lower().strip() in run.final_answer.lower())
        report = evaluate_trajectory(run, expected_tools=case.expected_tools,
                                     forbidden_tools=forbidden, outcome_success=outcome)
        dataframe([report.as_row()])
        st.metric("Composite score", f"{report.composite():.3f}",
                  help="Outcome 0.50, tool selection 0.20, sequence 0.10, efficiency 0.10, "
                       "recovery 0.10. Any unsafe action zeroes it outright.")
        for note in report.notes:
            st.warning(note)

# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Suite — outcome rate is only one of six numbers")
    limit = st.slider("Tasks to run", 2, len(dataset), 6)
    if st.button("Run suite", type="primary", key="agent_suite") and require_api_key("The agent suite"):
        agent = make_agent()
        runs, reports, rows = [], [], []
        progress = st.progress(0.0)
        cases = dataset.cases[:limit]
        for position, item in enumerate(cases, start=1):
            run = agent.run(item.input)
            outcome = float(str(item.expected_output).lower().strip() in run.final_answer.lower())
            report = evaluate_trajectory(run, expected_tools=item.expected_tools,
                                         forbidden_tools=forbidden, outcome_success=outcome)
            runs.append(run)
            reports.append(report)
            rows.append({"case_id": item.case_id, **report.as_row(),
                         "composite": round(report.composite(), 3),
                         "answer": run.final_answer[:90]})
            progress.progress(position / len(cases))

        dataframe(rows)
        summary = aggregate_trajectories(runs, reports)
        dataframe([summary.as_row()])
        explain("Cost per success, not per run. An agent that is cheap because it gives up "
                "early is not cheap — dividing by successes is the arithmetic that says so.")

        st.plotly_chart(
            go.Figure([
                go.Bar(name="outcome", x=[r["case_id"] for r in rows],
                       y=[r["outcome"] for r in rows]),
                go.Bar(name="composite", x=[r["case_id"] for r in rows],
                       y=[r["composite"] for r in rows]),
            ]).update_layout(barmode="group", height=340, yaxis_title="score",
                             margin=dict(l=10, r=10, t=20, b=10)),
            width="stretch")

# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Where did two runs diverge?")
    explain("The first question after an agent regression. The divergence index plus both "
            "tails usually identifies the offending prompt or tool-description change.")
    task_text = st.text_input("Task to run twice",
                              "A Growth customer used 2,400,000 events. What is the total bill?")
    if st.button("Run A/B", type="primary", key="agent_diff") and require_api_key("The agent"):
        with st.spinner("Running baseline…"):
            baseline_agent = EvaluableAgent(build_evaluation_tools(None), max_iterations=max_iterations)
            baseline_run = baseline_agent.run(task_text)
        with st.spinner("Running candidate…"):
            candidate_run = make_agent().run(task_text)

        dataframe([diff_trajectories(baseline_run, candidate_run)])
        left, right = st.columns(2)
        with left:
            st.caption("Baseline (no corpus search)")
            dataframe(baseline_run.to_records())
        with right:
            st.caption("Candidate (current configuration)")
            dataframe(candidate_run.to_records())
