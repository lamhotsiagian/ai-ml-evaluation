"""Chapter 3 lab — rubric judging, pairwise comparison, bias probes, calibration."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalcore.judge import (  # noqa: E402
    BUILTIN_RUBRICS,
    PairwiseJudge,
    RubricJudge,
    ValidatorSuite,
    bradley_terry_scores,
    calibrate_judge,
)
from evalcore.judge.validators import (  # noqa: E402
    exact_match,
    validate_json,
    validate_length,
    validate_python_syntax,
)
from ui.components import (  # noqa: E402
    dataframe,
    explain,
    page_header,
    require_api_key,
    settings_sidebar,
    verdict_banner,
)

page_header("LLM-as-a-Judge", 3, "rubrics, structured verdicts, bias, calibration")
settings_sidebar()

tabs = st.tabs(["Deterministic checks first", "Rubric judge", "Pairwise & arena",
                "Judge calibration"])

# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Everything a rule can check should never reach the judge")
    explain("These checks are free, instant and never hallucinate. Roughly half of real "
            "evaluation failures are caught here at zero marginal cost.")
    sample = st.text_area(
        "Model output to validate",
        '```json\n{"plan": "Growth", "monthly_usd": 249, "included_events": 2000000}\n```',
        height=140,
    )
    schema = {
        "type": "object",
        "required": ["plan", "monthly_usd", "included_events"],
        "properties": {
            "plan": {"type": "string"},
            "monthly_usd": {"type": "number"},
            "included_events": {"type": "integer"},
        },
    }
    suite = (
        ValidatorSuite()
        .add("json", lambda text: validate_json(text, schema))
        .add("length", lambda text: validate_length(text, max_words=120))
        .add("python_syntax", validate_python_syntax)
    )
    results = suite.run(sample)
    dataframe([{"check": name, "passed": r.passed, "detail": r.detail}
               for name, r in results.items()])
    st.metric("Deterministic pass rate", f"{suite.pass_rate(sample):.0%}")

    st.markdown("**Normalised exact match**")
    col1, col2 = st.columns(2)
    prediction = col1.text_input("Prediction", "The Growth plan.")
    reference = col2.text_input("Reference", "growth plan")
    match = exact_match(prediction, reference)
    verdict_banner(match.passed, f"exact_match = {match.score:.0f} — {match.detail}")

# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Rubric-driven single-response judging")
    rubric_name = st.selectbox("Rubric", list(BUILTIN_RUBRICS))
    rubric = BUILTIN_RUBRICS[rubric_name]
    with st.expander("Rendered rubric (this exact text goes into the prompt)"):
        st.code(rubric.render(), language="markdown")

    col1, col2 = st.columns(2)
    task = col1.text_area("Task given to the system",
                          "What is the overage rate on the Starter plan?", height=110)
    response = col2.text_area(
        "Response under evaluation",
        "Starter overage is billed at 0.0009 USD per event, and unused events roll over "
        "to the next month.",
        height=110,
    )
    contexts = st.text_area(
        "Context supplied to the system (one passage per line)",
        "Overage on Starter is billed at 0.0009 USD per event. Overage on Growth is billed "
        "at 0.0004 USD per event.",
        height=90,
    )
    n_samples = st.slider("Self-consistency samples", 1, 5, 1,
                          help="More than 1 switches the judge to temperature 0.7 and takes "
                               "the median; the agreement rate becomes a monitorable number.")

    if st.button("Run judge", type="primary") and require_api_key("The rubric judge"):
        with st.spinner("Judging…"):
            judge = RubricJudge(rubric, n_samples=n_samples)
            result = judge.judge(
                task, response,
                contexts=[c for c in contexts.splitlines() if c.strip()] or None,
            )
        cols = st.columns(4)
        cols[0].metric("Overall score", f"{result.verdict.overall_score:.1f} / 5")
        cols[1].metric("Weighted", f"{result.weighted_score:.2f}")
        cols[2].metric("Verdict", result.verdict.verdict.upper())
        cols[3].metric("Sample agreement", f"{result.sample_agreement:.0%}")
        dataframe([{"criterion": c.criterion, "score": c.score,
                    "evidence": c.evidence[:90], "reasoning": c.reasoning[:140]}
                   for c in result.verdict.criteria])
        if result.verdict.failure_modes:
            st.warning("Failure modes: " + ", ".join(result.verdict.failure_modes))
        st.caption(f"model={result.model}, cached={result.cached}, samples={result.sample_scores}")

# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Pairwise comparison with mandatory position swapping")
    explain("Every comparison runs in both presentation orders. When the orders disagree the "
            "result is recorded as a tie and flagged as position bias — that disagreement "
            "rate is itself the diagnostic.")
    task_p = st.text_input("Task", "Explain the Orbital circuit breaker in two sentences.")
    col1, col2 = st.columns(2)
    response_a = col1.text_area(
        "Response A",
        "When a workspace passes 300% of its included events in a billing period, ingestion "
        "pauses. Only an owner can lift the pause.", height=130)
    response_b = col2.text_area(
        "Response B",
        "Orbital has a safety mechanism. It stops things when usage gets very high, and then "
        "somebody with the right permissions has to turn it back on again.", height=130)

    if st.button("Compare", type="primary") and require_api_key("The pairwise judge"):
        with st.spinner("Comparing in both orders…"):
            outcome = PairwiseJudge(BUILTIN_RUBRICS["Instruction Following"]).compare(
                task_p, response_a, response_b)
        cols = st.columns(3)
        cols[0].metric("Winner", outcome["winner"])
        cols[1].metric("Margin", outcome["margin"])
        cols[2].metric("Position bias", "YES" if outcome["position_bias"] else "no")
        st.caption(f"forward={outcome['forward_winner']}, reverse={outcome['reverse_winner']}, "
                   f"deciding criterion={outcome['deciding_criterion']}")
        st.write(outcome["reasoning"])

    st.markdown("**Arena ranking from pairwise outcomes**")
    explain("Bradley-Terry converts a sparse set of head-to-head results into a ranking, "
            "which is how MT-Bench and Chatbot Arena leaderboards are built.")
    default = "\n".join([
        "gemini-flash,gemini-flash-lite,gemini-flash",
        "gemini-flash,baseline-prompt,gemini-flash",
        "gemini-flash-lite,baseline-prompt,gemini-flash-lite",
        "gemini-flash,gemini-flash-lite,tie",
        "baseline-prompt,gemini-flash-lite,baseline-prompt",
    ])
    raw = st.text_area("Comparisons: system_a,system_b,winner", default, height=140)
    triples = [tuple(part.strip() for part in line.split(",")) for line in raw.splitlines()
               if line.count(",") == 2]
    if triples:
        strengths = bradley_terry_scores(triples)  # type: ignore[arg-type]
        dataframe([{"system": name, "bradley_terry_strength": round(value, 4),
                    "elo_like": round(400 * np.log10(value / min(strengths.values())), 1)}
                   for name, value in sorted(strengths.items(), key=lambda kv: -kv[1])])

# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Meta-evaluation: does the judge agree with humans?")
    explain("An uncalibrated judge is a random number generator with good manners. Paste "
            "human scores and judge scores for the same items to get a ship/no-ship verdict.")
    col1, col2 = st.columns(2)
    human_raw = col1.text_area(
        "Human scores (1-5, one per line)",
        "\n".join(["5", "4", "2", "1", "3", "5", "4", "2", "3", "1",
                   "5", "3", "4", "2", "1", "4", "5", "3", "2", "4"]), height=220)
    judge_raw = col2.text_area(
        "Judge scores (1-5, one per line)",
        "\n".join(["5", "4", "3", "2", "3", "5", "5", "3", "3", "2",
                   "4", "3", "4", "3", "1", "4", "5", "4", "2", "4"]), height=220)

    try:
        human = [float(v) for v in human_raw.split()]
        judge_scores = [float(v) for v in judge_raw.split()]
    except ValueError:
        st.error("Scores must be numbers, one per line.")
        human, judge_scores = [], []

    if human and len(human) == len(judge_scores):
        report = calibrate_judge(human, judge_scores, pass_threshold=3.0)
        verdict_banner(report.deployable, report.verdict_text())
        cols = st.columns(5)
        cols[0].metric("Cohen's κ", f"{report.kappa:.3f}")
        cols[1].metric("Raw agreement", f"{report.raw_agreement.estimate:.3f}")
        cols[2].metric("Spearman ρ", f"{report.spearman:.3f}")
        cols[3].metric("False-pass rate", f"{report.false_pass_rate:.3f}")
        cols[4].metric("Mean bias", f"{report.mean_bias:+.3f}")
        explain("κ ≥ 0.6 and false-pass ≤ 0.10 is the ship gate. A judge that waves through "
                "bad output is worse than no judge: it launders a failure into a green dashboard.")
    elif human:
        st.error(f"Human ({len(human)}) and judge ({len(judge_scores)}) score counts differ.")
