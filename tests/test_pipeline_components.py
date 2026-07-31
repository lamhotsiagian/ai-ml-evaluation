"""Tests for the parts of the pipeline that need no live model."""

from __future__ import annotations

import numpy as np
import pytest

from evalcore.agents.graph import AgentRun, TrajectoryStep
from evalcore.agents.trajectory import aggregate_trajectories, diff_trajectories, evaluate_trajectory
from evalcore.datasets import EvalCase, GoldenDataset
from evalcore.judge.rubric import Criterion, JudgeVerdict, Rubric
from evalcore.judge.validators import (
    ValidatorSuite,
    exact_match,
    extract_json,
    validate_json,
    validate_python_syntax,
)
from evalcore.production.cost import Configuration, estimate_cost, pareto_frontier
from evalcore.production.drift import embedding_drift, ks_drift, population_stability_index
from evalcore.production.online import CanaryController, analyse_ab_test, assign_variant
from evalcore.rag.index import BM25Index, sentence_chunk
from evalcore.rag.metrics import context_precision, evaluate_abstention, evaluate_citations
from evalcore.report import MetricGate, run_regression_gate
from evalcore.splits import detect_leakage, grouped_split, stratified_split


# ---------------------------------------------------------------------------
# Datasets and splits
# ---------------------------------------------------------------------------
def _dataset(n: int = 20) -> GoldenDataset:
    return GoldenDataset("t", [
        EvalCase(case_id=f"c{i}", input=f"q{i}", expected_output=f"a{i}",
                 slice_tags=["even" if i % 2 == 0 else "odd"],
                 difficulty="hard" if i % 5 == 0 else "easy")
        for i in range(n)
    ])


def test_dataset_hash_is_order_independent_and_content_sensitive():
    dataset = _dataset()
    reversed_dataset = GoldenDataset("t", list(reversed(dataset.cases)))
    assert dataset.content_hash() == reversed_dataset.content_hash()

    edited = GoldenDataset("t", [dataset.cases[0].model_copy(update={"input": "changed"})]
                           + dataset.cases[1:])
    assert edited.content_hash() != dataset.content_hash()


def test_duplicate_case_ids_are_rejected():
    case = EvalCase(case_id="dup", input="x")
    with pytest.raises(ValueError, match="duplicate case_id"):
        GoldenDataset("t", [case, case.model_copy()])


def test_grouped_split_never_leaks_a_group():
    groups = [f"conv-{i // 12}" for i in range(600)]
    split = grouped_split(groups)
    train_groups = {groups[i] for i in split.train}
    test_groups = {groups[i] for i in split.test}
    assert not (train_groups & test_groups)


def test_stratified_split_preserves_a_rare_class():
    labels = ["rare"] * 30 + ["common"] * 970
    split = stratified_split(labels)
    test_labels = [labels[i] for i in split.test]
    assert test_labels.count("rare") >= 3


def test_leakage_detector_finds_near_duplicates():
    train = ["The Growth plan costs 249 USD per month and includes 2,000,000 events"]
    test = ["the growth plan costs 249 usd per month and includes 2,000,000 events today"]
    report = detect_leakage(train, test, near_duplicate_threshold=0.6, shingle_k=4)
    assert not report.clean
    assert report.near_duplicates or report.exact_duplicates


def test_leakage_detector_passes_a_clean_split():
    report = detect_leakage(["what is the overage rate"], ["how long are events retained"])
    assert report.clean


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("payload", [
    '{"a": 1}',
    '```json\n{"a": 1}\n```',
    'Sure! Here you go:\n{"a": 1}\nLet me know if you need more.',
])
def test_json_extraction_survives_model_formatting_habits(payload):
    assert extract_json(payload) == {"a": 1}


def test_json_extraction_raises_on_genuinely_malformed_output():
    with pytest.raises(ValueError):
        extract_json("there is no json here at all")


def test_json_schema_validation_reports_the_offending_field():
    schema = {"type": "object", "required": ["plan"], "properties": {"plan": {"type": "string"}}}
    assert validate_json('{"plan": "Growth"}', schema).passed
    assert not validate_json('{"tier": "Growth"}', schema).passed


def test_python_syntax_validator_does_not_execute_code():
    assert validate_python_syntax("def f():\n    return 1").passed
    assert not validate_python_syntax("def f(:\n  return").passed


def test_exact_match_normalisation():
    assert exact_match("Paris.", "the paris").passed
    assert not exact_match("Paris", "London").passed


def test_validator_suite_pass_rate():
    suite = ValidatorSuite().add("json", lambda t: validate_json(t)).add("py", validate_python_syntax)
    # A dict literal is both valid JSON and valid Python, so both checks pass.
    assert suite.pass_rate('{"a": 1}') == pytest.approx(1.0)
    # Prose is neither.
    assert suite.pass_rate("def f(: pass") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Rubrics
# ---------------------------------------------------------------------------
def test_rubric_requires_anchored_scale_points():
    with pytest.raises(ValueError, match="anchors must define"):
        Criterion(name="X", question="?", anchors={2: "a", 4: "b"})


def test_weighted_score_respects_criterion_weights():
    rubric = Rubric(name="r", criteria=[
        Criterion(name="Heavy", question="?", anchors={1: "a", 3: "b", 5: "c"}, weight=3.0),
        Criterion(name="Light", question="?", anchors={1: "a", 3: "b", 5: "c"}, weight=1.0),
    ])
    verdict = JudgeVerdict(
        criteria=[
            {"criterion": "Heavy", "score": 5, "evidence": "e", "reasoning": "r"},  # type: ignore[list-item]
            {"criterion": "Light", "score": 1, "evidence": "e", "reasoning": "r"},  # type: ignore[list-item]
        ],
        overall_score=4.0, verdict="pass", confidence=0.9,
    )
    assert rubric.weighted_score(verdict.criteria) == pytest.approx((5 * 3 + 1 * 1) / 4)


def test_verdict_normalisation_maps_one_to_five_onto_zero_to_one():
    verdict = JudgeVerdict(criteria=[], overall_score=3.0, verdict="pass", confidence=0.5)
    assert verdict.normalised() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------
def test_sentence_chunking_respects_the_token_budget():
    text = " ".join(f"Sentence number {i} carries some content." for i in range(120))
    chunks = sentence_chunk(text, "doc", target_tokens=60, overlap_tokens=10)
    assert len(chunks) > 1
    assert all(chunk.n_tokens <= 90 for chunk in chunks)
    assert all(chunk.chunk_id.startswith("doc::") for chunk in chunks)


def test_bm25_beats_naive_matching_on_an_exact_identifier():
    from evalcore.rag.index import Chunk

    chunks = [
        Chunk("c1", "Error ORB-1002 means an unknown metric key was submitted.", "d", 0),
        Chunk("c2", "The billing period closes at midnight UTC on the first of the month.", "d", 1),
        Chunk("c3", "Rate limits are enforced per workspace and per plan tier.", "d", 2),
    ]
    hits = BM25Index(chunks).search("ORB-1002", k=1)
    assert hits and hits[0].chunk.chunk_id == "c1"


def test_context_precision_rewards_early_relevant_passages():
    early = context_precision(["gold", "x", "y"], {"gold"})
    late = context_precision(["x", "y", "gold"], {"gold"})
    assert early > late


def test_citation_checker_detects_fabricated_indices():
    report = evaluate_citations(
        "The Growth plan costs 249 USD per month [1]. Retention is 400 days [9].", n_contexts=3)
    assert report.invalid_indices == [9]
    assert report.citation_rate == pytest.approx(1.0)
    assert report.valid_citation_rate == pytest.approx(0.5)


def test_abstention_metrics_separate_the_two_failure_modes():
    result = evaluate_abstention(
        abstained=[True, True, False, False, True],
        is_answerable=[False, False, True, True, True],
    )
    assert result["abstention_recall"] == pytest.approx(1.0)
    assert result["over_abstention_rate"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
def _run(tools: list[str], *, stop: str = "answered", errors: int = 0) -> AgentRun:
    steps: list[TrajectoryStep] = []
    for index, name in enumerate(tools):
        steps.append(TrajectoryStep(len(steps), "tool_call", tool=name, arguments={"i": index}))
        steps.append(TrajectoryStep(len(steps), "observation", tool=name, arguments={"i": index},
                                    error="boom" if index < errors else None))
    return AgentRun("task", "answer", steps, True, 100.0, len(tools) + 1, stop)


def test_trajectory_scores_the_path_not_only_the_answer():
    clean = evaluate_trajectory(_run(["calculator"]), expected_tools=["calculator"],
                                outcome_success=1.0)
    wandering = evaluate_trajectory(
        _run(["knowledge_search", "knowledge_search", "calculator"]),
        expected_tools=["calculator"], outcome_success=1.0)
    assert clean.composite() > wandering.composite()
    assert clean.step_efficiency > wandering.step_efficiency


def test_forbidden_tool_zeroes_the_composite():
    report = evaluate_trajectory(_run(["calculator", "delete_database"]),
                                 expected_tools=["calculator"],
                                 forbidden_tools=["delete_database"], outcome_success=1.0)
    assert report.unsafe_action_count == 1
    assert report.composite() == 0.0


def test_redundant_calls_are_detected():
    run = AgentRun("t", "a", [
        TrajectoryStep(0, "tool_call", tool="calculator", arguments={"expression": "1+1"}),
        TrajectoryStep(1, "tool_call", tool="calculator", arguments={"expression": "1+1"}),
    ], True, 10.0, 2, "answered")
    assert evaluate_trajectory(run).redundant_call_rate == pytest.approx(0.5)


def test_cost_per_success_is_infinite_when_nothing_succeeds():
    runs = [_run(["calculator"]) for _ in range(3)]
    reports = [evaluate_trajectory(r, outcome_success=0.0) for r in runs]
    assert aggregate_trajectories(runs, reports).cost_per_success == float("inf")


def test_diff_identifies_the_divergence_point():
    diff = diff_trajectories(_run(["a", "b", "c"]), _run(["a", "x", "y"]))
    assert diff["diverged_at_step"] == 1
    assert diff["candidate_tail"] == ["x", "y"]


# ---------------------------------------------------------------------------
# Production
# ---------------------------------------------------------------------------
def test_psi_is_near_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    sample = rng.normal(0, 1, 4000)
    result = population_stability_index(sample, rng.normal(0, 1, 4000))
    assert result.statistic < 0.1
    assert result.severity == "none"


def test_psi_and_ks_both_fire_on_a_large_shift():
    rng = np.random.default_rng(1)
    reference = rng.normal(0, 1, 3000)
    current = rng.normal(1.5, 1, 3000)
    assert population_stability_index(reference, current).drifted
    assert ks_drift(reference, current).drifted


def test_embedding_drift_detects_a_semantic_shift():
    rng = np.random.default_rng(2)
    reference = rng.normal(0, 1, (400, 16))
    assert not embedding_drift(reference, rng.normal(0, 1, (400, 16))).drifted
    assert embedding_drift(reference, rng.normal(2.0, 1, (400, 16))).drifted


def test_variant_assignment_is_sticky_and_independent_across_experiments():
    assert assign_variant("u1", "exp-a") == assign_variant("u1", "exp-a")
    splits = [assign_variant(f"u{i}", "exp-a") for i in range(4000)]
    assert 0.45 < splits.count("treatment") / len(splits) < 0.55


def test_ab_test_blocks_on_a_guardrail_regardless_of_lift():
    result = analyse_ab_test([0] * 500 + [1] * 500, [0] * 300 + [1] * 700,
                             guardrail_violated=True)
    assert result.decision == "rollback"


def test_ab_test_is_inconclusive_below_the_practical_threshold():
    rng = np.random.default_rng(5)
    control = (rng.random(20000) < 0.700).astype(float)
    treatment = (rng.random(20000) < 0.708).astype(float)
    assert analyse_ab_test(control, treatment, practical_threshold=0.05).decision != "ship"


def test_canary_rolls_back_on_a_breached_budget():
    decision = CanaryController().evaluate(
        n_observations=1000, error_rate=0.09, latency_p95_ms=1200,
        quality_score=0.85, baseline_quality=0.86)
    assert decision.action == "rollback"


def test_canary_holds_until_it_has_enough_observations():
    decision = CanaryController(min_observations=500).evaluate(
        n_observations=100, error_rate=0.0, latency_p95_ms=800,
        quality_score=0.9, baseline_quality=0.9)
    assert decision.action == "hold"


def test_pareto_frontier_drops_dominated_configurations():
    configurations = [
        Configuration("good", 0.90, 1.0, 1000),
        Configuration("dominated", 0.85, 2.0, 2000),
        Configuration("cheap", 0.80, 0.2, 500),
    ]
    names = {c.name for c in pareto_frontier(configurations)}
    assert "dominated" not in names
    assert {"good", "cheap"} <= names


def test_unknown_model_cost_falls_back_upward():
    known = estimate_cost("gemini-2.0-flash-lite", 1_000_000, 1_000_000).total_usd
    unknown = estimate_cost("some-future-model", 1_000_000, 1_000_000).total_usd
    assert unknown > known


# ---------------------------------------------------------------------------
# Regression gate
# ---------------------------------------------------------------------------
def test_gate_refuses_to_compare_incomparable_runs():
    report = run_regression_gate({}, {}, [MetricGate("q")], comparable=False,
                                 incomparable_reason="dataset changed")
    assert report.decision == "incomparable"
    assert report.exit_code == 2


def test_gate_blocks_a_significant_regression():
    baseline = {"q": {f"c{i}": 1.0 for i in range(100)}}
    candidate = {"q": {f"c{i}": (0.0 if i < 25 else 1.0) for i in range(100)}}
    report = run_regression_gate(baseline, candidate, [MetricGate("q", binary=True)])
    assert report.decision == "fail"
    assert report.exit_code == 1
    assert len(report.findings[0].regressed_cases) == 25


def test_gate_passes_within_the_regression_budget():
    baseline = {"q": {f"c{i}": 1.0 for i in range(200)}}
    candidate = {"q": {f"c{i}": (0.0 if i < 2 else 1.0) for i in range(200)}}
    report = run_regression_gate(baseline, candidate,
                                 [MetricGate("q", binary=True, max_regression=0.05)])
    assert report.decision == "pass"


def test_gate_abstains_on_a_tiny_suite():
    baseline = {"q": {f"c{i}": 1.0 for i in range(8)}}
    candidate = {"q": {f"c{i}": 0.0 for i in range(8)}}
    report = run_regression_gate(baseline, candidate, [MetricGate("q", binary=True)],
                                 min_cases=30)
    assert report.findings[0].decision == "insufficient_data"


def test_gate_markdown_lists_newly_failing_cases():
    baseline = {"q": {f"c{i}": 1.0 for i in range(60)}}
    candidate = {"q": {f"c{i}": (0.0 if i < 20 else 1.0) for i in range(60)}}
    markdown = run_regression_gate(baseline, candidate, [MetricGate("q", binary=True)]).markdown()
    assert "Newly failing cases" in markdown and "`c0`" in markdown
