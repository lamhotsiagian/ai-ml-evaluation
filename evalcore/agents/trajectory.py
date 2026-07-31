"""Trajectory evaluation: scoring *how* an agent worked, not only what it output.

Outcome-only evaluation of agents is a trap. Two agents both scoring 0.8 on
final answers can differ by 4x in cost, by an order of magnitude in blast
radius, and completely in whether they will keep working when a tool changes.
The metrics here make those differences visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Sequence

from evalcore.agents.graph import AgentRun


@dataclass
class TrajectoryReport:
    """Per-run trajectory metrics."""

    outcome_success: float
    tool_selection_accuracy: float
    tool_sequence_similarity: float
    exact_sequence_match: float
    redundant_call_rate: float
    error_recovery_rate: float
    step_efficiency: float
    unsafe_action_count: int
    n_tool_calls: int
    n_expected_calls: int
    stop_reason: str
    notes: list[str] = field(default_factory=list)

    def as_row(self) -> dict[str, Any]:
        return {
            "outcome": round(self.outcome_success, 4),
            "tool_selection": round(self.tool_selection_accuracy, 4),
            "sequence_similarity": round(self.tool_sequence_similarity, 4),
            "exact_sequence": round(self.exact_sequence_match, 4),
            "redundant_rate": round(self.redundant_call_rate, 4),
            "error_recovery": round(self.error_recovery_rate, 4),
            "step_efficiency": round(self.step_efficiency, 4),
            "unsafe_actions": self.unsafe_action_count,
            "tool_calls": self.n_tool_calls,
            "stop_reason": self.stop_reason,
        }

    def composite(self, weights: dict[str, float] | None = None) -> float:
        """Weighted composite score.

        The default weighting puts outcome at half. Trajectory quality matters,
        but a system that reliably reaches the wrong answer through a beautiful
        plan is still broken; the trajectory terms are there to break ties and
        catch cost and safety regressions that outcome alone hides.
        """
        weights = weights or {
            "outcome": 0.50, "tool_selection": 0.20,
            "sequence": 0.10, "efficiency": 0.10, "recovery": 0.10,
        }
        score = (
            weights["outcome"] * self.outcome_success
            + weights["tool_selection"] * self.tool_selection_accuracy
            + weights["sequence"] * self.tool_sequence_similarity
            + weights["efficiency"] * self.step_efficiency
            + weights["recovery"] * self.error_recovery_rate
        )
        # An unsafe action is not a deduction to be averaged away.
        return 0.0 if self.unsafe_action_count > 0 else score


def evaluate_trajectory(
    run: AgentRun,
    *,
    expected_tools: Sequence[str] = (),
    forbidden_tools: Sequence[str] = (),
    outcome_success: float | None = None,
    optimal_steps: int | None = None,
) -> TrajectoryReport:
    """Score one agent run against an expected tool plan.

    Args:
        run: The recorded agent execution.
        expected_tools: The reference tool sequence. Order matters for
            ``exact_sequence_match`` and ``tool_sequence_similarity``; set
            membership drives ``tool_selection_accuracy``.
        forbidden_tools: Tools that must never be called for this task. Any call
            is an unsafe action and zeroes the composite.
        outcome_success: Externally judged success of the final answer. When
            omitted, the agent's own completion flag is used, which is a weak
            proxy and should be replaced by a judge or an exact-match check.
        optimal_steps: Reference step count for the efficiency term. Defaults to
            ``len(expected_tools)``.
    """
    actual = run.tool_sequence
    expected = list(expected_tools)

    # -- selection: did it use the right tools, ignoring order -------------
    if expected:
        expected_set, actual_set = set(expected), set(actual)
        true_positive = len(expected_set & actual_set)
        precision = true_positive / len(actual_set) if actual_set else 0.0
        recall = true_positive / len(expected_set)
        selection = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    else:
        # No expected plan: penalise only the use of a tool that does not exist.
        selection = 1.0 if not any(s.error == "unknown_tool" for s in run.steps) else 0.0

    similarity = SequenceMatcher(None, expected, actual).ratio() if expected else 1.0
    exact = 1.0 if expected and actual == expected else (1.0 if not expected else 0.0)

    # -- redundancy: identical (tool, args) invoked more than once ---------
    signatures = [
        (step.tool, tuple(sorted((k, str(v)) for k, v in step.arguments.items())))
        for step in run.steps if step.kind == "tool_call"
    ]
    unique = len(set(signatures))
    redundant_rate = 1 - unique / len(signatures) if signatures else 0.0

    # -- recovery: after an errored observation, did the next call differ? --
    recoveries, opportunities = 0, 0
    observations = [s for s in run.steps if s.kind == "observation"]
    for position, step in enumerate(observations):
        if not step.error:
            continue
        opportunities += 1
        following = observations[position + 1 :]
        if following and not following[0].error:
            recoveries += 1
    recovery = recoveries / opportunities if opportunities else 1.0

    # -- efficiency: steps taken versus a reference optimum ----------------
    reference = optimal_steps if optimal_steps is not None else (len(expected) or 1)
    efficiency = min(1.0, reference / len(actual)) if actual else (1.0 if reference == 0 else 0.0)

    unsafe = sum(1 for name in actual if name in set(forbidden_tools))

    notes: list[str] = []
    if run.stop_reason == "iteration_limit":
        notes.append("hit iteration limit -- likely a planning or stopping-condition failure")
    if redundant_rate > 0.3:
        notes.append(f"{redundant_rate:.0%} of tool calls were exact repeats")
    if unsafe:
        notes.append(f"called {unsafe} forbidden tool(s)")
    if opportunities and recovery < 0.5:
        notes.append("failed to adapt after tool errors")

    return TrajectoryReport(
        outcome_success=float(outcome_success if outcome_success is not None else run.succeeded),
        tool_selection_accuracy=selection,
        tool_sequence_similarity=similarity,
        exact_sequence_match=exact,
        redundant_call_rate=redundant_rate,
        error_recovery_rate=recovery,
        step_efficiency=efficiency,
        unsafe_action_count=unsafe,
        n_tool_calls=len(actual),
        n_expected_calls=len(expected),
        stop_reason=run.stop_reason,
        notes=notes,
    )


@dataclass
class AgentSuiteReport:
    """Aggregate over a suite of agent runs."""

    n_runs: int
    outcome_rate: float
    mean_composite: float
    mean_tool_selection: float
    mean_steps: float
    limit_hit_rate: float
    unsafe_run_rate: float
    cost_per_success: float

    def as_row(self) -> dict[str, float]:
        return {
            "n_runs": self.n_runs,
            "outcome_rate": round(self.outcome_rate, 4),
            "composite": round(self.mean_composite, 4),
            "tool_selection": round(self.mean_tool_selection, 4),
            "mean_steps": round(self.mean_steps, 2),
            "limit_hit_rate": round(self.limit_hit_rate, 4),
            "unsafe_run_rate": round(self.unsafe_run_rate, 4),
            "llm_calls_per_success": round(self.cost_per_success, 2),
        }


def aggregate_trajectories(
    runs: Sequence[AgentRun], reports: Sequence[TrajectoryReport]
) -> AgentSuiteReport:
    """Suite-level agent metrics, including cost per *success*.

    Cost per success rather than cost per run: an agent that is cheap because it
    gives up early is not cheap, it is useless, and dividing by successes is the
    arithmetic that says so.
    """
    if not runs:
        return AgentSuiteReport(0, 0, 0, 0, 0, 0, 0, float("nan"))
    n = len(runs)
    successes = sum(r.outcome_success for r in reports)
    llm_calls = sum(run.n_llm_calls for run in runs)
    return AgentSuiteReport(
        n_runs=n,
        outcome_rate=successes / n,
        mean_composite=sum(r.composite() for r in reports) / n,
        mean_tool_selection=sum(r.tool_selection_accuracy for r in reports) / n,
        mean_steps=sum(r.n_tool_calls for r in reports) / n,
        limit_hit_rate=sum(1 for run in runs if run.stop_reason == "iteration_limit") / n,
        unsafe_run_rate=sum(1 for r in reports if r.unsafe_action_count > 0) / n,
        cost_per_success=llm_calls / successes if successes else float("inf"),
    )


def diff_trajectories(baseline: AgentRun, candidate: AgentRun) -> dict[str, Any]:
    """Side-by-side comparison used for debugging an agent regression.

    The first question after an agent regression is always "where did the two
    runs diverge?". This returns the divergence index and both tails, which is
    usually enough to identify the offending prompt or tool-description change.
    """
    a, b = baseline.tool_sequence, candidate.tool_sequence
    divergence = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return {
        "diverged_at_step": divergence,
        "baseline_tail": a[divergence:],
        "candidate_tail": b[divergence:],
        "baseline_steps": len(a),
        "candidate_steps": len(b),
        "baseline_stop": baseline.stop_reason,
        "candidate_stop": candidate.stop_reason,
        "step_delta": len(b) - len(a),
    }
