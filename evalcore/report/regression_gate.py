"""The regression gate: turning evaluation output into a build decision.

This is the component that makes evaluation *operational*. Everything upstream
produces numbers; this decides whether a change ships. Its three rules come from
watching gates fail in practice:

1. **Refuse to compare incomparable runs.** Different dataset hash or different
   settings fingerprint means the ruler moved, and the comparison is invalid.
2. **Test the difference, do not eyeball it.** A 1.5-point drop on 80 cases is
   noise; a paired test says so and prevents both false alarms and false calm.
3. **Report what broke, not only that something broke.** A gate that says
   "quality regressed" wastes an hour; a gate that names the six cases that
   flipped from pass to fail is a debugging session that has already started.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

from evalcore.stats import Interval, TestResult, mcnemar_test, paired_bootstrap_test

Decision = Literal["pass", "fail", "incomparable", "insufficient_data"]


@dataclass
class MetricGate:
    """Gate configuration for one metric."""

    metric: str
    floor: float | None = None
    max_regression: float = 0.02
    alpha: float = 0.05
    binary: bool = False
    blocking: bool = True

    def describe(self) -> str:
        parts = [f"max regression {self.max_regression:+.3f}"]
        if self.floor is not None:
            parts.append(f"floor {self.floor:.3f}")
        parts.append("blocking" if self.blocking else "advisory")
        return f"{self.metric}: " + ", ".join(parts)


@dataclass
class GateFinding:
    metric: str
    baseline: float
    candidate: float
    delta: float
    test: TestResult | None
    decision: Decision
    reason: str
    blocking: bool
    regressed_cases: list[str] = field(default_factory=list)
    fixed_cases: list[str] = field(default_factory=list)

    def as_row(self) -> dict[str, float | str | int]:
        return {
            "metric": self.metric,
            "baseline": round(self.baseline, 4),
            "candidate": round(self.candidate, 4),
            "delta": round(self.delta, 4),
            "p_value": round(self.test.p_value, 5) if self.test else float("nan"),
            "decision": self.decision,
            "blocking": int(self.blocking),
            "n_regressed": len(self.regressed_cases),
            "n_fixed": len(self.fixed_cases),
        }


@dataclass
class GateReport:
    findings: list[GateFinding]
    comparable: bool
    baseline_run_id: str
    candidate_run_id: str
    note: str = ""

    @property
    def decision(self) -> Decision:
        if not self.comparable:
            return "incomparable"
        blocking_failures = [f for f in self.findings if f.blocking and f.decision == "fail"]
        return "fail" if blocking_failures else "pass"

    @property
    def exit_code(self) -> int:
        """0 ships, 1 blocks, 2 means the comparison could not be made.

        Distinguishing 1 from 2 matters in CI: a blocked build is a signal about
        the change, while an incomparable one is a signal about the evaluation
        setup, and treating them alike trains everyone to ignore both.
        """
        return {"pass": 0, "fail": 1, "incomparable": 2, "insufficient_data": 2}[self.decision]

    def summary(self) -> str:
        if not self.comparable:
            return f"INCOMPARABLE: {self.note}"
        head = "PASS" if self.decision == "pass" else "FAIL"
        failed = [f.metric for f in self.findings if f.decision == "fail"]
        detail = f" -- blocked by {', '.join(failed)}" if failed else ""
        return f"{head}: {len(self.findings)} metrics compared{detail}"

    def markdown(self) -> str:
        """A CI-comment-ready table. This is what a reviewer actually reads."""
        lines = [
            f"### Evaluation gate: {self.decision.upper()}",
            "",
            f"Baseline `{self.baseline_run_id}` vs candidate `{self.candidate_run_id}`",
            "",
            "| Metric | Baseline | Candidate | Delta | p | Decision |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for finding in self.findings:
            p_text = f"{finding.test.p_value:.3f}" if finding.test else "n/a"
            lines.append(
                f"| {finding.metric} | {finding.baseline:.4f} | {finding.candidate:.4f} | "
                f"{finding.delta:+.4f} | {p_text} | {finding.decision} |"
            )
        regressed = [c for f in self.findings for c in f.regressed_cases]
        if regressed:
            lines += ["", f"**Newly failing cases ({len(regressed)}):** "
                          + ", ".join(f"`{c}`" for c in regressed[:15])]
        return "\n".join(lines)


def run_regression_gate(
    baseline_scores: dict[str, dict[str, float]],
    candidate_scores: dict[str, dict[str, float]],
    gates: Sequence[MetricGate],
    *,
    comparable: bool = True,
    baseline_run_id: str = "baseline",
    candidate_run_id: str = "candidate",
    incomparable_reason: str = "",
    min_cases: int = 30,
) -> GateReport:
    """Compare two runs case-by-case and decide whether the candidate ships.

    Args:
        baseline_scores: ``{metric: {case_id: score}}`` for the baseline run.
        candidate_scores: The same structure for the candidate.
        gates: Per-metric gate configuration.
        comparable: Set False when dataset hash or settings fingerprint differ.
        min_cases: Below this many shared cases the gate abstains rather than
            guessing; a 12-case suite cannot distinguish a regression from noise
            and pretending otherwise is worse than having no gate.
    """
    if not comparable:
        return GateReport([], False, baseline_run_id, candidate_run_id,
                          incomparable_reason or "dataset or settings fingerprint differ")

    findings: list[GateFinding] = []
    for gate in gates:
        base_map = baseline_scores.get(gate.metric, {})
        cand_map = candidate_scores.get(gate.metric, {})
        shared = sorted(set(base_map) & set(cand_map))

        if len(shared) < min_cases:
            findings.append(GateFinding(
                gate.metric, float("nan"), float("nan"), float("nan"), None,
                "insufficient_data",
                f"only {len(shared)} shared cases; need {min_cases}", gate.blocking,
            ))
            continue

        base_values = [base_map[c] for c in shared]
        cand_values = [cand_map[c] for c in shared]
        base_mean = sum(base_values) / len(base_values)
        cand_mean = sum(cand_values) / len(cand_values)
        delta = cand_mean - base_mean

        test = (
            mcnemar_test([int(v >= 0.5) for v in base_values], [int(v >= 0.5) for v in cand_values])
            if gate.binary
            else paired_bootstrap_test(base_values, cand_values)
        )

        regressed = [c for c in shared if cand_map[c] < base_map[c] - 1e-9]
        fixed = [c for c in shared if cand_map[c] > base_map[c] + 1e-9]

        if gate.floor is not None and cand_mean < gate.floor:
            decision, reason = "fail", f"below absolute floor {gate.floor:.4f}"
        elif delta >= -gate.max_regression:
            decision, reason = "pass", f"within regression budget ({delta:+.4f})"
        elif test.p_value >= gate.alpha:
            decision = "pass"
            reason = (f"drop {delta:+.4f} exceeds budget but is not significant "
                      f"(p={test.p_value:.3f}); suite lacks power to call it")
        else:
            decision = "fail"
            reason = f"significant regression {delta:+.4f} (p={test.p_value:.4f})"

        findings.append(GateFinding(
            gate.metric, base_mean, cand_mean, delta, test, decision, reason,  # type: ignore[arg-type]
            gate.blocking, regressed, fixed,
        ))

    return GateReport(findings, True, baseline_run_id, candidate_run_id)


def format_interval_row(name: str, interval: Interval) -> dict[str, float | str]:
    return {
        "metric": name,
        "value": round(interval.estimate, 4),
        "ci": f"[{interval.low:.4f}, {interval.high:.4f}]",
        "n": interval.n,
        "half_width": round(interval.half_width, 4),
    }
