#!/usr/bin/env python3
"""Continuous-evaluation gate for CI.

Compares the two most recent runs of a suite in the experiment store and exits
with a code CI can act on:

    0  ship        -- no blocking regression
    1  blocked     -- a significant regression exceeded its budget
    2  incomparable -- dataset hash or settings fingerprint moved; the comparison
                       is invalid and must not be interpreted either way

Usage:
    python scripts/ci_gate.py --suite rag_qa-harness --metric contains_expected \\
        --budget 0.02 --floor 0.70 --comment-path gate.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalcore.report import MetricGate, run_regression_gate  # noqa: E402
from evalcore.runner import ExperimentStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--budget", type=float, default=0.02,
                        help="Maximum tolerated regression before the gate blocks")
    parser.add_argument("--floor", type=float, default=None,
                        help="Absolute minimum; below this the gate blocks regardless of delta")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-cases", type=int, default=30)
    parser.add_argument("--binary", action="store_true",
                        help="Metric is pass/fail; use McNemar instead of a paired bootstrap")
    parser.add_argument("--baseline-label", default=None)
    parser.add_argument("--comment-path", default=None,
                        help="Write the markdown gate report here for a PR comment")
    args = parser.parse_args()

    store = ExperimentStore()
    runs = store.list_runs(args.suite, limit=50)
    if len(runs) < 2:
        print(f"::warning::only {len(runs)} run(s) recorded for suite '{args.suite}'; "
              "nothing to compare", file=sys.stderr)
        return 2

    candidate = runs[0]
    baseline = next((r for r in runs[1:] if args.baseline_label is None
                     or r.label == args.baseline_label), None)
    if baseline is None:
        print(f"::warning::no baseline run with label '{args.baseline_label}'", file=sys.stderr)
        return 2

    report = run_regression_gate(
        {args.metric: store.case_scores(baseline.run_id, args.metric)},
        {args.metric: store.case_scores(candidate.run_id, args.metric)},
        [MetricGate(args.metric, floor=args.floor, max_regression=args.budget,
                    alpha=args.alpha, binary=args.binary)],
        comparable=baseline.comparable_with(candidate),
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        incomparable_reason=(
            f"dataset hash {baseline.dataset_hash} -> {candidate.dataset_hash}, "
            f"settings {baseline.settings_fingerprint} -> {candidate.settings_fingerprint}"
        ),
        min_cases=args.min_cases,
    )

    print(report.summary())
    for finding in report.findings:
        print(f"  {finding.metric}: {finding.baseline:.4f} -> {finding.candidate:.4f} "
              f"({finding.delta:+.4f}) — {finding.reason}")
        if finding.regressed_cases:
            print(f"    newly failing: {', '.join(finding.regressed_cases[:20])}")

    if args.comment_path:
        Path(args.comment_path).write_text(report.markdown(), encoding="utf-8")
        print(f"Wrote gate comment to {args.comment_path}")

    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
