"""SQLite experiment store: runs, per-case results, and metric aggregates.

Experiment tracking is not a nice-to-have. Without a durable record keyed on
(suite, dataset hash, settings fingerprint, git commit) you cannot answer the
only question that matters after a regression: *what changed?* The schema here
is deliberately the minimum that supports that question, and it maps one-to-one
onto the MLflow / Weights & Biases concepts covered in Chapter 9.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from evalcore.config import get_settings
from evalcore.runner.runner import RunResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id                TEXT PRIMARY KEY,
    suite                 TEXT NOT NULL,
    dataset_hash          TEXT NOT NULL,
    settings_fingerprint  TEXT NOT NULL,
    git_commit            TEXT,
    label                 TEXT,
    started_at            TEXT NOT NULL,
    finished_at           TEXT NOT NULL,
    n_total               INTEGER NOT NULL,
    n_ok                  INTEGER NOT NULL,
    error_rate            REAL NOT NULL,
    wall_clock_s          REAL NOT NULL,
    summary_json          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS case_results (
    run_id      TEXT NOT NULL,
    case_id     TEXT NOT NULL,
    status      TEXT NOT NULL,
    latency_ms  REAL NOT NULL,
    attempts    INTEGER NOT NULL,
    error       TEXT,
    scores_json TEXT NOT NULL,
    slices_json TEXT NOT NULL,
    output      TEXT,
    PRIMARY KEY (run_id, case_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS metrics (
    run_id  TEXT NOT NULL,
    metric  TEXT NOT NULL,
    slice   TEXT NOT NULL DEFAULT '__overall__',
    value   REAL NOT NULL,
    ci_low  REAL,
    ci_high REAL,
    n       INTEGER NOT NULL,
    PRIMARY KEY (run_id, metric, slice),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_runs_suite ON runs(suite, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_metrics_metric ON metrics(metric, slice);
"""


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    suite: str
    dataset_hash: str
    settings_fingerprint: str
    git_commit: str | None
    label: str | None
    started_at: str
    n_total: int
    n_ok: int
    error_rate: float

    def comparable_with(self, other: "RunRecord") -> bool:
        """Two runs are comparable only if the ruler did not move.

        The regression gate refuses to compare across dataset or settings
        changes. Comparing a run on dataset v3 against a baseline on v2 is the
        most common way teams convince themselves of an improvement that does
        not exist.
        """
        return (self.suite == other.suite
                and self.dataset_hash == other.dataset_hash
                and self.settings_fingerprint == other.settings_fingerprint)


def current_git_commit() -> str | None:
    """Short git SHA, or None outside a repository. Never raises."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - provenance is best effort
        return None


class ExperimentStore:
    """Durable store for evaluation runs."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = str(path or get_settings().store_path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save_run(
        self,
        result: RunResult,
        *,
        label: str | None = None,
        metrics: dict[str, dict[str, Any]] | None = None,
    ) -> str:
        """Persist a run and return its ``run_id``.

        Args:
            result: The runner output.
            label: Human-readable tag, e.g. ``"baseline"`` or ``"pr-482"``.
            metrics: Optional aggregate metrics, each
                ``{"value": x, "ci_low": l, "ci_high": h, "n": n, "slice": s}``.
        """
        run_id = f"{result.suite}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{result.dataset_hash[:6]}"
        summary = result.summary()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id, result.suite, result.dataset_hash, result.settings_fingerprint,
                    current_git_commit(), label, result.started_at, result.finished_at,
                    result.n_total, result.n_ok, result.error_rate, result.wall_clock_s,
                    json.dumps(summary, sort_keys=True),
                ),
            )
            conn.executemany(
                "INSERT OR REPLACE INTO case_results VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    (
                        run_id, case.case_id, case.status, case.latency_ms, case.attempts,
                        case.error, json.dumps(case.scores, sort_keys=True),
                        json.dumps(case.slice_tags), case.output[:8000],
                    )
                    for case in result.results
                ],
            )
            if metrics:
                conn.executemany(
                    "INSERT OR REPLACE INTO metrics VALUES (?,?,?,?,?,?,?)",
                    [
                        (
                            run_id, name, payload.get("slice", "__overall__"),
                            float(payload["value"]), payload.get("ci_low"),
                            payload.get("ci_high"), int(payload.get("n", result.n_total)),
                        )
                        for name, payload in metrics.items()
                    ],
                )
        return run_id

    def list_runs(self, suite: str | None = None, limit: int = 50) -> list[RunRecord]:
        query = "SELECT * FROM runs"
        params: tuple = ()
        if suite:
            query += " WHERE suite = ?"
            params = (suite,)
        query += " ORDER BY started_at DESC LIMIT ?"
        with self._connect() as conn:
            rows = conn.execute(query, (*params, limit)).fetchall()
        return [
            RunRecord(
                r["run_id"], r["suite"], r["dataset_hash"], r["settings_fingerprint"],
                r["git_commit"], r["label"], r["started_at"], r["n_total"], r["n_ok"], r["error_rate"],
            )
            for r in rows
        ]

    def case_scores(self, run_id: str, metric: str) -> dict[str, float]:
        """Per-case scores for one metric, keyed by ``case_id``.

        Returned as a dict so a caller can align two runs on ``case_id`` before
        a paired test -- alignment by list position is a silent correctness bug
        the moment one run skips a case.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT case_id, status, scores_json FROM case_results WHERE run_id = ?", (run_id,)
            ).fetchall()
        scores: dict[str, float] = {}
        for row in rows:
            payload = json.loads(row["scores_json"])
            if row["status"] == "ok" and metric in payload:
                scores[row["case_id"]] = float(payload[metric])
            elif row["status"] != "ok":
                scores[row["case_id"]] = 0.0
        return scores

    def metric_history(self, suite: str, metric: str, *, slice_name: str = "__overall__",
                       limit: int = 100) -> list[dict[str, Any]]:
        """Time series for one metric -- the data behind a trend chart."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT r.run_id, r.started_at, r.label, r.git_commit,
                       m.value, m.ci_low, m.ci_high, m.n
                FROM metrics m JOIN runs r ON r.run_id = m.run_id
                WHERE r.suite = ? AND m.metric = ? AND m.slice = ?
                ORDER BY r.started_at ASC LIMIT ?
                """,
                (suite, metric, slice_name, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def latest_run(self, suite: str, *, label: str | None = None) -> RunRecord | None:
        runs = self.list_runs(suite, limit=200)
        for record in runs:
            if label is None or record.label == label:
                return record
        return None

    def delete_run(self, run_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
