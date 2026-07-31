"""Async evaluation runner: concurrency, retries, rate limiting, partial failure.

This is the component that turns "a metric function" into "an evaluation
system". Its contract is the one that matters in production:

* a failed row is **recorded as a failure**, never silently dropped -- dropping
  errors is how a suite reports 0.92 while a third of it never ran;
* every row carries **latency, token, retry and error** metadata, so cost and
  reliability are first-class outputs rather than an afterthought;
* the whole run is **resumable** and **cancellable**, because a 4,000-row suite
  against a free-tier quota does not finish in one sitting.
"""

from __future__ import annotations

import asyncio
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Iterable, Sequence

from evalcore.config import Settings, get_settings
from evalcore.datasets import EvalCase
from evalcore.llm import RateLimiter

CaseRunner = Callable[[EvalCase], Awaitable[dict[str, Any]]]
ProgressHook = Callable[[int, int, "CaseResult"], None]


@dataclass
class CaseResult:
    """The outcome of evaluating one case, success or failure."""

    case_id: str
    status: str  # "ok" | "error" | "skipped"
    scores: dict[str, float] = field(default_factory=dict)
    output: str = ""
    slice_tags: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    attempts: int = 1
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass
class RunResult:
    """Everything produced by one suite execution."""

    suite: str
    dataset_hash: str
    settings_fingerprint: str
    started_at: str
    finished_at: str
    results: list[CaseResult]

    @property
    def n_total(self) -> int:
        return len(self.results)

    @property
    def n_ok(self) -> int:
        return sum(1 for r in self.results if r.ok)

    @property
    def error_rate(self) -> float:
        return 0.0 if not self.results else 1 - self.n_ok / self.n_total

    @property
    def wall_clock_s(self) -> float:
        start = datetime.fromisoformat(self.started_at)
        end = datetime.fromisoformat(self.finished_at)
        return (end - start).total_seconds()

    def scores(self, metric: str, *, include_errors: bool = True,
               error_value: float = 0.0) -> list[float]:
        """Per-item scores for one metric.

        ``include_errors=True`` is the default on purpose. An item the system
        could not answer is an item the system got wrong from the user's point
        of view, and scoring only successful rows produces the classic
        survivorship-biased dashboard that reports 0.94 for a service that
        errors on 20% of requests.
        """
        values: list[float] = []
        for result in self.results:
            if result.ok:
                if metric in result.scores:
                    values.append(float(result.scores[metric]))
            elif include_errors:
                values.append(error_value)
        return values

    def summary(self) -> dict[str, Any]:
        latencies = sorted(r.latency_ms for r in self.results if r.ok)
        def _pct(p: float) -> float:
            return latencies[min(int(len(latencies) * p), len(latencies) - 1)] if latencies else 0.0
        return {
            "suite": self.suite,
            "n_total": self.n_total,
            "n_ok": self.n_ok,
            "error_rate": round(self.error_rate, 4),
            "wall_clock_s": round(self.wall_clock_s, 2),
            "latency_p50_ms": round(_pct(0.50), 1),
            "latency_p95_ms": round(_pct(0.95), 1),
            "total_retries": sum(r.attempts - 1 for r in self.results),
            "dataset_hash": self.dataset_hash,
            "settings_fingerprint": self.settings_fingerprint,
        }

    def to_records(self) -> list[dict[str, Any]]:
        return [asdict(result) for result in self.results]


class EvaluationRunner:
    """Bounded-concurrency, rate-limited, retrying batch evaluator."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        max_concurrency: int | None = None,
        requests_per_minute: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.max_concurrency = max_concurrency or self.settings.max_concurrency
        self.max_retries = self.settings.max_retries if max_retries is None else max_retries
        self._limiter = RateLimiter(requests_per_minute or self.settings.requests_per_minute)

    async def _run_one(self, case: EvalCase, runner: CaseRunner) -> CaseResult:
        """Execute a single case with bounded exponential backoff.

        Only transient failures are retried. A schema-validation error means the
        prompt or parser is wrong and retrying it five times just burns quota to
        produce the same failure -- it is recorded immediately.
        """
        attempts = 0
        start = time.perf_counter()
        last_error = ""

        while attempts <= self.max_retries:
            attempts += 1
            await self._limiter.acquire()
            try:
                payload = await runner(case)
                return CaseResult(
                    case_id=case.case_id,
                    status="ok",
                    scores={k: float(v) for k, v in payload.get("scores", {}).items()},
                    output=str(payload.get("output", "")),
                    slice_tags=list(case.slice_tags),
                    latency_ms=(time.perf_counter() - start) * 1000,
                    attempts=attempts,
                    metadata=payload.get("metadata", {}),
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a failed row is data, not a crash
                last_error = f"{type(exc).__name__}: {exc}"
                if not _is_transient(exc) or attempts > self.max_retries:
                    break
                await asyncio.sleep(min(2 ** (attempts - 1) * 0.75, 20.0))

        return CaseResult(
            case_id=case.case_id,
            status="error",
            slice_tags=list(case.slice_tags),
            latency_ms=(time.perf_counter() - start) * 1000,
            attempts=attempts,
            error=last_error,
            metadata={"traceback": traceback.format_exc(limit=3)},
        )

    async def run_async(
        self,
        suite: str,
        cases: Sequence[EvalCase],
        runner: CaseRunner,
        *,
        dataset_hash: str = "",
        progress: ProgressHook | None = None,
    ) -> RunResult:
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        semaphore = asyncio.Semaphore(self.max_concurrency)
        completed = 0
        total = len(cases)
        results: list[CaseResult] = [None] * total  # type: ignore[list-item]

        async def _guarded(index: int, case: EvalCase) -> None:
            nonlocal completed
            async with semaphore:
                result = await self._run_one(case, runner)
            results[index] = result
            completed += 1
            if progress:
                progress(completed, total, result)

        await asyncio.gather(*(_guarded(i, case) for i, case in enumerate(cases)))
        finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return RunResult(suite, dataset_hash, self.settings.fingerprint(), started, finished, list(results))

    def run(
        self,
        suite: str,
        cases: Sequence[EvalCase],
        runner: CaseRunner,
        *,
        dataset_hash: str = "",
        progress: ProgressHook | None = None,
    ) -> RunResult:
        """Synchronous entry point used by Streamlit and pytest.

        Streamlit runs its own script thread without an event loop, so a fresh
        loop is created per call rather than reusing ``asyncio.run`` semantics
        that would clash with a notebook or an already-running loop.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.run_async(suite, cases, runner, dataset_hash=dataset_hash, progress=progress)
            )
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.run_async(suite, cases, runner, dataset_hash=dataset_hash, progress=progress)
            )
        finally:
            loop.close()


_TRANSIENT_MARKERS = (
    "429", "rate limit", "resource exhausted", "quota",
    "timeout", "timed out", "deadline exceeded",
    "503", "502", "500", "unavailable", "internal error",
    "connection reset", "connection aborted", "temporarily",
)


def _is_transient(exc: Exception) -> bool:
    """Classify an exception as retryable.

    String matching is unglamorous but correct here: the provider SDKs raise a
    dozen exception types across versions, and the stable signal is the status
    code and message. Anything unrecognised is treated as permanent so a genuine
    bug fails fast instead of retrying five times.
    """
    if isinstance(exc, (ValueError, TypeError, KeyError, AttributeError)):
        return False
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def batched(items: Iterable[Any], size: int) -> Iterable[list[Any]]:
    """Chunk an iterable; used by the batch-evaluation and indexing paths."""
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
