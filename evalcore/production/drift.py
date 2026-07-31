"""Drift detection across the four surfaces that actually drift in an LLM system.

Model weights are usually the *last* thing to change. What changes constantly is
the input distribution, the retrieved-corpus embedding space, the output
distribution, and the prompt itself. Each needs a different detector, and each
needs a threshold you can defend at 3am.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
from scipy import stats as sps

Severity = Literal["none", "watch", "alert", "critical"]


@dataclass
class DriftResult:
    metric: str
    statistic: float
    p_value: float | None
    severity: Severity
    threshold: float
    n_reference: int
    n_current: int
    detail: dict[str, float] = field(default_factory=dict)

    @property
    def drifted(self) -> bool:
        return self.severity in ("alert", "critical")

    def as_row(self) -> dict[str, float | str]:
        return {
            "metric": self.metric,
            "statistic": round(self.statistic, 5),
            "p_value": round(self.p_value, 5) if self.p_value is not None else float("nan"),
            "severity": self.severity,
            "threshold": self.threshold,
            "n_ref": self.n_reference,
            "n_cur": self.n_current,
        }


def _severity_from_psi(psi: float) -> Severity:
    """Conventional PSI bands from credit-risk monitoring.

    < 0.1 stable, 0.1-0.25 moderate shift worth watching, > 0.25 significant.
    These bands predate LLMs by decades and have held up; they are a defensible
    default when nobody has yet measured what a harmful shift looks like for
    your system.
    """
    if psi < 0.10:
        return "none"
    if psi < 0.25:
        return "watch"
    return "alert" if psi < 0.5 else "critical"


def population_stability_index(
    reference: Sequence[float], current: Sequence[float], *, n_bins: int = 10
) -> DriftResult:
    """PSI over quantile bins of a numeric feature.

    Bins are cut on the *reference* distribution and then frozen. Re-cutting
    bins on each new window is a classic bug that makes PSI structurally
    incapable of detecting the shift it exists to detect.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if ref.size < n_bins or cur.size == 0:
        return DriftResult("psi", 0.0, None, "none", 0.25, ref.size, cur.size,
                           {"note": 1.0})

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    # Laplace smoothing: an empty bin makes the log term infinite, which turns
    # a small sample into a fake critical alert.
    ref_frac = (ref_counts + 0.5) / (ref_counts.sum() + 0.5 * len(ref_counts))
    cur_frac = (cur_counts + 0.5) / (cur_counts.sum() + 0.5 * len(cur_counts))
    psi = float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))

    return DriftResult("psi", psi, None, _severity_from_psi(psi), 0.25, ref.size, cur.size,
                       {"max_bin_shift": float(np.max(np.abs(cur_frac - ref_frac)))})


def ks_drift(reference: Sequence[float], current: Sequence[float], *, alpha: float = 0.01) -> DriftResult:
    """Two-sample Kolmogorov-Smirnov test on a numeric distribution.

    More sensitive than PSI and it produces a p-value, but that sensitivity is a
    liability at production volumes: with 200k requests per window, KS flags
    shifts far too small to matter. Use alpha = 0.01 with an effect-size floor on
    the KS statistic, or use PSI.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if ref.size < 8 or cur.size < 8:
        return DriftResult("ks", 0.0, None, "none", alpha, ref.size, cur.size)
    statistic, p_value = sps.ks_2samp(ref, cur)
    severity: Severity = "none"
    if p_value < alpha:
        severity = "alert" if statistic < 0.25 else "critical"
    elif p_value < 0.05:
        severity = "watch"
    return DriftResult("ks", float(statistic), float(p_value), severity, alpha, ref.size, cur.size)


def chi_square_drift(
    reference: Sequence[str], current: Sequence[str], *, alpha: float = 0.01
) -> DriftResult:
    """Chi-square test for categorical drift (intent labels, routes, languages)."""
    categories = sorted(set(reference) | set(current))
    if len(categories) < 2:
        return DriftResult("chi2", 0.0, 1.0, "none", alpha, len(reference), len(current))
    ref_counts = np.array([sum(1 for v in reference if v == c) for c in categories], dtype=float)
    cur_counts = np.array([sum(1 for v in current if v == c) for c in categories], dtype=float)
    expected = ref_counts / max(ref_counts.sum(), 1) * cur_counts.sum()
    expected = np.maximum(expected, 0.5)
    statistic = float(np.sum((cur_counts - expected) ** 2 / expected))
    p_value = float(sps.chi2.sf(statistic, df=len(categories) - 1))
    severity: Severity = "none" if p_value >= 0.05 else ("watch" if p_value >= alpha else "alert")
    return DriftResult("chi2", statistic, p_value, severity, alpha, len(reference), len(current),
                       {"n_categories": float(len(categories))})


def embedding_drift(
    reference_embeddings: np.ndarray, current_embeddings: np.ndarray, *, threshold: float = 0.05
) -> DriftResult:
    """Centroid-distance and MMD drift over an embedding space.

    This is the detector for *semantic* drift: the questions users ask, or the
    documents the corpus contains, have moved in meaning even though every
    surface statistic is unchanged. It is the earliest reliable warning that a
    RAG system's retrieval quality is about to degrade, typically firing weeks
    before user-visible metrics move.
    """
    ref = np.asarray(reference_embeddings, dtype=float)
    cur = np.asarray(current_embeddings, dtype=float)
    if ref.ndim != 2 or cur.ndim != 2 or ref.shape[1] != cur.shape[1]:
        raise ValueError("embedding matrices must be 2-D with matching dimensionality")
    if ref.shape[0] < 5 or cur.shape[0] < 5:
        return DriftResult("embedding_mmd", 0.0, None, "none", threshold, ref.shape[0], cur.shape[0])

    ref_unit = ref / np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-12)
    cur_unit = cur / np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
    centroid_distance = float(1.0 - float(ref_unit.mean(axis=0) @ cur_unit.mean(axis=0)))
    mmd = _linear_mmd(ref_unit, cur_unit)

    severity: Severity = "none"
    if mmd >= threshold * 3:
        severity = "critical"
    elif mmd >= threshold:
        severity = "alert"
    elif mmd >= threshold / 2:
        severity = "watch"

    return DriftResult("embedding_mmd", mmd, None, severity, threshold, ref.shape[0], cur.shape[0],
                       {"centroid_cosine_distance": round(centroid_distance, 5)})


def _linear_mmd(x: np.ndarray, y: np.ndarray) -> float:
    """Unbiased linear-kernel MMD^2 between two sample sets."""
    n, m = x.shape[0], y.shape[0]
    xx = (x @ x.T).sum() - np.trace(x @ x.T)
    yy = (y @ y.T).sum() - np.trace(y @ y.T)
    xy = (x @ y.T).sum()
    return float(xx / (n * (n - 1)) + yy / (m * (m - 1)) - 2 * xy / (n * m))


# ---------------------------------------------------------------------------
# Prompt drift
# ---------------------------------------------------------------------------
@dataclass
class PromptVersion:
    name: str
    template: str
    recorded_at: str

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.template.encode("utf-8")).hexdigest()[:16]


def prompt_drift(versions: Sequence[PromptVersion]) -> list[dict[str, str | int]]:
    """Detect silent prompt changes by hashing every version.

    The most common undiagnosed production regression in LLM systems is that
    somebody edited a prompt template and nobody linked the metric drop to it.
    Hash the assembled prompt at request time, log the hash, and this becomes a
    one-query root cause instead of a two-day investigation.
    """
    rows: list[dict[str, str | int]] = []
    previous: PromptVersion | None = None
    for version in versions:
        changed = previous is not None and previous.digest != version.digest
        rows.append({
            "name": version.name,
            "recorded_at": version.recorded_at,
            "digest": version.digest,
            "changed": int(changed),
            "char_delta": len(version.template) - len(previous.template) if previous else 0,
        })
        previous = version
    return rows


@dataclass
class DriftDashboard:
    """The set of detectors run on every monitoring window."""

    results: list[DriftResult] = field(default_factory=list)

    def add(self, result: DriftResult) -> "DriftDashboard":
        self.results.append(result)
        return self

    @property
    def worst_severity(self) -> Severity:
        order = {"none": 0, "watch": 1, "alert": 2, "critical": 3}
        return max((r.severity for r in self.results), key=lambda s: order[s], default="none")

    def should_page(self) -> bool:
        """Page a human only on critical, or on two simultaneous alerts.

        Single-detector alerting produces enough noise that the on-call rota
        starts ignoring it, which is the real failure mode of drift monitoring.
        Correlated evidence across two detectors is a far better predictor of a
        genuine incident.
        """
        criticals = sum(1 for r in self.results if r.severity == "critical")
        alerts = sum(1 for r in self.results if r.severity == "alert")
        return criticals >= 1 or alerts >= 2

    def as_rows(self) -> list[dict[str, float | str]]:
        return [result.as_row() for result in self.results]
