"""Dataset splitting and leakage detection.

Most "mysteriously excellent" offline numbers are leakage, not skill. This
module implements the three splitting strategies that actually matter in
production evaluation -- stratified, grouped, and temporal -- plus a leakage
auditor that runs as a hard gate before any suite is trusted.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Hashable, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Split:
    """Index arrays for one train/validation/test partition."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    strategy: str

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": self.train.size, "validation": self.validation.size, "test": self.test.size}


def _hash_to_unit(value: Hashable, salt: str) -> float:
    """Map any hashable to a stable float in [0, 1).

    Hash-based assignment beats a shuffled index because it is *stable under
    dataset growth*: adding 200 new rows tomorrow does not move yesterday's rows
    between splits, so today's test scores stay comparable with tomorrow's.
    """
    digest = hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(1 << 64)


def hash_split(
    keys: Sequence[Hashable],
    *,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    salt: str = "eval-v1",
) -> Split:
    """Deterministic, growth-stable split keyed on a per-row identifier."""
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1.0")
    train_cut, val_cut = fractions[0], fractions[0] + fractions[1]
    buckets: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for i, key in enumerate(keys):
        u = _hash_to_unit(key, salt)
        target = "train" if u < train_cut else ("validation" if u < val_cut else "test")
        buckets[target].append(i)
    return Split(
        np.array(buckets["train"], dtype=int),
        np.array(buckets["validation"], dtype=int),
        np.array(buckets["test"], dtype=int),
        "hash",
    )


def stratified_split(
    labels: Sequence[Hashable],
    *,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 1337,
) -> Split:
    """Preserve the label distribution in every partition.

    Required whenever a class is rare. A random split of a 2%-positive fraud
    dataset into 100 test rows can easily yield zero positives, at which point
    recall is undefined and the dashboard shows a green tick for a model that
    never fires.
    """
    rng = np.random.default_rng(seed)
    by_label: dict[Hashable, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        by_label[label].append(i)

    train, validation, test = [], [], []
    for label, indices in by_label.items():
        idx = np.array(indices)
        rng.shuffle(idx)
        n = idx.size
        n_train = int(round(n * fractions[0]))
        n_val = int(round(n * fractions[1]))
        train.extend(idx[:n_train].tolist())
        validation.extend(idx[n_train : n_train + n_val].tolist())
        test.extend(idx[n_train + n_val :].tolist())
    return Split(np.sort(np.array(train, int)), np.sort(np.array(validation, int)),
                 np.sort(np.array(test, int)), "stratified")


def grouped_split(
    groups: Sequence[Hashable],
    *,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    salt: str = "eval-v1",
) -> Split:
    """Keep every row of a group on one side of the split.

    The classic production leak: a support-ticket dataset with 12 messages per
    conversation, split row-wise. The model sees message 3 in training and is
    scored on message 4 of the same conversation, and offline accuracy runs 15
    points above live accuracy forever. Split on ``conversation_id``, not row.
    """
    unique = list(dict.fromkeys(groups))
    assignment = {g: _hash_to_unit(g, salt) for g in unique}
    train_cut, val_cut = fractions[0], fractions[0] + fractions[1]
    buckets: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for i, group in enumerate(groups):
        u = assignment[group]
        target = "train" if u < train_cut else ("validation" if u < val_cut else "test")
        buckets[target].append(i)
    return Split(np.array(buckets["train"], int), np.array(buckets["validation"], int),
                 np.array(buckets["test"], int), "grouped")


def temporal_split(
    timestamps: Sequence[float],
    *,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    embargo: float = 0.0,
) -> Split:
    """Chronological split with an optional embargo gap.

    Any system whose inputs drift -- which is all of them -- must be evaluated
    forward in time. The ``embargo`` drops rows straddling each boundary, which
    removes the short-horizon correlation leak in time-series and session data.
    """
    order = np.argsort(np.asarray(timestamps, dtype=float))
    ordered_times = np.asarray(timestamps, dtype=float)[order]
    n = order.size
    train_end = int(n * fractions[0])
    val_end = train_end + int(n * fractions[1])

    def _apply_embargo(start: int, end: int, boundary_time: float) -> np.ndarray:
        if embargo <= 0:
            return order[start:end]
        keep = ordered_times[start:end] > boundary_time + embargo
        return order[start:end][keep]

    train = order[:train_end]
    boundary_train = ordered_times[train_end - 1] if train_end > 0 else -np.inf
    validation = _apply_embargo(train_end, val_end, boundary_train)
    boundary_val = ordered_times[val_end - 1] if val_end > 0 else boundary_train
    test = _apply_embargo(val_end, n, boundary_val)
    return Split(train, validation, test, f"temporal(embargo={embargo})")


# ---------------------------------------------------------------------------
# Leakage detection
# ---------------------------------------------------------------------------
@dataclass
class LeakageReport:
    """Findings from the pre-flight leakage audit."""

    exact_duplicates: list[tuple[int, int]] = field(default_factory=list)
    near_duplicates: list[tuple[int, int, float]] = field(default_factory=list)
    group_overlap: list[Hashable] = field(default_factory=list)
    temporal_violations: int = 0
    n_train: int = 0
    n_test: int = 0

    @property
    def clean(self) -> bool:
        return not (self.exact_duplicates or self.near_duplicates
                    or self.group_overlap or self.temporal_violations)

    @property
    def contamination_rate(self) -> float:
        """Fraction of test rows touched by any leak."""
        if self.n_test == 0:
            return 0.0
        touched = {j for _, j in self.exact_duplicates}
        touched |= {j for _, j, _ in self.near_duplicates}
        return len(touched) / self.n_test

    def summary(self) -> str:
        if self.clean:
            return f"CLEAN: no leakage across {self.n_train} train / {self.n_test} test rows."
        return (
            f"LEAKAGE: {len(self.exact_duplicates)} exact, {len(self.near_duplicates)} near-duplicate, "
            f"{len(self.group_overlap)} group overlaps, {self.temporal_violations} temporal violations "
            f"({self.contamination_rate:.1%} of test contaminated)."
        )


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _shingles(text: str, k: int = 5) -> set[str]:
    """Word-level k-shingles, the unit of near-duplicate comparison."""
    tokens = _TOKEN_RE.findall(text.lower())
    if len(tokens) < k:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def detect_leakage(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    *,
    train_groups: Sequence[Hashable] | None = None,
    test_groups: Sequence[Hashable] | None = None,
    train_times: Sequence[float] | None = None,
    test_times: Sequence[float] | None = None,
    near_duplicate_threshold: float = 0.8,
    shingle_k: int = 5,
) -> LeakageReport:
    """Audit a train/test partition for the four leaks that matter.

    1. **Exact duplicates** -- normalised string equality (whitespace and case
       folded), found in O(n) with a hash index.
    2. **Near duplicates** -- Jaccard similarity over word k-shingles, with an
       inverted index so only candidates sharing at least one shingle are
       compared. This is the leak that survives a naive ``set()`` de-duplication
       and is the usual explanation for a benchmark score nobody can reproduce.
    3. **Group overlap** -- the same entity, conversation, or user on both sides.
    4. **Temporal violations** -- training rows dated after test rows.

    Run this as a *gate*, not a report: a suite with contamination above a
    threshold should fail the build.
    """
    report = LeakageReport(n_train=len(train_texts), n_test=len(test_texts))

    def normalise(text: str) -> str:
        return " ".join(text.lower().split())

    train_index: dict[str, list[int]] = defaultdict(list)
    for i, text in enumerate(train_texts):
        train_index[normalise(text)].append(i)
    for j, text in enumerate(test_texts):
        for i in train_index.get(normalise(text), []):
            report.exact_duplicates.append((i, j))

    # Inverted shingle index keeps this near-linear instead of O(n*m).
    train_shingles = [_shingles(t, shingle_k) for t in train_texts]
    shingle_to_train: dict[str, set[int]] = defaultdict(set)
    for i, shingles in enumerate(train_shingles):
        for shingle in shingles:
            shingle_to_train[shingle].add(i)

    exact_pairs = set(report.exact_duplicates)
    for j, text in enumerate(test_texts):
        test_shingles = _shingles(text, shingle_k)
        candidates: set[int] = set()
        for shingle in test_shingles:
            candidates |= shingle_to_train.get(shingle, set())
        for i in candidates:
            if (i, j) in exact_pairs:
                continue
            score = jaccard(train_shingles[i], test_shingles)
            if score >= near_duplicate_threshold:
                report.near_duplicates.append((i, j, round(score, 4)))

    if train_groups is not None and test_groups is not None:
        report.group_overlap = sorted(set(train_groups) & set(test_groups), key=str)

    if train_times is not None and test_times is not None and len(test_times):
        earliest_test = min(test_times)
        report.temporal_violations = int(sum(1 for t in train_times if t > earliest_test))

    return report


def class_balance(labels: Iterable[Hashable]) -> dict[Hashable, float]:
    """Label proportions -- the first thing to check when a split looks odd."""
    counts = Counter(labels)
    total = sum(counts.values())
    return {label: count / total for label, count in counts.most_common()} if total else {}
