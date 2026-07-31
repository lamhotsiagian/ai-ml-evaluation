"""Golden dataset registry with content-addressed versioning.

An evaluation number is meaningless without the dataset version that produced
it. This registry gives every dataset a content hash, so "score went from 0.81
to 0.88" can always be decomposed into "the system changed" versus "the ruler
changed" -- a distinction that has ended more than one release argument.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

from pydantic import BaseModel, Field, field_validator

from evalcore.config import get_settings

Difficulty = Literal["easy", "medium", "hard"]


class EvalCase(BaseModel):
    """One row of a golden dataset.

    Deliberately wide enough to carry classification, generation, RAG and agent
    cases in a single schema, because in practice a production suite mixes them
    and splitting the schema means splitting the tooling.
    """

    case_id: str
    input: str
    expected_output: str | None = None
    expected_label: str | None = None
    contexts: list[str] = Field(default_factory=list)
    expected_tools: list[str] = Field(default_factory=list)
    slice_tags: list[str] = Field(default_factory=list)
    difficulty: Difficulty = "medium"
    group_id: str | None = None
    is_answerable: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("case_id")
    @classmethod
    def _non_empty_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("case_id must be non-empty; it is the join key for every result table")
        return value.strip()

    def content_digest(self) -> str:
        """Hash of the fields that define the *test*, not its bookkeeping."""
        payload = {
            "input": self.input,
            "expected_output": self.expected_output,
            "expected_label": self.expected_label,
            "contexts": self.contexts,
            "expected_tools": self.expected_tools,
            "is_answerable": self.is_answerable,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


@dataclass
class DatasetVersion:
    """Immutable metadata describing one materialised dataset version."""

    name: str
    version: str
    n_cases: int
    content_hash: str
    created_at: str
    path: str
    slice_counts: dict[str, int] = field(default_factory=dict)
    difficulty_counts: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


class GoldenDataset:
    """An ordered, hashed collection of :class:`EvalCase` rows."""

    def __init__(self, name: str, cases: list[EvalCase], version: str = "v1") -> None:
        self.name = name
        self.version = version
        self._cases = cases
        self._validate_unique_ids()

    def _validate_unique_ids(self) -> None:
        seen: set[str] = set()
        duplicates = sorted({c.case_id for c in self._cases if c.case_id in seen or seen.add(c.case_id)})
        if duplicates:
            raise ValueError(f"duplicate case_id values in {self.name}: {duplicates[:5]}")

    def __len__(self) -> int:
        return len(self._cases)

    def __iter__(self) -> Iterator[EvalCase]:
        return iter(self._cases)

    def __getitem__(self, index: int) -> EvalCase:
        return self._cases[index]

    @property
    def cases(self) -> list[EvalCase]:
        return list(self._cases)

    def content_hash(self) -> str:
        """Order-independent hash over per-case digests.

        Order independence matters: re-exporting a dataset from a database with
        a different ``ORDER BY`` must not look like a new dataset version.
        """
        digests = sorted(case.content_digest() for case in self._cases)
        return hashlib.sha256("".join(digests).encode()).hexdigest()[:16]

    def filter(self, *, slice_tag: str | None = None, difficulty: Difficulty | None = None,
               answerable: bool | None = None) -> "GoldenDataset":
        selected = [
            case for case in self._cases
            if (slice_tag is None or slice_tag in case.slice_tags)
            and (difficulty is None or case.difficulty == difficulty)
            and (answerable is None or case.is_answerable == answerable)
        ]
        return GoldenDataset(f"{self.name}[{slice_tag or ''}{difficulty or ''}]", selected, self.version)

    def slice_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for case in self._cases:
            for tag in case.slice_tags:
                counts[tag] = counts.get(tag, 0) + 1
        return dict(sorted(counts.items()))

    def difficulty_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for case in self._cases:
            counts[case.difficulty] = counts.get(case.difficulty, 0) + 1
        return dict(sorted(counts.items()))

    def describe(self) -> DatasetVersion:
        return DatasetVersion(
            name=self.name,
            version=self.version,
            n_cases=len(self._cases),
            content_hash=self.content_hash(),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            path="(in memory)",
            slice_counts=self.slice_counts(),
            difficulty_counts=self.difficulty_counts(),
        )

    # -- serialisation -----------------------------------------------------
    def to_jsonl(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for case in self._cases:
                handle.write(case.model_dump_json() + "\n")
        return path

    @classmethod
    def from_jsonl(cls, path: Path, *, name: str | None = None, version: str = "v1") -> "GoldenDataset":
        cases: list[EvalCase] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                try:
                    cases.append(EvalCase.model_validate_json(line))
                except Exception as exc:  # noqa: BLE001 - we want the line number
                    raise ValueError(f"{path.name}:{line_no} is not a valid EvalCase: {exc}") from exc
        return cls(name or path.stem, cases, version)


class DatasetRegistry:
    """Filesystem-backed registry over ``data/golden/*.jsonl``."""

    def __init__(self, root: Path | None = None) -> None:
        settings = get_settings()
        self.root = Path(root) if root else settings.data_dir / "golden"
        self.root.mkdir(parents=True, exist_ok=True)

    def available(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.jsonl"))

    def load(self, name: str, version: str = "v1") -> GoldenDataset:
        path = self.root / f"{name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"golden dataset '{name}' not found in {self.root}. Available: {self.available()}"
            )
        return GoldenDataset.from_jsonl(path, name=name, version=version)

    def save(self, dataset: GoldenDataset) -> DatasetVersion:
        path = self.root / f"{dataset.name}.jsonl"
        dataset.to_jsonl(path)
        descriptor = dataset.describe()
        descriptor.path = str(path)
        (self.root / f"{dataset.name}.meta.json").write_text(descriptor.to_json(), encoding="utf-8")
        return descriptor

    def diff(self, left: GoldenDataset, right: GoldenDataset) -> dict[str, list[str]]:
        """Which cases were added, removed, or edited between two versions.

        This is the report you attach when a metric moves and the dataset also
        moved -- without it, the two effects are unattributable.
        """
        left_map = {c.case_id: c.content_digest() for c in left}
        right_map = {c.case_id: c.content_digest() for c in right}
        return {
            "added": sorted(set(right_map) - set(left_map)),
            "removed": sorted(set(left_map) - set(right_map)),
            "modified": sorted(k for k in set(left_map) & set(right_map) if left_map[k] != right_map[k]),
        }
