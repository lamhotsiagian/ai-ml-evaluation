"""Deterministic validators -- the checks that must never be given to a judge.

A rule that can be written as code should be written as code. Deterministic
checks are free, instant, perfectly reproducible and never hallucinate; every
constraint moved out of the judge prompt and into this module makes the judge
cheaper and more reliable at the same time.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

try:  # jsonschema is optional; the validator degrades to structural checks.
    import jsonschema  # type: ignore
    _HAS_JSONSCHEMA = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_JSONSCHEMA = False


@dataclass
class ValidationResult:
    name: str
    passed: bool
    detail: str = ""
    score: float = field(init=False)

    def __post_init__(self) -> None:
        self.score = 1.0 if self.passed else 0.0


_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Pull the first JSON value out of a model response.

    Models wrap JSON in prose and fences no matter how firmly the prompt forbids
    it. Rather than failing the response for a formatting habit, three
    strategies are tried in order: whole-string parse, fenced block, first
    balanced brace/bracket span. A response that survives none of these is
    genuinely malformed.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence = _FENCE_RE.search(text)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth, in_string, escaped = 0, False, False
        for index in range(start, len(text)):
            char = text[index]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError("no parseable JSON value found in response")


def validate_json(text: str, schema: dict[str, Any] | None = None) -> ValidationResult:
    """Parse-and-schema check for structured outputs."""
    try:
        payload = extract_json(text)
    except ValueError as exc:
        return ValidationResult("json_parse", False, str(exc))

    if schema is None:
        return ValidationResult("json_parse", True, f"parsed {type(payload).__name__}")

    if not _HAS_JSONSCHEMA:
        missing = [key for key in schema.get("required", []) if key not in (payload or {})]
        return ValidationResult(
            "json_schema", not missing,
            "jsonschema not installed; required-key check only"
            + (f"; missing {missing}" if missing else ""),
        )

    try:
        jsonschema.validate(payload, schema)
        return ValidationResult("json_schema", True, "valid")
    except jsonschema.ValidationError as exc:  # type: ignore[attr-defined]
        path = "/".join(str(p) for p in exc.absolute_path) or "(root)"
        return ValidationResult("json_schema", False, f"{path}: {exc.message}")


def validate_python_syntax(code: str) -> ValidationResult:
    """Compile-only check for generated Python.

    Deliberately does not execute anything. Syntax validity is the cheapest
    possible filter and removes a large share of failures before any sandboxed
    execution -- which is the expensive, risky step -- is attempted.
    """
    cleaned = _strip_fence(code)
    try:
        ast.parse(cleaned)
        return ValidationResult("python_syntax", True, "parses")
    except SyntaxError as exc:
        return ValidationResult("python_syntax", False, f"line {exc.lineno}: {exc.msg}")


def _strip_fence(text: str) -> str:
    match = re.search(r"```(?:python|py)?\s*(.*?)```", text, re.DOTALL)
    return match.group(1) if match else text


def validate_regex(text: str, pattern: str, *, name: str = "regex", must_match: bool = True) -> ValidationResult:
    found = re.search(pattern, text, re.MULTILINE | re.DOTALL) is not None
    passed = found if must_match else not found
    return ValidationResult(name, passed, f"pattern={'found' if found else 'absent'}")


def validate_length(text: str, *, max_words: int | None = None,
                    min_words: int | None = None) -> ValidationResult:
    words = len(text.split())
    problems = []
    if max_words is not None and words > max_words:
        problems.append(f"{words} words > max {max_words}")
    if min_words is not None and words < min_words:
        problems.append(f"{words} words < min {min_words}")
    return ValidationResult("length", not problems, "; ".join(problems) or f"{words} words")


def validate_no_forbidden_terms(text: str, forbidden: Sequence[str]) -> ValidationResult:
    lowered = text.lower()
    hits = sorted({term for term in forbidden if term.lower() in lowered})
    return ValidationResult("forbidden_terms", not hits, f"found {hits}" if hits else "clean")


def exact_match(prediction: str, reference: str, *, normalise: bool = True) -> ValidationResult:
    """Normalised exact match -- the correct metric for extraction tasks.

    Normalisation (case, whitespace, surrounding punctuation, leading articles)
    is what separates a usable exact-match score from one that fails on
    ``"Paris."`` versus ``"paris"``. Anything beyond that should be a judge.
    """
    def _norm(value: str) -> str:
        if not normalise:
            return value
        value = value.strip().lower()
        value = re.sub(r"^(the|a|an)\s+", "", value)
        value = re.sub(r"[\s]+", " ", value)
        return value.strip(" .,:;!?\"'")

    matched = _norm(prediction) == _norm(reference)
    return ValidationResult("exact_match", matched, f"'{prediction[:60]}' vs '{reference[:60]}'")


class ValidatorSuite:
    """An ordered set of deterministic checks applied to every response.

    Run this *before* the judge. Roughly half of real evaluation failures --
    unparseable JSON, missing required fields, blown length limits, forbidden
    terms -- are caught here at zero marginal cost, and the judge is then asked
    only about the things that genuinely need judgement.
    """

    def __init__(self, checks: dict[str, Callable[[str], ValidationResult]] | None = None) -> None:
        self._checks = dict(checks or {})

    def add(self, name: str, check: Callable[[str], ValidationResult]) -> "ValidatorSuite":
        self._checks[name] = check
        return self

    def run(self, text: str) -> dict[str, ValidationResult]:
        return {name: check(text) for name, check in self._checks.items()}

    def scores(self, text: str) -> dict[str, float]:
        return {name: result.score for name, result in self.run(text).items()}

    def pass_rate(self, text: str) -> float:
        results = self.run(text)
        return sum(r.passed for r in results.values()) / len(results) if results else 1.0

    @property
    def names(self) -> list[str]:
        return list(self._checks)
