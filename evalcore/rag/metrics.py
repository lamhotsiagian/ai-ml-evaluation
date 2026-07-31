"""RAG metrics: context precision/recall, faithfulness, claim verification.

The organising idea of this module is **stage-wise attribution**. A single
end-to-end score tells you a RAG system is at 0.68 and nothing else. Decomposing
into retrieval recall, context precision, faithfulness and answer relevance
tells you which of four teams owns the fix.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from evalcore.config import Settings, get_settings
from evalcore.llm import CacheKey, ResponseCache, build_chat_model, default_cache, render_messages
from evalcore.metrics.ranking import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank


# ---------------------------------------------------------------------------
# Deterministic retrieval-stage metrics
# ---------------------------------------------------------------------------
@dataclass
class RetrievalStageReport:
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    context_precision: float
    k: int

    def as_row(self) -> dict[str, float]:
        return {
            "retrieval_recall@k": round(self.recall_at_k, 4),
            "retrieval_precision@k": round(self.precision_at_k, 4),
            "mrr": round(self.mrr, 4),
            "ndcg@k": round(self.ndcg_at_k, 4),
            "context_precision": round(self.context_precision, 4),
            "k": self.k,
        }


def context_precision(retrieved_ids: Sequence[str], relevant_ids: set[str]) -> float:
    """Rank-weighted precision: relevant passages must appear *early*.

    Plain precision@k treats a gold passage at rank 1 and at rank 5 identically.
    Generators do not: attention degrades with position, and a gold passage
    buried under four irrelevant ones is materially less likely to be used. This
    is the RAGAS formulation -- mean of precision@i over the ranks that hold a
    relevant passage.
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0
    hits, running = 0, 0.0
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            hits += 1
            running += hits / rank
    return running / min(len(relevant_ids), len(retrieved_ids)) if hits else 0.0


def evaluate_retrieval_stage(
    retrieved_ids: Sequence[str], relevant_ids: set[str], *, k: int = 5,
    graded: dict[str, float] | None = None,
) -> RetrievalStageReport:
    """Every retrieval-stage metric for one query."""
    top_k = list(retrieved_ids[:k])
    relevance = graded or {chunk_id: 1.0 for chunk_id in relevant_ids}
    return RetrievalStageReport(
        recall_at_k=recall_at_k(top_k, relevant_ids, k),
        precision_at_k=precision_at_k(top_k, relevant_ids, k),
        mrr=reciprocal_rank(top_k, relevant_ids),
        ndcg_at_k=ndcg_at_k(top_k, relevance, k),
        context_precision=context_precision(top_k, relevant_ids),
        k=k,
    )


def context_recall_from_claims(supported_claims: int, total_reference_claims: int) -> float:
    """Fraction of the reference answer's claims that the context can support.

    This is the metric that isolates retrieval failure from generation failure.
    If context recall is 0.4, the generator was never given enough to answer,
    and no prompt engineering will fix it.
    """
    return supported_claims / total_reference_claims if total_reference_claims else float("nan")


# ---------------------------------------------------------------------------
# Claim extraction and verification (LLM-assisted)
# ---------------------------------------------------------------------------
class ClaimList(BaseModel):
    claims: list[str] = Field(description="Atomic, independently checkable factual claims")


class ClaimVerdict(BaseModel):
    claim: str
    supported: bool
    supporting_passage: int | None = Field(
        default=None, description="1-based index of the passage that supports the claim"
    )
    reason: str = Field(max_length=300)


class ClaimVerdicts(BaseModel):
    verdicts: list[ClaimVerdict]


_CLAIM_SYSTEM = """Split the answer into atomic factual claims.

Rules:
- One verifiable assertion per claim. Split compound sentences.
- Resolve pronouns and references so each claim stands alone.
- Drop hedges, opinions, and pure discourse ("Let me explain", "In summary").
- Preserve numbers, names, and units exactly as written.
- Do not add claims that are not in the answer."""

_VERIFY_SYSTEM = """Decide whether each claim is supported by the numbered passages.

A claim is supported ONLY if a passage states it or directly entails it.
- Plausible, widely known, or probably-true is NOT supported.
- A claim contradicted by a passage is not supported.
- Cite the 1-based index of the single passage that best supports the claim."""


@dataclass
class FaithfulnessReport:
    faithfulness: float
    n_claims: int
    unsupported_claims: list[str] = field(default_factory=list)
    verdicts: list[ClaimVerdict] = field(default_factory=list)

    def as_row(self) -> dict[str, Any]:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "n_claims": self.n_claims,
            "n_unsupported": len(self.unsupported_claims),
            "first_unsupported": self.unsupported_claims[0][:120] if self.unsupported_claims else "",
        }


class ClaimVerifier:
    """Two-step faithfulness measurement: decompose, then verify.

    Asking one model call "is this answer faithful?" produces a number that
    correlates weakly with reality, because the model averages over an answer
    that may be 90% grounded and 10% fabricated. Decomposition makes the unit of
    judgement an atomic claim, which is a question a model can actually answer,
    and it yields a *list of the specific fabrications* -- which is the part an
    engineer can act on.
    """

    def __init__(self, *, settings: Settings | None = None, cache: ResponseCache | None = None) -> None:
        self.settings = settings or get_settings()
        self._extractor = build_chat_model(
            role="judge", temperature=0.0, max_output_tokens=900, settings=self.settings
        ).with_structured_output(ClaimList)
        self._verifier = build_chat_model(
            role="judge", temperature=0.0, max_output_tokens=1600, settings=self.settings
        ).with_structured_output(ClaimVerdicts)
        self._cache = cache if cache is not None else default_cache(self.settings)

    async def aextract_claims(self, answer: str) -> list[str]:
        messages = [SystemMessage(content=_CLAIM_SYSTEM), HumanMessage(content=answer)]
        key = CacheKey(self.settings.judge_model, 0.0, render_messages(messages), "claims-v1")
        cached = self._cache.get(key)
        if cached:
            return ClaimList.model_validate_json(cached).claims
        result: ClaimList = await self._extractor.ainvoke(messages)
        self._cache.put(key, result.model_dump_json())
        return result.claims

    async def averify(self, claims: Sequence[str], contexts: Sequence[str]) -> list[ClaimVerdict]:
        if not claims:
            return []
        numbered = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
        claim_block = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
        messages = [
            SystemMessage(content=_VERIFY_SYSTEM),
            HumanMessage(content=f"## Passages\n{numbered}\n\n## Claims\n{claim_block}"),
        ]
        key = CacheKey(self.settings.judge_model, 0.0, render_messages(messages), "verify-v1")
        cached = self._cache.get(key)
        if cached:
            return ClaimVerdicts.model_validate_json(cached).verdicts
        result: ClaimVerdicts = await self._verifier.ainvoke(messages)
        self._cache.put(key, result.model_dump_json())
        return result.verdicts

    async def afaithfulness(self, answer: str, contexts: Sequence[str]) -> FaithfulnessReport:
        """Fraction of the answer's atomic claims supported by the context."""
        if answer.upper().startswith("INSUFFICIENT_CONTEXT"):
            # A correct abstention has nothing to fabricate; scoring it 0.0
            # would punish exactly the behaviour the system is meant to learn.
            return FaithfulnessReport(1.0, 0, [], [])
        claims = await self.aextract_claims(answer)
        if not claims:
            return FaithfulnessReport(float("nan"), 0, [], [])
        verdicts = await self.averify(claims, contexts)
        supported = sum(1 for v in verdicts if v.supported)
        unsupported = [v.claim for v in verdicts if not v.supported]
        return FaithfulnessReport(supported / len(verdicts) if verdicts else float("nan"),
                                  len(claims), unsupported, verdicts)

    def faithfulness(self, answer: str, contexts: Sequence[str]) -> FaithfulnessReport:
        return asyncio.run(self.afaithfulness(answer, contexts))

    async def acontext_recall(self, reference_answer: str, contexts: Sequence[str]) -> float:
        """Claims of the *gold* answer that the retrieved context supports."""
        claims = await self.aextract_claims(reference_answer)
        if not claims:
            return float("nan")
        verdicts = await self.averify(claims, contexts)
        return sum(1 for v in verdicts if v.supported) / len(verdicts) if verdicts else float("nan")


# ---------------------------------------------------------------------------
# Citation checking (deterministic)
# ---------------------------------------------------------------------------
_CITATION_RE = re.compile(r"\[(\d+)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class CitationReport:
    citation_rate: float
    valid_citation_rate: float
    n_sentences: int
    n_citations: int
    invalid_indices: list[int] = field(default_factory=list)

    def as_row(self) -> dict[str, float]:
        return {
            "citation_rate": round(self.citation_rate, 4),
            "valid_citation_rate": round(self.valid_citation_rate, 4),
            "n_sentences": self.n_sentences,
            "n_citations": self.n_citations,
            "n_invalid": len(self.invalid_indices),
        }


def evaluate_citations(answer: str, n_contexts: int) -> CitationReport:
    """Check citation *presence* and *validity* without any model call.

    Two distinct failures are separated here. A sentence with no citation is an
    uncited claim. A citation pointing at passage [9] when only 5 were supplied
    is a *fabricated citation*, which is worse: it looks grounded to a user and
    to any pipeline that checks for the presence of brackets.
    """
    # Sentences shorter than 15 characters are headers, list bullets and
    # fragments; requiring a citation on them produces noise, not signal.
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(answer) if len(s.strip()) > 15]
    if not sentences:
        return CitationReport(float("nan"), float("nan"), 0, 0, [])

    cited_sentences = 0
    all_indices: list[int] = []
    for sentence in sentences:
        indices = [int(m) for m in _CITATION_RE.findall(sentence)]
        if indices:
            cited_sentences += 1
        all_indices.extend(indices)

    invalid = sorted({i for i in all_indices if i < 1 or i > n_contexts})
    valid_rate = (
        (len(all_indices) - len([i for i in all_indices if i in invalid])) / len(all_indices)
        if all_indices else float("nan")
    )
    return CitationReport(
        citation_rate=cited_sentences / len(sentences),
        valid_citation_rate=valid_rate,
        n_sentences=len(sentences),
        n_citations=len(all_indices),
        invalid_indices=invalid,
    )


# ---------------------------------------------------------------------------
# Abstention on unanswerable questions
# ---------------------------------------------------------------------------
def evaluate_abstention(
    abstained: Sequence[bool], is_answerable: Sequence[bool]
) -> dict[str, float]:
    """Abstention quality on a suite containing unanswerable questions.

    Every serious RAG evaluation set must contain questions the corpus cannot
    answer -- typically 15-25% of it. Without them, "faithfulness 0.94" is
    measured only on questions where the right answer was available, and the
    system's actual behaviour on the long tail is completely unmeasured.
    """
    abstain = list(abstained)
    answerable = list(is_answerable)
    if len(abstain) != len(answerable):
        raise ValueError("abstention vectors must be aligned")

    unanswerable_total = sum(1 for a in answerable if not a)
    answerable_total = sum(1 for a in answerable if a)
    correct_abstain = sum(1 for ab, an in zip(abstain, answerable) if ab and not an)
    over_abstain = sum(1 for ab, an in zip(abstain, answerable) if ab and an)

    return {
        "abstention_recall": correct_abstain / unanswerable_total if unanswerable_total else float("nan"),
        "over_abstention_rate": over_abstain / answerable_total if answerable_total else float("nan"),
        "n_unanswerable": unanswerable_total,
        "n_answerable": answerable_total,
    }
