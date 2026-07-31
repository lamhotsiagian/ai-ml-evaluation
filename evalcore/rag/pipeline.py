"""The RAG pipeline under evaluation, instrumented for stage-wise attribution.

The pipeline emits a trace with a record for every stage -- retrieve, rerank,
assemble, generate -- because "the RAG system is wrong" is not an actionable
statement. The only useful question is *which stage* lost the answer, and that
requires per-stage evidence captured at run time, not reconstructed afterwards.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from evalcore.config import Settings, get_settings
from evalcore.llm import build_chat_model
from evalcore.rag.index import Chunk, RagIndex, RetrievalHit

_ANSWER_SYSTEM = """You answer strictly from the numbered context passages provided.

Rules:
- Every factual sentence must be followed by a citation of the form [n] naming the
  passage it came from. A sentence with no citation is a bug.
- If the passages do not contain the answer, reply exactly:
  INSUFFICIENT_CONTEXT: <one sentence naming what is missing>
- Do not use knowledge that is not in the passages, even if you are confident it is true.
- Do not speculate, and do not soften a missing answer into a plausible one."""

_ANSWER_USER = """## Context passages
{context}

## Question
{question}

Answer using only the passages above, citing each claim as [n]."""


@dataclass
class StageTrace:
    """One stage of pipeline execution."""

    stage: str
    latency_ms: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class RagResponse:
    """A generated answer plus the full evidence trail behind it."""

    question: str
    answer: str
    hits: list[RetrievalHit]
    contexts: list[str]
    trace: list[StageTrace]
    abstained: bool
    total_latency_ms: float
    prompt_tokens_estimate: int

    @property
    def retrieved_chunk_ids(self) -> list[str]:
        return [hit.chunk.chunk_id for hit in self.hits]

    @property
    def cited_indices(self) -> set[int]:
        """1-based passage indices the answer actually cited."""
        import re
        return {int(m) for m in re.findall(r"\[(\d+)\]", self.answer)}

    def citation_coverage(self) -> float:
        """Fraction of retrieved passages the answer cited.

        Low coverage with a correct answer means the context window is being
        paid for and not used -- a direct, measurable cost saving. High coverage
        with a wrong answer means retrieval brought in convincing but irrelevant
        material.
        """
        return len(self.cited_indices) / len(self.contexts) if self.contexts else 0.0


class RagPipeline:
    """Retrieve, optionally rerank, assemble, generate -- each stage traced."""

    def __init__(
        self,
        index: RagIndex,
        *,
        k: int = 5,
        mode: str = "hybrid",
        rerank: bool = False,
        max_context_tokens: int = 2000,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.index = index
        self.k = k
        self.mode = mode
        self.rerank = rerank
        self.max_context_tokens = max_context_tokens
        self._llm = build_chat_model(role="generation", settings=self.settings)
        self._reranker = LLMReranker(settings=self.settings) if rerank else None

    async def aanswer(self, question: str) -> RagResponse:
        trace: list[StageTrace] = []
        started = time.perf_counter()

        t0 = time.perf_counter()
        candidate_k = self.k * 4 if self.rerank else self.k
        hits = self.index.search(question, candidate_k, mode=self.mode)
        trace.append(StageTrace("retrieve", (time.perf_counter() - t0) * 1000, {
            "mode": self.mode, "k": candidate_k,
            "chunk_ids": [h.chunk.chunk_id for h in hits],
            "top_score": hits[0].score if hits else 0.0,
        }))

        if self._reranker and hits:
            t0 = time.perf_counter()
            hits = await self._reranker.arerank(question, hits, top_k=self.k)
            trace.append(StageTrace("rerank", (time.perf_counter() - t0) * 1000, {
                "kept": [h.chunk.chunk_id for h in hits],
            }))
        else:
            hits = hits[: self.k]

        t0 = time.perf_counter()
        contexts, dropped = self._assemble(hits)
        trace.append(StageTrace("assemble", (time.perf_counter() - t0) * 1000, {
            "n_contexts": len(contexts), "dropped_for_budget": dropped,
            "approx_tokens": sum(len(c.split()) for c in contexts),
        }))

        t0 = time.perf_counter()
        numbered = "\n\n".join(f"[{i + 1}] {text}" for i, text in enumerate(contexts))
        messages = [
            SystemMessage(content=_ANSWER_SYSTEM),
            HumanMessage(content=_ANSWER_USER.format(context=numbered, question=question)),
        ]
        response = await self._llm.ainvoke(messages)
        answer = str(response.content).strip()
        trace.append(StageTrace("generate", (time.perf_counter() - t0) * 1000, {
            "model": self.settings.generation_model, "chars": len(answer),
        }))

        return RagResponse(
            question=question,
            answer=answer,
            hits=hits,
            contexts=contexts,
            trace=trace,
            abstained=answer.upper().startswith("INSUFFICIENT_CONTEXT"),
            total_latency_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens_estimate=len(numbered.split()) + len(question.split()),
        )

    def answer(self, question: str) -> RagResponse:
        import asyncio
        return asyncio.run(self.aanswer(question))

    def _assemble(self, hits: Sequence[RetrievalHit]) -> tuple[list[str], int]:
        """Pack contexts up to the token budget, highest ranked first.

        Truncating the *last* passage rather than dropping it is tempting and
        wrong: a half-passage produces a half-claim, and the generator will
        confidently complete it. Passages that do not fit are dropped whole and
        the drop is recorded in the trace so context-budget pressure is visible
        rather than mysterious.
        """
        contexts: list[str] = []
        used, dropped = 0, 0
        for hit in hits:
            tokens = hit.chunk.n_tokens
            if used + tokens > self.max_context_tokens and contexts:
                dropped += 1
                continue
            contexts.append(hit.chunk.text)
            used += tokens
        return contexts, dropped


class LLMReranker:
    """Cross-encoder-style reranking using the chat model as a relevance scorer.

    A dedicated cross-encoder is better and faster in production. This exists so
    the reranking *stage* is present and measurable with only a Gemini key: the
    lab's point is to show that reranking moves MRR and precision@k far more
    than recall@k, and that holds for any reranker.
    """

    _SYSTEM = (
        "Rate how well a passage helps answer a question. Reply with one integer 0-3:\n"
        "0 = irrelevant, 1 = topically related but does not answer, "
        "2 = partially answers, 3 = directly and completely answers.\n"
        "Reply with the digit only."
    )

    def __init__(self, *, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._llm = build_chat_model(role="judge", temperature=0.0, max_output_tokens=4,
                                     settings=self.settings)

    async def arerank(self, question: str, hits: Sequence[RetrievalHit], *, top_k: int = 5) -> list[RetrievalHit]:
        import asyncio

        async def _score(hit: RetrievalHit) -> float:
            messages = [
                SystemMessage(content=self._SYSTEM),
                HumanMessage(content=f"Question: {question}\n\nPassage: {hit.chunk.text}"),
            ]
            try:
                response = await self._llm.ainvoke(messages)
                digits = "".join(ch for ch in str(response.content) if ch.isdigit())
                return float(digits[0]) if digits else 0.0
            except Exception:  # noqa: BLE001 - a failed rerank falls back to retrieval order
                return -1.0

        scores = await asyncio.gather(*(_score(hit) for hit in hits))
        # -1.0 marks a failed score; those keep their original relative order
        # behind everything successfully scored rather than being deleted.
        ordered = sorted(
            zip(hits, scores),
            key=lambda pair: (pair[1] if pair[1] >= 0 else -1, -pair[0].rank),
            reverse=True,
        )
        return [
            RetrievalHit(hit.chunk, float(score), rank, "rerank")
            for rank, (hit, score) in enumerate(ordered[:top_k], start=1)
        ]


def format_contexts_for_judge(contexts: Sequence[str], limit: int = 8) -> list[str]:
    """Trim the context list handed to a judge.

    A judge given 30 passages will not read them all, and its verdict silently
    becomes a function of what fitted in its attention. Cap the list and say so
    in the report.
    """
    return list(contexts[:limit])


def chunks_to_contexts(chunks: Sequence[Chunk]) -> list[str]:
    return [chunk.text for chunk in chunks]
