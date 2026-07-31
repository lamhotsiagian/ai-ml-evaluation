"""Ranking and retrieval metrics.

These are implemented from first principles rather than pulled from a library,
because the details that matter -- how ties are handled, what the ideal DCG is
when relevance is graded, whether recall is capped at K -- differ between
libraries and quietly change reported numbers by several points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

import numpy as np


@dataclass
class RankingReport:
    k: int
    recall_at_k: float
    precision_at_k: float
    hit_rate: float
    mrr: float
    map_score: float
    ndcg: float
    n_queries: int

    def as_row(self) -> dict[str, float]:
        return {
            "k": self.k,
            "recall@k": round(self.recall_at_k, 4),
            "precision@k": round(self.precision_at_k, 4),
            "hit_rate": round(self.hit_rate, 4),
            "mrr": round(self.mrr, 4),
            "map": round(self.map_score, 4),
            "ndcg@k": round(self.ndcg, 4),
            "queries": self.n_queries,
        }


def recall_at_k(retrieved: Sequence[Hashable], relevant: set[Hashable], k: int) -> float:
    """Fraction of the relevant set found in the top K.

    The single most important retrieval number in a RAG system: whatever
    retrieval misses, no generator can recover. Note the denominator is the size
    of the full relevant set, so a query with 10 gold documents evaluated at
    K=5 is capped at 0.5 by construction -- report ``recall@k`` alongside K, and
    never compare across different K values.
    """
    if not relevant:
        return float("nan")
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: Sequence[Hashable], relevant: set[Hashable], k: int) -> float:
    """Fraction of the top K that is relevant -- the context-pollution metric."""
    if k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def hit_rate_at_k(retrieved: Sequence[Hashable], relevant: set[Hashable], k: int) -> float:
    """1.0 if any relevant document appears in the top K.

    The right headline when the generator only needs one good passage; recall@k
    understates such a system, because finding 1 of 4 gold passages may be
    entirely sufficient to answer.
    """
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def reciprocal_rank(retrieved: Sequence[Hashable], relevant: set[Hashable]) -> float:
    """1 / rank of the first relevant result, 0 if none.

    Sensitive to position in a way recall is not. When a reranker is added,
    recall@10 often does not move at all while MRR jumps -- that gap is exactly
    the reranker's contribution and the reason to keep both metrics.
    """
    for position, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


def average_precision(retrieved: Sequence[Hashable], relevant: set[Hashable]) -> float:
    """Mean of precision@i over the positions where a relevant doc appears."""
    if not relevant:
        return float("nan")
    hits, running = 0, 0.0
    for position, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            hits += 1
            running += hits / position
    return running / len(relevant)


def dcg(gains: Sequence[float], k: int) -> float:
    """Discounted cumulative gain with the standard 2^rel - 1 formulation."""
    gains = np.asarray(gains[:k], dtype=float)
    if gains.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    return float(np.sum((np.power(2.0, gains) - 1.0) * discounts))


def ndcg_at_k(
    retrieved: Sequence[Hashable], relevance: Mapping[Hashable, float], k: int
) -> float:
    """Normalised DCG for graded relevance.

    ``relevance`` maps document id to a graded score (0 = irrelevant,
    3 = perfect). The ideal DCG is computed over the *full* relevance map, not
    just retrieved documents -- computing it over retrieved documents only is a
    common bug that inflates NDCG toward 1.0 for a system that retrieved nothing
    good.
    """
    gains = [float(relevance.get(doc_id, 0.0)) for doc_id in retrieved[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal, k)
    return 0.0 if ideal_dcg == 0 else dcg(gains, k) / ideal_dcg


def evaluate_ranking(
    results: Sequence[Sequence[Hashable]],
    gold: Sequence[set[Hashable]],
    *,
    k: int = 5,
    graded_relevance: Sequence[Mapping[Hashable, float]] | None = None,
) -> RankingReport:
    """Aggregate every ranking metric over a query set.

    Args:
        results: Per-query ranked document id lists.
        gold: Per-query relevant document id sets.
        k: Cutoff.
        graded_relevance: Optional per-query graded relevance maps. When absent,
            binary relevance from ``gold`` is used for NDCG.
    """
    if len(results) != len(gold):
        raise ValueError("results and gold must be aligned per query")
    if not results:
        return RankingReport(k, 0, 0, 0, 0, 0, 0, 0)

    recalls, precisions, hits, rrs, aps, ndcgs = [], [], [], [], [], []
    for i, (retrieved, relevant) in enumerate(zip(results, gold)):
        recalls.append(recall_at_k(retrieved, relevant, k))
        precisions.append(precision_at_k(retrieved, relevant, k))
        hits.append(hit_rate_at_k(retrieved, relevant, k))
        rrs.append(reciprocal_rank(retrieved, relevant))
        aps.append(average_precision(retrieved, relevant))
        relevance = (graded_relevance[i] if graded_relevance
                     else {doc_id: 1.0 for doc_id in relevant})
        ndcgs.append(ndcg_at_k(retrieved, relevance, k))

    def _nanmean(values: list[float]) -> float:
        array = np.asarray(values, dtype=float)
        return float(np.nanmean(array)) if array.size else 0.0

    return RankingReport(
        k=k,
        recall_at_k=_nanmean(recalls),
        precision_at_k=_nanmean(precisions),
        hit_rate=_nanmean(hits),
        mrr=_nanmean(rrs),
        map_score=_nanmean(aps),
        ndcg=_nanmean(ndcgs),
        n_queries=len(results),
    )


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hashable]], *, k_constant: int = 60
) -> list[Hashable]:
    """Fuse several ranked lists without needing comparable scores.

    RRF is the workhorse behind hybrid search: BM25 scores and cosine
    similarities are not on the same scale, so fusing raw scores requires
    fragile normalisation. RRF fuses ranks instead and needs no tuning beyond
    ``k_constant``, which damps the influence of top positions.
    """
    fused: dict[Hashable, float] = {}
    for ranking in rankings:
        for position, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k_constant + position)
    return [doc_id for doc_id, _ in sorted(fused.items(), key=lambda item: -item[1])]
