"""Chroma-backed retrieval: chunking, indexing, dense/BM25/hybrid search.

Retrieval is evaluated stage by stage, so the index must expose each stage
separately rather than hiding everything behind one ``retrieve()`` call. This
module therefore returns chunk ids and scores from each retriever independently,
which is what makes the stage-wise attribution in Chapter 4 possible.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from evalcore.config import Settings, get_settings
from evalcore.llm import build_embeddings
from evalcore.metrics.ranking import reciprocal_rank_fusion


@dataclass
class Chunk:
    """A retrievable unit of text with stable provenance."""

    chunk_id: str
    text: str
    doc_id: str
    position: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_tokens(self) -> int:
        """Whitespace token count -- a deliberate approximation.

        Chunk sizing decisions are made in relative terms ("is 400 better than
        800?"), and a tokenizer-exact count would add a dependency and a model
        assumption without changing any of those decisions.
        """
        return len(self.text.split())


@dataclass
class RetrievalHit:
    chunk: Chunk
    score: float
    rank: int
    retriever: str


def sentence_chunk(
    text: str,
    doc_id: str,
    *,
    target_tokens: int = 220,
    overlap_tokens: int = 40,
) -> list[Chunk]:
    """Sentence-aware chunking with overlap.

    Splitting on a fixed character count severs sentences and, more damagingly,
    separates a claim from the qualifier that scopes it -- which is a direct
    cause of "faithful-looking but wrong" generations. Packing whole sentences up
    to a token budget keeps every chunk independently readable, and the overlap
    keeps a claim that straddles a boundary retrievable from either side.
    """
    sentences = _split_sentences(text)
    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_tokens = 0
    position = 0

    def _flush() -> None:
        nonlocal buffer, buffer_tokens, position
        if not buffer:
            return
        body = " ".join(buffer).strip()
        chunks.append(Chunk(
            chunk_id=_chunk_id(doc_id, position, body),
            text=body,
            doc_id=doc_id,
            position=position,
            metadata={"n_tokens": len(body.split())},
        ))
        position += 1
        if overlap_tokens > 0:
            tail, tail_tokens = [], 0
            for sentence in reversed(buffer):
                count = len(sentence.split())
                if tail_tokens + count > overlap_tokens:
                    break
                tail.insert(0, sentence)
                tail_tokens += count
            buffer, buffer_tokens = tail, tail_tokens
        else:
            buffer, buffer_tokens = [], 0

    for sentence in sentences:
        count = len(sentence.split())
        if buffer_tokens + count > target_tokens and buffer:
            _flush()
        buffer.append(sentence)
        buffer_tokens += count
    _flush()
    return chunks


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = " ".join(paragraph.split())
        if paragraph:
            parts.extend(s for s in _SENTENCE_RE.split(paragraph) if s)
    return parts


def _chunk_id(doc_id: str, position: int, body: str) -> str:
    digest = hashlib.sha1(body.encode("utf-8")).hexdigest()[:8]
    return f"{doc_id}::{position:04d}::{digest}"


# ---------------------------------------------------------------------------
# BM25 (lexical) retriever
# ---------------------------------------------------------------------------
class BM25Index:
    """Okapi BM25 over the same chunks the vector store holds.

    Included because dense retrieval reliably loses on exact identifiers --
    error codes, SKUs, function names, version numbers. Those are precisely the
    queries a support or developer-facing RAG system receives most often, so a
    dense-only system posts a good average recall and fails the queries that
    matter.
    """

    _TOKEN_RE = re.compile(r"[a-z0-9_\-\.]+")

    def __init__(self, chunks: Sequence[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.chunks = list(chunks)
        self._tokenised = [self._tokenise(c.text) for c in self.chunks]
        self._lengths = [len(t) for t in self._tokenised]
        self._avg_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        self._postings: dict[str, list[int]] = defaultdict(list)
        self._term_frequencies: list[Counter] = []
        for index, tokens in enumerate(self._tokenised):
            counts = Counter(tokens)
            self._term_frequencies.append(counts)
            for term in counts:
                self._postings[term].append(index)
        self._n_docs = len(self.chunks)

    @classmethod
    def _tokenise(cls, text: str) -> list[str]:
        return cls._TOKEN_RE.findall(text.lower())

    def _idf(self, term: str) -> float:
        df = len(self._postings.get(term, ()))
        if df == 0:
            return 0.0
        return math.log(1 + (self._n_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 5) -> list[RetrievalHit]:
        scores: dict[int, float] = defaultdict(float)
        for term in self._tokenise(query):
            idf = self._idf(term)
            if idf == 0:
                continue
            for doc in self._postings[term]:
                tf = self._term_frequencies[doc][term]
                norm = 1 - self.b + self.b * (self._lengths[doc] / (self._avg_length or 1))
                scores[doc] += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)
        ranked = sorted(scores.items(), key=lambda item: -item[1])[:k]
        return [
            RetrievalHit(self.chunks[doc], float(score), rank, "bm25")
            for rank, (doc, score) in enumerate(ranked, start=1)
        ]


# ---------------------------------------------------------------------------
# Chroma-backed dense retriever + hybrid search
# ---------------------------------------------------------------------------
class RagIndex:
    """Dense (Chroma + Gemini embeddings), lexical (BM25), and hybrid retrieval."""

    def __init__(
        self,
        collection_name: str = "eval_corpus",
        *,
        settings: Settings | None = None,
        persist: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.collection_name = collection_name
        self._persist = persist
        self._chunks: dict[str, Chunk] = {}
        self._bm25: BM25Index | None = None
        self._store = None  # lazily built so the module imports without an API key

    # -- construction ------------------------------------------------------
    def build(self, chunks: Sequence[Chunk], *, reset: bool = True) -> "RagIndex":
        """Index chunks into Chroma and build the BM25 side index."""
        from langchain_chroma import Chroma  # imported lazily: heavy dependency
        from langchain_core.documents import Document

        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self._bm25 = BM25Index(list(self._chunks.values()))

        embeddings = build_embeddings(self.settings)
        directory = str(self.settings.chroma_dir) if self._persist else None
        store = Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            persist_directory=directory,
        )
        if reset:
            try:
                store.delete_collection()
            except Exception:  # noqa: BLE001 - a missing collection is not an error
                pass
            store = Chroma(
                collection_name=self.collection_name,
                embedding_function=embeddings,
                persist_directory=directory,
            )

        documents = [
            Document(
                page_content=chunk.text,
                metadata={"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id,
                          "position": chunk.position},
            )
            for chunk in chunks
        ]
        # Batched to stay inside embedding request limits on the free tier.
        for start in range(0, len(documents), 64):
            batch = documents[start : start + 64]
            store.add_documents(batch, ids=[d.metadata["chunk_id"] for d in batch])
        self._store = store
        return self

    def load_corpus(self, directory: Path | str | None = None, **chunk_kwargs: Any) -> list[Chunk]:
        """Chunk every ``.md``/``.txt`` file in a corpus directory."""
        root = Path(directory) if directory else (self.settings.data_dir / "corpus")
        chunks: list[Chunk] = []
        for path in sorted(root.glob("**/*")):
            if path.suffix.lower() not in {".md", ".txt"}:
                continue
            chunks.extend(sentence_chunk(path.read_text(encoding="utf-8"), path.stem, **chunk_kwargs))
        return chunks

    # -- retrieval ---------------------------------------------------------
    def dense_search(self, query: str, k: int = 5) -> list[RetrievalHit]:
        if self._store is None:
            raise RuntimeError("RagIndex.build must be called before searching")
        scored = self._store.similarity_search_with_relevance_scores(query, k=k)
        hits: list[RetrievalHit] = []
        for rank, (document, score) in enumerate(scored, start=1):
            chunk_id = document.metadata.get("chunk_id", "")
            chunk = self._chunks.get(chunk_id) or Chunk(
                chunk_id, document.page_content, document.metadata.get("doc_id", ""),
                int(document.metadata.get("position", 0)),
            )
            hits.append(RetrievalHit(chunk, float(score), rank, "dense"))
        return hits

    def lexical_search(self, query: str, k: int = 5) -> list[RetrievalHit]:
        if self._bm25 is None:
            raise RuntimeError("RagIndex.build must be called before searching")
        return self._bm25.search(query, k)

    def hybrid_search(self, query: str, k: int = 5, *, candidate_k: int | None = None) -> list[RetrievalHit]:
        """Dense + BM25 fused with reciprocal rank fusion.

        Both retrievers are asked for more candidates than the final K
        (``candidate_k``, default 3K) because fusion can only reorder what it is
        given -- fusing two top-5 lists caps the achievable recall at the union
        of two top-5 lists, which defeats the point.
        """
        candidate_k = candidate_k or max(k * 3, 10)
        dense = self.dense_search(query, candidate_k)
        lexical = self.lexical_search(query, candidate_k)
        by_id = {hit.chunk.chunk_id: hit.chunk for hit in (*dense, *lexical)}
        fused_ids = reciprocal_rank_fusion([
            [hit.chunk.chunk_id for hit in dense],
            [hit.chunk.chunk_id for hit in lexical],
        ])
        return [
            RetrievalHit(by_id[chunk_id], 1.0 / (60 + rank), rank, "hybrid")
            for rank, chunk_id in enumerate(fused_ids[:k], start=1)
        ]

    def search(self, query: str, k: int = 5, *, mode: str = "hybrid") -> list[RetrievalHit]:
        dispatch = {"dense": self.dense_search, "bm25": self.lexical_search,
                    "hybrid": self.hybrid_search}
        if mode not in dispatch:
            raise ValueError(f"unknown retrieval mode '{mode}'; expected one of {sorted(dispatch)}")
        return dispatch[mode](query, k)

    # -- inspection --------------------------------------------------------
    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    def stats(self) -> dict[str, Any]:
        token_counts = [chunk.n_tokens for chunk in self._chunks.values()]
        return {
            "collection": self.collection_name,
            "n_chunks": len(self._chunks),
            "n_docs": len({chunk.doc_id for chunk in self._chunks.values()}),
            "mean_tokens": round(sum(token_counts) / len(token_counts), 1) if token_counts else 0,
            "max_tokens": max(token_counts, default=0),
            "min_tokens": min(token_counts, default=0),
        }


def chunk_size_grid(
    text_by_doc: Iterable[tuple[str, str]],
    sizes: Sequence[int] = (128, 256, 512),
    overlaps: Sequence[int] = (0, 32, 64),
) -> list[dict[str, Any]]:
    """Enumerate chunking configurations for the chunk-strategy sweep in the UI.

    Chunk size is the highest-leverage retrieval hyper-parameter and the one
    most often left at a library default. Sweeping it against retrieval recall
    typically moves recall@5 by 10-20 points, which is more than any reranker
    will give you.
    """
    rows: list[dict[str, Any]] = []
    for size in sizes:
        for overlap in overlaps:
            if overlap >= size:
                continue
            chunks: list[Chunk] = []
            for doc_id, text in text_by_doc:
                chunks.extend(sentence_chunk(text, doc_id, target_tokens=size, overlap_tokens=overlap))
            counts = [c.n_tokens for c in chunks]
            rows.append({
                "target_tokens": size,
                "overlap_tokens": overlap,
                "n_chunks": len(chunks),
                "mean_tokens": round(sum(counts) / len(counts), 1) if counts else 0,
                "p95_tokens": sorted(counts)[int(len(counts) * 0.95)] if counts else 0,
            })
    return rows
