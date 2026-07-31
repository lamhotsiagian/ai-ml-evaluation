"""Chapter 4 lab — stage-wise RAG evaluation over the Orbital corpus."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalcore.datasets import DatasetRegistry  # noqa: E402
from evalcore.rag import (  # noqa: E402
    ClaimVerifier,
    RagIndex,
    RagPipeline,
    evaluate_abstention,
    evaluate_citations,
    evaluate_retrieval_stage,
    sentence_chunk,
)
from evalcore.rag.index import chunk_size_grid  # noqa: E402
from evalcore.stats import bootstrap_interval  # noqa: E402
from ui.components import (  # noqa: E402
    dataframe,
    explain,
    interval_metric,
    page_header,
    require_api_key,
    settings_sidebar,
)

page_header("RAG Evaluation", 4, "retrieval, faithfulness, citations, abstention")
settings_sidebar()

registry = DatasetRegistry()
dataset = registry.load("rag_qa")

with st.sidebar:
    st.subheader("Pipeline configuration")
    mode = st.selectbox("Retrieval mode", ["hybrid", "dense", "bm25"])
    top_k = st.slider("Top K", 1, 10, 5)
    chunk_tokens = st.slider("Chunk target tokens", 80, 500, 220, 20)
    chunk_overlap = st.slider("Chunk overlap tokens", 0, 120, 40, 10)
    use_reranker = st.checkbox("Enable LLM reranker", value=False)

tabs = st.tabs(["Chunking", "Retrieval stage", "Generation stage", "Full suite"])


@st.cache_resource(show_spinner="Building the Chroma index…")
def build_index(target: int, overlap: int, collection: str) -> RagIndex:
    index = RagIndex(collection_name=collection)
    corpus = index.load_corpus(target_tokens=target, overlap_tokens=overlap)
    return index.build(corpus)


def corpus_texts() -> list[tuple[str, str]]:
    root = Path(__file__).resolve().parent.parent / "data" / "corpus"
    return [(p.stem, p.read_text(encoding="utf-8")) for p in sorted(root.glob("*.md"))]


# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Chunk size is the highest-leverage retrieval knob")
    explain("Sweeping chunk size against retrieval recall typically moves recall@5 by 10–20 "
            "points — more than any reranker will give you. It is also the parameter most "
            "often left at a library default.")
    grid = chunk_size_grid(corpus_texts(), sizes=(128, 220, 320, 512), overlaps=(0, 40, 80))
    dataframe(grid)

    preview = sentence_chunk(corpus_texts()[0][1], "preview",
                             target_tokens=chunk_tokens, overlap_tokens=chunk_overlap)
    st.write(f"Current configuration produces **{len(preview)} chunks** for the first document.")
    with st.expander("First three chunks"):
        for chunk in preview[:3]:
            st.code(f"[{chunk.chunk_id}] ({chunk.n_tokens} tokens)\n{chunk.text}", language="text")

# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Retrieval stage — whatever retrieval misses, no generator recovers")
    if require_api_key("Indexing and retrieval"):
        collection = f"orbital_{chunk_tokens}_{chunk_overlap}"
        index = build_index(chunk_tokens, chunk_overlap, collection)
        st.write(index.stats())

        question = st.selectbox("Question", [c.input for c in dataset if c.is_answerable])
        if st.button("Retrieve", type="primary"):
            comparison = []
            for retrieval_mode in ("dense", "bm25", "hybrid"):
                hits = index.search(question, top_k, mode=retrieval_mode)
                comparison.append({
                    "mode": retrieval_mode,
                    "top_chunk": hits[0].chunk.chunk_id if hits else "",
                    "top_doc": hits[0].chunk.doc_id if hits else "",
                    "chunk_ids": ", ".join(h.chunk.chunk_id.split("::")[1] for h in hits),
                })
            dataframe(comparison)
            explain("Dense retrieval loses on exact identifiers — error codes, SKUs, version "
                    "numbers. Those are exactly the queries a developer-facing RAG system "
                    "receives most often, which is why hybrid is the production default.")

            for hit in index.search(question, top_k, mode=mode):
                with st.expander(f"#{hit.rank} · {hit.chunk.chunk_id} · score {hit.score:.4f}"):
                    st.write(hit.chunk.text)

        st.markdown("**Scoring retrieval against labelled relevance**")
        explain("Mark which retrieved chunks are actually relevant, then the stage metrics "
                "become computable. This is the labelling step teams skip and then wonder "
                "why they cannot attribute a failure.")
        hits = index.search(question, top_k, mode=mode)
        relevant = st.multiselect("Relevant chunk ids",
                                  [h.chunk.chunk_id for h in hits],
                                  default=[h.chunk.chunk_id for h in hits[:1]])
        if relevant:
            report = evaluate_retrieval_stage([h.chunk.chunk_id for h in hits],
                                              set(relevant), k=top_k)
            dataframe([report.as_row()])
            explain("Context precision is rank-weighted: a gold passage at rank 1 counts for "
                    "more than the same passage at rank 5, because generator attention "
                    "degrades with position.")

# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Generation stage — faithfulness, citations, abstention")
    if require_api_key("The RAG pipeline"):
        index = build_index(chunk_tokens, chunk_overlap, f"orbital_{chunk_tokens}_{chunk_overlap}")
        pipeline = RagPipeline(index, k=top_k, mode=mode, rerank=use_reranker)

        case = st.selectbox("Case", dataset.cases,
                            format_func=lambda c: f"{c.case_id} · {c.input[:70]}")
        if st.button("Run pipeline", type="primary"):
            with st.spinner("Retrieving and generating…"):
                response = pipeline.answer(case.input)

            st.markdown("**Answer**")
            st.write(response.answer)

            cols = st.columns(4)
            cols[0].metric("Abstained", "yes" if response.abstained else "no")
            cols[1].metric("Should abstain", "no" if case.is_answerable else "yes")
            cols[2].metric("Latency", f"{response.total_latency_ms:.0f} ms")
            cols[3].metric("Citation coverage", f"{response.citation_coverage():.0%}")

            st.markdown("**Stage trace**")
            dataframe([{"stage": s.stage, "latency_ms": round(s.latency_ms, 1), **{
                k: str(v)[:70] for k, v in s.payload.items()}} for s in response.trace])

            citations = evaluate_citations(response.answer, len(response.contexts))
            dataframe([citations.as_row()])
            if citations.invalid_indices:
                st.error(f"Fabricated citations pointing at passages {citations.invalid_indices} "
                         f"— only {len(response.contexts)} were supplied.")

            with st.spinner("Decomposing claims and verifying against context…"):
                faithfulness = asyncio.run(
                    ClaimVerifier().afaithfulness(response.answer, response.contexts))
            st.metric("Faithfulness", f"{faithfulness.faithfulness:.3f}",
                      help="Fraction of the answer's atomic claims supported by the context.")
            if faithfulness.unsupported_claims:
                st.warning("Unsupported claims:\n\n" +
                           "\n".join(f"- {c}" for c in faithfulness.unsupported_claims))
            dataframe([{"claim": v.claim[:110], "supported": v.supported,
                        "passage": v.supporting_passage, "reason": v.reason[:100]}
                       for v in faithfulness.verdicts])

# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Full suite — stage-wise attribution across the dataset")
    explain("The point of the whole chapter: one end-to-end number tells you the system is at "
            "0.68 and nothing else. Decomposing tells you which of four teams owns the fix.")
    limit = st.slider("Cases to run", 4, len(dataset), 8)
    if st.button("Run suite", type="primary", key="rag_suite") and require_api_key("The RAG suite"):
        index = build_index(chunk_tokens, chunk_overlap, f"orbital_{chunk_tokens}_{chunk_overlap}")
        pipeline = RagPipeline(index, k=top_k, mode=mode, rerank=use_reranker)
        verifier = ClaimVerifier()

        rows, faithfulness_scores, abstained, answerable = [], [], [], []
        progress = st.progress(0.0)
        cases = dataset.cases[:limit]
        for position, case in enumerate(cases, start=1):
            response = pipeline.answer(case.input)
            report = asyncio.run(verifier.afaithfulness(response.answer, response.contexts))
            citations = evaluate_citations(response.answer, len(response.contexts))
            rows.append({
                "case_id": case.case_id,
                "answerable": case.is_answerable,
                "abstained": response.abstained,
                "faithfulness": round(report.faithfulness, 3),
                "citation_rate": round(citations.citation_rate, 3),
                "invalid_citations": len(citations.invalid_indices),
                "latency_ms": round(response.total_latency_ms),
                "answer": response.answer[:120],
            })
            faithfulness_scores.append(report.faithfulness)
            abstained.append(response.abstained)
            answerable.append(case.is_answerable)
            progress.progress(position / len(cases))

        dataframe(rows)
        clean = [s for s in faithfulness_scores if s == s]
        if clean:
            interval_metric("Faithfulness", bootstrap_interval(clean, n_resamples=4000))
        abstention = evaluate_abstention(abstained, answerable)
        dataframe([abstention])
        explain("A suite with no unanswerable questions measures faithfulness only where the "
                "answer was available. 15–25% unanswerable is the working minimum.")

        st.plotly_chart(
            go.Figure(go.Histogram(x=clean, nbinsx=10)).update_layout(
                title="Faithfulness distribution", xaxis_title="faithfulness",
                yaxis_title="cases", height=320, margin=dict(l=10, r=10, t=40, b=10)),
            width="stretch")
