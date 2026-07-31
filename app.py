"""AI & ML Evaluation Lab — Streamlit entry point.

Run with:  streamlit run app.py

Each page in ``pages/`` is the runnable companion to one chapter of the ebook.
Pages that need a Gemini key say so and degrade gracefully; the statistics,
metrics, drift and gate labs run entirely offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evalcore.config import get_settings  # noqa: E402
from evalcore.datasets import DatasetRegistry  # noqa: E402
from evalcore.llm import default_cache  # noqa: E402
from evalcore.runner.store import ExperimentStore  # noqa: E402

st.set_page_config(page_title="AI & ML Evaluation Lab", page_icon="📐", layout="wide")

settings = get_settings()

st.title("AI & ML Evaluation Lab")
st.caption("Companion code project for *AI & ML Evaluation System Design*")

st.markdown(
    """
This application is the runnable half of the book. Every chapter has a page here that
exercises the same `evalcore` code the chapter walks through — same functions, same
defaults, no separate demo path.

**Start here if you are new:** open *1 — Evaluation Foundations*, set the sample size to
80, and read the minimum detectable effect. That single number reframes most arguments
about whether a change "improved" anything.
"""
)

left, right = st.columns([2, 1])

with left:
    st.subheader("Environment")
    checks = {
        "Gemini API key configured": settings.has_api_key,
        "Artifact directory writable": settings.artifact_dir.exists(),
        "Chroma directory present": settings.chroma_dir.exists(),
    }
    for label, ok in checks.items():
        st.write(("✅ " if ok else "⚠️ ") + label)
    if not settings.has_api_key:
        st.info(
            "Pages 1, 2, 8, 9 and 10 run fully offline. Pages 3–7 call Gemini; add a free key "
            "to `.env` to enable them."
        )

    st.subheader("Golden datasets")
    registry = DatasetRegistry()
    for name in registry.available():
        dataset = registry.load(name)
        st.write(
            f"**{name}** — {len(dataset)} cases, hash `{dataset.content_hash()}`, "
            f"difficulty {dataset.difficulty_counts()}"
        )

with right:
    st.subheader("Run configuration")
    st.code(
        f"""generation : {settings.generation_model}
judge      : {settings.judge_model}
embeddings : {settings.embedding_model}
temperature: {settings.temperature}
seed       : {settings.seed}
fingerprint: {settings.fingerprint()}""",
        language="text",
    )

    st.subheader("Response cache")
    st.write(default_cache().stats())
    if st.button("Clear LLM cache"):
        st.write(f"Deleted {default_cache().clear()} cached responses.")

    st.subheader("Experiment store")
    runs = ExperimentStore().list_runs(limit=5)
    st.write(f"{len(runs)} recent runs" if runs else "No runs recorded yet.")
    for record in runs:
        st.caption(f"`{record.run_id}` — {record.suite}, {record.n_ok}/{record.n_total} ok")

st.divider()
st.subheader("Chapter to lab map")
st.table(
    [
        {"Chapter": 1, "Lab": "Evaluation Foundations", "Needs Gemini": "no"},
        {"Chapter": 2, "Lab": "Classical ML Evaluation", "Needs Gemini": "no"},
        {"Chapter": 3, "Lab": "LLM Judge", "Needs Gemini": "yes"},
        {"Chapter": 4, "Lab": "RAG Evaluation", "Needs Gemini": "yes"},
        {"Chapter": 5, "Lab": "Agent Evaluation", "Needs Gemini": "yes"},
        {"Chapter": 6, "Lab": "Safety Evaluation", "Needs Gemini": "yes"},
        {"Chapter": 7, "Lab": "Evaluation Harness", "Needs Gemini": "partly"},
        {"Chapter": 8, "Lab": "Production Monitoring", "Needs Gemini": "no"},
        {"Chapter": 9, "Lab": "Evaluation Infrastructure", "Needs Gemini": "no"},
        {"Chapter": 10, "Lab": "Evaluation Dashboard", "Needs Gemini": "no"},
    ]
)
