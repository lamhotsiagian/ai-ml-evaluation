"""Shared Streamlit components used by every lab page."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evalcore.config import get_settings  # noqa: E402
from evalcore.stats import Interval  # noqa: E402


def page_header(title: str, chapter: int, subtitle: str) -> None:
    st.title(title)
    st.caption(f"Chapter {chapter} lab — {subtitle}")


def require_api_key(feature: str = "this lab") -> bool:
    """Gate live-model features behind a configured key, without crashing.

    Offline labs (statistics, metrics, drift, gates) must keep working with no
    key at all; only the pages that call Gemini are gated.
    """
    settings = get_settings()
    if settings.has_api_key:
        return True
    st.warning(
        f"{feature} calls the Gemini API. Copy `.env.example` to `.env` and set "
        "`GOOGLE_API_KEY` with a free key from https://aistudio.google.com/app/apikey, "
        "then restart the app."
    )
    return False


def interval_metric(label: str, interval: Interval, *, digits: int = 4, help_text: str = "") -> None:
    """Render a metric with its confidence interval as the delta line."""
    st.metric(
        label,
        f"{interval.estimate:.{digits}f}",
        delta=f"±{interval.half_width:.{digits}f} ({interval.confidence:.0%} CI)",
        delta_color="off",
        help=help_text or f"{interval.method}, n={interval.n}",
    )


def dataframe(rows: Iterable[dict[str, Any]], *, height: int | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        st.info("No rows to display.")
        return frame
    st.dataframe(frame, width="stretch", height=height, hide_index=True)
    return frame


def verdict_banner(passed: bool, message: str) -> None:
    (st.success if passed else st.error)(message)


def settings_sidebar() -> None:
    settings = get_settings()
    with st.sidebar:
        st.subheader("Run configuration")
        st.caption(f"Fingerprint `{settings.fingerprint()}`")
        st.write(
            {
                "generation model": settings.generation_model,
                "judge model": settings.judge_model,
                "embeddings": settings.embedding_model,
                "temperature": settings.temperature,
                "concurrency": settings.max_concurrency,
                "rpm limit": settings.requests_per_minute,
                "API key set": settings.has_api_key,
            }
        )
        st.caption(
            "Two runs are comparable only when their fingerprints match. "
            "The regression gate enforces this."
        )


def explain(text: str) -> None:
    """A consistent style for the 'why this matters' note under each control."""
    st.caption(text)
