"""Central configuration for the AI & ML Evaluation Lab.

Every module reads its knobs from :func:`get_settings`. Nothing anywhere in the
code base reads ``os.environ`` directly, so an evaluation run can be reproduced
by capturing a single object: the settings snapshot is written into every
experiment record (see :mod:`evalcore.runner.store`).
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration, loaded from environment variables or ``.env``.

    The ``EVAL_`` prefix is applied to every field except ``google_api_key``,
    which keeps its conventional Google name so the standard Google SDKs pick
    it up without extra wiring.
    """

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_prefix="EVAL_",
        extra="ignore",
    )

    google_api_key: SecretStr = Field(default=SecretStr(""), alias="GOOGLE_API_KEY")

    generation_model: str = "gemini-2.0-flash"
    judge_model: str = "gemini-2.0-flash-lite"
    embedding_model: str = "models/text-embedding-004"

    temperature: float = 0.0
    judge_temperature: float = 0.0
    seed: int = 1337

    max_concurrency: int = 4
    requests_per_minute: int = 14
    max_retries: int = 5
    request_timeout_s: float = 90.0

    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts")
    chroma_dir: Path = Path("artifacts/chroma")
    cache_path: Path = Path("artifacts/llm_cache.sqlite")
    store_path: Path = Path("artifacts/experiments.sqlite")

    @field_validator("data_dir", "artifact_dir", "chroma_dir", "cache_path", "store_path")
    @classmethod
    def _absolutise(cls, value: Path) -> Path:
        """Resolve relative paths against the project root, not the CWD.

        Streamlit, pytest and ``python -m`` all start with different working
        directories; anchoring on the package location keeps artefacts in one
        place regardless of how the code was launched.
        """
        return value if value.is_absolute() else (PROJECT_ROOT / value)

    @property
    def has_api_key(self) -> bool:
        return bool(self.google_api_key.get_secret_value().strip())

    def ensure_dirs(self) -> None:
        """Create every artefact directory the lab writes to."""
        for path in (self.artifact_dir, self.chroma_dir):
            path.mkdir(parents=True, exist_ok=True)
        for path in (self.cache_path, self.store_path):
            path.parent.mkdir(parents=True, exist_ok=True)

    def fingerprint(self) -> str:
        """A stable hash of the settings that can change an evaluation result.

        Paths and concurrency are excluded on purpose: they affect *where* and
        *how fast* a run happens, never *what* it scores. Two runs with the same
        fingerprint are comparable; two runs with different fingerprints are not,
        and the regression gate refuses to compare them.
        """
        payload = {
            "generation_model": self.generation_model,
            "judge_model": self.judge_model,
            "embedding_model": self.embedding_model,
            "temperature": self.temperature,
            "judge_temperature": self.judge_temperature,
            "seed": self.seed,
        }
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    settings = Settings()
    settings.ensure_dirs()
    return settings


def reset_settings_cache() -> None:
    """Drop the cached singleton (used by tests that patch the environment)."""
    get_settings.cache_clear()
