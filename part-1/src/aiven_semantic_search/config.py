"""
Configuration loading for Part 1.

Why environment variables?
--------------------------
The connection string for Aiven for OpenSearch includes a username and password.
The Google Cloud project ID is not secret by itself, but it is account-specific.
Neither belongs in source code that gets committed to git.

Environment variables loaded from a local `.env` file are a well-established
pattern for this. The file is gitignored so credentials stay off the repo, and
anyone following along can swap in their own service by editing one file - no
code changes required.

The `Settings` dataclass is `frozen=True`, which means once it is created it
cannot be changed. That makes it safe to pass a single `Settings` instance to
every function that needs it without worrying that one function will accidentally
modify a value another function depends on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str | None = None) -> str:
    """
    Read a named environment variable and raise a clear error if it is missing.

    We intentionally do not print the value in the error message. Environment
    variables often contain secrets, and printing them - even in an error - can
    expose them in CI logs, terminal history, or screenshots.
    """
    v = os.environ.get(name, default)
    if v is None or v == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


@dataclass(frozen=True)
class Settings:
    # The Aiven service URI bundles host, port, and credentials in one string.
    # Format: https://USER:PASSWORD@HOST:PORT
    # Treat this value like a password - do not log it or print it.
    opensearch_uri: str

    # The name of the OpenSearch index that stores the lure catalog and its
    # embedding vectors. Defaults to "lures" to avoid collisions with other
    # indexes on the same service.
    opensearch_index: str

    # The Google Cloud project ID where Vertex AI is enabled.
    gcp_project_id: str

    # Vertex AI operates per region. "us-central1" is the most widely available
    # region for Gemini embedding models.
    gcp_location: str

    # The Gemini embedding model to use. Both indexing and search use the same
    # model so that document and query vectors live in the same embedding space.
    gemini_embed_model: str

    # The number of dimensions in the output embedding vector.
    # This MUST match the `dimension` value declared in the OpenSearch index
    # mapping. If they differ, indexing will fail with a dimension mismatch error.
    embed_dim: int

    @staticmethod
    def from_env() -> "Settings":
        """Build a Settings instance from environment variables."""
        return Settings(
            opensearch_uri=_env("OPENSEARCH_URI"),
            opensearch_index=os.environ.get("OPENSEARCH_INDEX", "lures"),
            gcp_project_id=_env("GCP_PROJECT_ID"),
            gcp_location=os.environ.get("GCP_LOCATION", "us-central1"),
            gemini_embed_model=os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-001"),
            embed_dim=int(os.environ.get("EMBED_DIM", "3072")),
        )
