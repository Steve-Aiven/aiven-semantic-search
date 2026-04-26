"""
Configuration loading (Part 1).

We deliberately use environment variables (loaded from a local `.env` file)
instead of hardcoding secrets like the Aiven OpenSearch password or Google
Cloud project ID into the source. That keeps secrets out of git and makes it
easy for a reader to point the demo at their own Aiven service / GCP project.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str | None = None) -> str:
    """Read a required environment variable, raising a clear error if missing."""
    v = os.environ.get(name, default)
    if v is None or v == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


@dataclass(frozen=True)
class Settings:
    # Connection string for Aiven for OpenSearch, including credentials.
    # Format: https://USER:PASSWORD@HOST:PORT
    opensearch_uri: str

    # Name of the OpenSearch index that will hold the lure catalog + vectors.
    opensearch_index: str

    # Google Cloud project that has Vertex AI enabled.
    gcp_project_id: str

    # Vertex AI region, e.g. "us-central1".
    gcp_location: str

    # Gemini embedding model to use for both documents and queries.
    gemini_embed_model: str

    # Output dimensionality for embeddings. MUST match the `dimension` we set
    # on the OpenSearch `knn_vector` field.
    embed_dim: int

    # Optional path to an Aiven project CA certificate file (PEM).
    opensearch_ca_certs: str | None

    @staticmethod
    def from_env() -> "Settings":
        return Settings(
            opensearch_uri=_env("OPENSEARCH_URI"),
            opensearch_index=os.environ.get("OPENSEARCH_INDEX", "lures"),
            gcp_project_id=_env("GCP_PROJECT_ID"),
            gcp_location=os.environ.get("GCP_LOCATION", "us-central1"),
            gemini_embed_model=os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-001"),
            embed_dim=int(os.environ.get("EMBED_DIM", "3072")),
            opensearch_ca_certs=os.environ.get("OPENSEARCH_CA_CERTS") or None,
        )

