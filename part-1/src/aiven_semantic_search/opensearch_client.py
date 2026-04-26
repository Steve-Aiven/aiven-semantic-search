"""
OpenSearch connection + index mapping helpers (Part 1).

We follow Aiven's recommended pattern:

- Keep your Aiven OpenSearch "service URI" in an environment variable
  (it includes credentials).
- Use TLS and verify certificates.
- Optionally configure the Aiven project CA certificate if your service type
  requires it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse, unquote

from opensearchpy import OpenSearch


@dataclass(frozen=True)
class ParsedOpenSearchUri:
    host: str
    port: int
    username: str | None
    password: str | None
    use_ssl: bool


def parse_opensearch_uri(uri: str) -> ParsedOpenSearchUri:
    parsed = urlparse(uri)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported OPENSEARCH_URI scheme: {parsed.scheme!r}")
    if not parsed.hostname or not parsed.port:
        raise ValueError("OPENSEARCH_URI must include hostname and port")

    username = unquote(parsed.username) if parsed.username else None
    password = unquote(parsed.password) if parsed.password else None

    return ParsedOpenSearchUri(
        host=parsed.hostname,
        port=int(parsed.port),
        username=username,
        password=password,
        use_ssl=(parsed.scheme == "https"),
    )


def get_opensearch_client(uri: str, *, ca_certs: str | None = None) -> OpenSearch:
    p = parse_opensearch_uri(uri)

    http_auth = None
    if p.username is not None and p.password is not None:
        http_auth = (p.username, p.password)

    client_kwargs: dict[str, Any] = {
        "hosts": [{"host": p.host, "port": p.port}],
        "http_auth": http_auth,
        "use_ssl": p.use_ssl,
        "verify_certs": True,
        "ssl_show_warn": False,
        "http_compress": True,
        "timeout": 30,
    }
    if ca_certs:
        client_kwargs["ca_certs"] = ca_certs

    return OpenSearch(**client_kwargs)


def build_index_mapping(embed_dim: int) -> dict[str, Any]:
    """
    Vector index mapping for the lure catalog.

    Best practice for a demo:
    - store human-readable fields (name/category/description)
    - store the embedding vector in a dedicated `knn_vector` field
    """
    return {
        "settings": {
            "index": {
                "number_of_shards": "1",
                "number_of_replicas": "0",
                "knn": True,
            }
        },
        "mappings": {
            "properties": {
                "name": {"type": "text"},
                "category": {"type": "keyword"},
                "species": {"type": "keyword"},
                "water": {"type": "keyword"},
                "depth": {"type": "keyword"},
                "color": {"type": "keyword"},
                "weight_oz": {"type": "float"},
                "price_usd": {"type": "float"},
                "description": {"type": "text"},
                "description_vector": {"type": "knn_vector", "dimension": embed_dim},
            }
        },
    }

