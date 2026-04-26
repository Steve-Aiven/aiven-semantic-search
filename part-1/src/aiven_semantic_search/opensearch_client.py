"""
OpenSearch connection and index mapping helpers for Part 1.

How Aiven's connection string works
-------------------------------------
Aiven gives you a single "service URI" that looks like:

    https://avnadmin:PASSWORD@my-service.aivencloud.com:24526

This bundles the hostname, port, username, and password together. The
`opensearch-py` client does not accept a URI string directly - it expects the
host and port as separate arguments and credentials as an `http_auth` tuple.

`parse_opensearch_uri` handles the translation. It also URL-decodes the
username and password because Aiven passwords can contain special characters
that are percent-encoded in a URI (e.g. `%40` for `@`).

Why verify TLS certificates?
------------------------------
Disabling certificate verification (`verify_certs=False`) is a common shortcut
that is easy to find in quick-start examples online. It defeats the purpose of
TLS entirely because it allows a man-in-the-middle attacker to intercept the
connection. All traffic to Aiven services is protected by TLS. Leave verification
on and provide a CA bundle if needed.

What is a k-NN index?
-----------------------
A regular OpenSearch index stores and retrieves documents by field values and
full-text relevance. A k-NN index adds the ability to store high-dimensional
vectors and find the closest ones to a query vector. That is the foundation of
semantic search.

Enabling k-NN requires two things:
1. Set `"knn": true` in the index settings - this tells the k-NN plugin to build
   the nearest-neighbor data structures for this index.
2. Declare a `knn_vector` field in the mappings with the correct `dimension` -
   this is where each document's embedding vector is stored.

The dimension must match the output size of the embedding model you are using.
Here we use `gemini-embedding-001` with `output_dimensionality=3072`, so the
index mapping also sets `dimension: 3072`. A mismatch causes indexing to fail.
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
    """
    Break an Aiven-style service URI into components the OpenSearch client accepts.

    We validate that the URI has both a hostname and a port because a missing
    port is the most common cause of "Failed to resolve host" errors when
    copying URIs from the Aiven console.
    """
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
    """
    Build an authenticated, TLS-verified OpenSearch client from an Aiven service URI.

    `http_compress=True` enables gzip compression on request bodies. This reduces
    bandwidth when indexing large documents or large batches, at the cost of a
    small amount of CPU on both ends.

    `timeout=30` prevents the client from hanging indefinitely if the Aiven free
    tier service has auto-paused due to inactivity. Wake the service in the Aiven
    console and try again if you see a timeout error.
    """
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
    Return the index settings and field mappings for the lure catalog.

    OpenSearch mappings serve two purposes here:
    - Structured fields (sku, category, species, price, etc.) use `keyword`,
      `float`, or `integer` types so they can be filtered and sorted precisely.
      We will use these for faceted filtering in later parts of the series.
    - The `description` field uses `text` for full-text analysis. This is the
      field whose content we embed into a vector.
    - `description_vector` is a `knn_vector` field. OpenSearch uses this to run
      approximate nearest-neighbor (ANN) searches at query time.

    One shard, zero replicas is appropriate for a single-node free tier service.
    In a production multi-node cluster you would increase both values.
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
                "sku":              {"type": "keyword"},
                "name":             {"type": "text"},
                "category":         {"type": "keyword"},
                "species":          {"type": "keyword"},
                "water":            {"type": "keyword"},
                "depth":            {"type": "keyword"},
                "color":            {"type": "keyword"},
                "style":            {"type": "keyword"},
                "material":         {"type": "keyword"},
                "buoyancy":         {"type": "keyword"},
                "hook":             {"type": "keyword"},
                "size_in":          {"type": "float"},
                "weight_oz":        {"type": "float"},
                "batch_size":       {"type": "integer"},
                "inventory":        {"type": "integer"},
                "lead_time_days":   {"type": "integer"},
                "price_usd":        {"type": "float"},
                "description":      {"type": "text"},
                "description_vector": {"type": "knn_vector", "dimension": embed_dim},
            }
        },
    }
