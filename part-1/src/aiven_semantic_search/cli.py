"""
CLI for Part 1 (semantic search over a handmade lure catalog).

Design goals for the tutorial audience:
- explicit: each command does one thing
- reproducible: deterministic catalog generation
- cost-aware: easy to smoke-test (max-docs) and batch embeddings (batch-size)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

from .catalog import generate_products, read_jsonl, write_jsonl
from .config import Settings
from .gemini_embedder import GeminiEmbedder
from .opensearch_client import build_index_mapping, get_opensearch_client


def _chunked(items: list[str], batch_size: int) -> list[list[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def cmd_create_index(settings: Settings) -> int:
    client = get_opensearch_client(settings.opensearch_uri, ca_certs=settings.opensearch_ca_certs)
    mapping = build_index_mapping(settings.embed_dim)

    if client.indices.exists(index=settings.opensearch_index):
        print(f"Index already exists: {settings.opensearch_index}")
        return 0

    client.indices.create(index=settings.opensearch_index, body=mapping)
    print(f"Created index: {settings.opensearch_index}")
    return 0


def cmd_reset_index(settings: Settings, *, force: bool) -> int:
    """
    Delete the index (and ALL documents), then recreate it.

    We require `--force` because this is destructive. In the tutorial, this is
    the cleanest way to "start from scratch" without leaving old vectors in
    place.
    """
    if not force:
        raise RuntimeError("Refusing to delete index without --force")

    client = get_opensearch_client(settings.opensearch_uri, ca_certs=settings.opensearch_ca_certs)
    mapping = build_index_mapping(settings.embed_dim)

    if client.indices.exists(index=settings.opensearch_index):
        client.indices.delete(index=settings.opensearch_index)
        print(f"Deleted index: {settings.opensearch_index}")
    else:
        print(f"Index does not exist: {settings.opensearch_index}")

    client.indices.create(index=settings.opensearch_index, body=mapping)
    print(f"Created index: {settings.opensearch_index}")
    return 0


def cmd_generate_catalog(*, out_path: str, count: int, seed: int) -> int:
    products = generate_products(count, seed=seed)
    write_jsonl(out_path, (p.to_dict() for p in products))
    print(f"Wrote {len(products)} products to {out_path}")
    return 0


def cmd_estimate_embedding_cost(*, path: str) -> int:
    """
    Estimate Vertex AI embedding cost from a JSONL catalog.

    Transparency note:
    - Vertex AI charges by input tokens, but tokenization depends on the model.
    - For a tutorial, we use a simple approximation: 1 token ~= 4 characters.
      This is not exact, but it is directionally useful.
    """
    rows = read_jsonl(path)
    descriptions = [str(r.get("description", "")) for r in rows]
    if any(d == "" for d in descriptions):
        raise RuntimeError("All rows must have a non-empty 'description' field")

    total_chars = sum(len(d) for d in descriptions)
    approx_tokens = math.ceil(total_chars / 4)

    # From the Vertex AI Generative AI pricing page (global):
    # Gemini Embedding input: $0.00015 / 1,000 input tokens (online).
    price_per_1k_tokens = 0.00015
    approx_cost_usd = (approx_tokens / 1000.0) * price_per_1k_tokens

    print(
        json.dumps(
            {
                "documents": len(rows),
                "total_description_chars": total_chars,
                "approx_input_tokens": approx_tokens,
                "pricing_assumption_usd_per_1k_tokens": price_per_1k_tokens,
                "approx_embedding_cost_usd": round(approx_cost_usd, 6),
                "note": "Token estimate uses 1 token ~= 4 characters (rough). See Vertex AI pricing for exact billing.",
            },
            indent=2,
        )
    )
    return 0


def cmd_index_catalog(
    settings: Settings,
    *,
    path: str,
    refresh: bool,
    batch_size: int,
    max_docs: int | None,
) -> int:
    client = get_opensearch_client(settings.opensearch_uri, ca_certs=settings.opensearch_ca_certs)
    embedder = GeminiEmbedder(
        model=settings.gemini_embed_model,
        project_id=settings.gcp_project_id,
        location=settings.gcp_location,
        output_dimensionality=settings.embed_dim,
    )

    rows = read_jsonl(path)
    if max_docs is not None:
        rows = rows[:max_docs]

    descriptions = [str(r.get("description", "")) for r in rows]
    if any(d == "" for d in descriptions):
        raise RuntimeError("All rows must have a non-empty 'description' field")

    embedded_vectors: list[list[float]] = []
    for batch in _chunked(descriptions, batch_size):
        embedded_vectors.extend(embedder.embed_documents(batch))

    if len(embedded_vectors) != len(rows):
        raise RuntimeError("Embedding count mismatch (unexpected Vertex response)")

    for r, v in zip(rows, embedded_vectors, strict=True):
        doc_id = str(r.get("id") or "")
        if not doc_id:
            raise RuntimeError("All rows must have an 'id' field")

        body = dict(r)
        body["description_vector"] = v
        client.index(index=settings.opensearch_index, id=doc_id, body=body, refresh=refresh)

    print(f"Indexed {len(rows)} documents from {path} into {settings.opensearch_index}")
    return 0


def cmd_search(settings: Settings, *, query: str, k: int) -> int:
    client = get_opensearch_client(settings.opensearch_uri, ca_certs=settings.opensearch_ca_certs)
    embedder = GeminiEmbedder(
        model=settings.gemini_embed_model,
        project_id=settings.gcp_project_id,
        location=settings.gcp_location,
        output_dimensionality=settings.embed_dim,
    )

    query_vector = embedder.embed_query(query)

    search_body: dict[str, Any] = {
        "size": k,
        "query": {
            "knn": {
                "description_vector": {
                    "vector": query_vector,
                    "k": k,
                }
            }
        },
    }

    results = client.search(index=settings.opensearch_index, body=search_body)
    hits = results.get("hits", {}).get("hits", [])
    simplified = [
        {
            "score": h.get("_score"),
            "id": h.get("_id"),
            "name": (h.get("_source") or {}).get("name"),
            "category": (h.get("_source") or {}).get("category"),
            "description": (h.get("_source") or {}).get("description"),
            "price_usd": (h.get("_source") or {}).get("price_usd"),
        }
        for h in hits
    ]

    print(json.dumps({"query": query, "results": simplified}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aiven-semantic-search-part-1")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("create-index", help="Create the OpenSearch k-NN index")

    sp = sub.add_parser("reset-index", help="Delete and recreate the index (DESTRUCTIVE)")
    sp.add_argument("--force", action="store_true", help="Actually delete the index")

    sp = sub.add_parser("generate-catalog", help="Generate a JSONL lure catalog")
    sp.add_argument("--out", default="data/lures.jsonl", help="Output JSONL path")
    sp.add_argument("--count", type=int, default=100, help="Number of lures to generate")
    sp.add_argument("--seed", type=int, default=42, help="Random seed (deterministic output)")

    sp = sub.add_parser("estimate-embedding-cost", help="Estimate embedding cost for a JSONL catalog")
    sp.add_argument("path", help="Path to a JSONL catalog")

    sp = sub.add_parser("index-catalog", help="Index a JSONL catalog file (embed + upsert)")
    sp.add_argument("path", help="Path to a JSONL catalog (one product per line)")
    sp.add_argument("--refresh", action="store_true", help="Refresh index after each insert")
    sp.add_argument("--batch-size", type=int, default=5, help="How many descriptions to embed per request")
    sp.add_argument("--max-docs", type=int, default=0, help="If >0, only index the first N docs")

    sp = sub.add_parser("search", help="Run a semantic vector search")
    sp.add_argument("query", help="Query text to embed and search for")
    sp.add_argument("--k", type=int, default=5, help="Number of nearest neighbors to return")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()

    if args.command == "create-index":
        return cmd_create_index(settings)
    if args.command == "reset-index":
        return cmd_reset_index(settings, force=bool(args.force))
    if args.command == "generate-catalog":
        return cmd_generate_catalog(out_path=str(args.out), count=int(args.count), seed=int(args.seed))
    if args.command == "estimate-embedding-cost":
        return cmd_estimate_embedding_cost(path=str(args.path))
    if args.command == "index-catalog":
        max_docs = int(args.max_docs)
        return cmd_index_catalog(
            settings,
            path=str(args.path),
            refresh=bool(args.refresh),
            batch_size=int(args.batch_size),
            max_docs=(max_docs if max_docs > 0 else None),
        )
    if args.command == "search":
        return cmd_search(settings, query=str(args.query), k=int(args.k))

    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

