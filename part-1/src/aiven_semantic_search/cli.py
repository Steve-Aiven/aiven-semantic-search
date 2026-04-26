"""
Command-line interface for Part 1.

Each subcommand corresponds to one step in the tutorial:

    create-index            Create the OpenSearch index with the k-NN mapping.
    reset-index             Wipe and recreate the index (start over).
    generate-catalog        Build the JSONL lure catalog from scratch.
    estimate-embedding-cost Estimate Vertex AI embedding cost before you index.
    index-catalog           Embed descriptions with Gemini and write to OpenSearch.
    search                  Embed a query and retrieve the nearest products.

The commands are intentionally separate so you can run them one at a time and
observe what each step does. In a production system you would combine these steps
into a pipeline or a background job, but for learning purposes, explicit steps
make the data flow easier to follow.

The general flow of data through these commands is:

    catalog.jsonl  -->  Gemini (embed)  -->  OpenSearch (index)
                                                   |
                                    query string -> Gemini (embed)
                                                   |
                                              OpenSearch (k-NN search)
                                                   |
                                             ranked results
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
    """
    Split a list into consecutive chunks of at most `batch_size` items.

    This is used when calling the Vertex AI embedding API. The API has a maximum
    number of inputs per request. Sending descriptions in small batches keeps
    each request within that limit and also makes it easier to add retry logic
    or progress reporting in future iterations.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def cmd_create_index(settings: Settings) -> int:
    """
    Create the OpenSearch index with the k-NN vector mapping.

    This command is idempotent - running it a second time when the index already
    exists prints a message and exits cleanly. That makes it safe to include at
    the top of a setup script or Makefile without checking first.

    If you need to change the index mapping (for example, to add a new field or
    change the embedding dimension), use `reset-index --force` instead. OpenSearch
    does not allow most mapping changes on a live index.
    """
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
    Delete the index and all of its documents, then recreate it with a clean mapping.

    Deleting an index in OpenSearch removes every document and every stored vector
    immediately. The `--force` flag is required to prevent accidental data loss
    when the command is run non-interactively (e.g. in a script).

    When would you use this?
    - You changed the catalog schema and need to re-index from scratch.
    - You changed the embedding dimension and the old index mapping no longer matches.
    - You want to start the tutorial over with a clean slate.
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
    """
    Generate the lure catalog and write it to a JSONL file.

    The seed makes generation deterministic. Using the default seed of 42 means
    every reader of this tutorial gets the same catalog, which makes it possible
    to compare search results and discuss specific products by ID or SKU.
    """
    products = generate_products(count, seed=seed)
    write_jsonl(out_path, (p.to_dict() for p in products))
    print(f"Wrote {len(products)} products to {out_path}")
    return 0


def cmd_estimate_embedding_cost(*, path: str) -> int:
    """
    Print a projected Vertex AI cost for embedding all descriptions in a
    JSONL catalog, before you spend anything on the actual embedding run.

    Why approximation instead of exact token counts?
    The Vertex AI countTokens API only supports generative models (e.g.
    gemini-2.0-flash). Embedding models like gemini-embedding-001 return an
    INVALID_ARGUMENT error when you call countTokens against them. Google does
    not expose a public token-counting endpoint for embedding models.

    We use the widely-cited approximation of 1 token per 4 characters, which
    is directionally accurate for English prose. For the short product
    descriptions in this catalog the error is small - typically a few percent
    in either direction.

    The pricing used here is the online (synchronous) rate for
    `gemini-embedding-001` as of April 2026. Always verify current rates at
    https://cloud.google.com/vertex-ai/generative-ai/pricing before making
    budget decisions.
    """
    rows = read_jsonl(path)
    descriptions = [str(r.get("description", "")) for r in rows]
    if any(d == "" for d in descriptions):
        raise RuntimeError("All rows must have a non-empty 'description' field")

    total_chars = sum(len(d) for d in descriptions)
    approx_tokens = math.ceil(total_chars / 4)

    price_per_1k_tokens = 0.00015  # USD, online, gemini-embedding-001, global region
    approx_cost_usd = (approx_tokens / 1000.0) * price_per_1k_tokens

    print(
        json.dumps(
            {
                "documents": len(rows),
                "total_description_chars": total_chars,
                "approx_input_tokens": approx_tokens,
                "pricing_usd_per_1k_tokens": price_per_1k_tokens,
                "approx_embedding_cost_usd": round(approx_cost_usd, 6),
                "note": (
                    "Token count uses 1 token ~= 4 chars (approximation - "
                    "countTokens is not supported for embedding models). "
                    "See https://cloud.google.com/vertex-ai/generative-ai/pricing for current rates."
                ),
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
    """
    Read a JSONL catalog, embed each description, and write the documents to OpenSearch.

    This is where the semantic layer is created. Each product description is
    converted into a 3072-dimensional vector by Gemini. That vector is stored
    alongside the original document fields in OpenSearch. At search time, a query
    goes through the same embedding process and OpenSearch finds the stored vectors
    that are closest to the query vector.

    About `--batch-size`:
    Descriptions are sent to Vertex AI in groups of `batch_size`. Larger batches
    reduce the number of API round-trips but increase the size of each request.
    The default of 5 is conservative and works reliably on all plan sizes.

    About `--max-docs`:
    Use this to index a small subset of the catalog first (e.g. `--max-docs 10`)
    to verify the pipeline works before committing to embedding the full catalog.
    This also lets you control costs while iterating.

    About `refresh`:
    Setting `refresh=True` makes newly indexed documents immediately searchable.
    OpenSearch normally refreshes on a timer (every second by default). For this
    tutorial we pass refresh so that you can run `search` immediately after
    `index-catalog` and see results. In high-throughput production indexing you
    would turn this off to avoid the per-document overhead.
    """
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

    # Embed all descriptions, respecting the batch size limit.
    embedded_vectors: list[list[float]] = []
    for batch in _chunked(descriptions, batch_size):
        embedded_vectors.extend(embedder.embed_documents(batch))

    if len(embedded_vectors) != len(rows):
        raise RuntimeError("Embedding count mismatch - Vertex AI returned an unexpected number of vectors")

    for r, v in zip(rows, embedded_vectors, strict=True):
        doc_id = str(r.get("id") or "")
        if not doc_id:
            raise RuntimeError("All rows must have a non-empty 'id' field")

        body = dict(r)
        body["description_vector"] = v  # attach the embedding vector to the document
        client.index(index=settings.opensearch_index, id=doc_id, body=body, refresh=refresh)

    print(f"Indexed {len(rows)} documents from {path} into {settings.opensearch_index}")
    return 0


def cmd_search(settings: Settings, *, query: str, k: int) -> int:
    """
    Embed a search query and return the k nearest products from OpenSearch.

    The k-NN query in OpenSearch works like this:
    1. The query string is embedded into a vector using Gemini (RETRIEVAL_QUERY).
    2. That vector is sent to OpenSearch as a `knn` query against the
       `description_vector` field.
    3. OpenSearch uses an approximate nearest-neighbor algorithm (ANN) to find
       the `k` stored vectors closest to the query vector.
    4. Results are returned ranked by similarity score, highest first.

    The score is a cosine similarity value between 0 and 1. Higher is more similar.
    Notice that results do not require any keyword overlap with the query - they
    are ranked purely on vector distance. That is what makes this semantic search.

    The response only includes human-readable fields. The embedding vectors are
    not returned because they are large (3072 floats each) and not meaningful to
    a reader.
    """
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
            "score":       h.get("_score"),
            "id":          h.get("_id"),
            "sku":         (h.get("_source") or {}).get("sku"),
            "name":        (h.get("_source") or {}).get("name"),
            "category":    (h.get("_source") or {}).get("category"),
            "species":     (h.get("_source") or {}).get("species"),
            "depth":       (h.get("_source") or {}).get("depth"),
            "color":       (h.get("_source") or {}).get("color"),
            "description": (h.get("_source") or {}).get("description"),
            "price_usd":   (h.get("_source") or {}).get("price_usd"),
            "inventory":   (h.get("_source") or {}).get("inventory"),
        }
        for h in hits
    ]

    print(json.dumps({"query": query, "results": simplified}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aiven-semantic-search-part-1")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("create-index", help="Create the OpenSearch k-NN index")

    sp = sub.add_parser("reset-index", help="Delete and recreate the index (DESTRUCTIVE - requires --force)")
    sp.add_argument("--force", action="store_true", help="Confirm deletion of all documents and vectors")

    sp = sub.add_parser("generate-catalog", help="Generate a JSONL lure catalog")
    sp.add_argument("--out", default="data/lures.jsonl", help="Output JSONL path (default: data/lures.jsonl)")
    sp.add_argument("--count", type=int, default=100, help="Number of lures to generate (default: 100)")
    sp.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")

    sp = sub.add_parser(
        "estimate-embedding-cost",
        help="Estimate Vertex AI embedding cost before indexing",
    )
    sp.add_argument("path", help="Path to a JSONL catalog file")

    sp = sub.add_parser("index-catalog", help="Embed descriptions and index documents into OpenSearch")
    sp.add_argument("path", help="Path to a JSONL catalog file (one product per line)")
    sp.add_argument("--refresh", action="store_true", help="Make documents searchable immediately after each insert")
    sp.add_argument("--batch-size", type=int, default=5, help="Descriptions per Vertex AI embedding request (default: 5)")
    sp.add_argument("--max-docs", type=int, default=0, help="Stop after N documents (0 = index all)")

    sp = sub.add_parser("search", help="Embed a query and return the nearest products")
    sp.add_argument("query", help="Natural-language search query")
    sp.add_argument("--k", type=int, default=5, help="Number of results to return (default: 5)")

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
