# Aiven semantic search (tutorial series)

This repo is a **multi-part** walkthrough. Each part lives in its own directory so dependencies, commands, and blog posts can stay in sync.

## Current part

- **[Part 1](part-1/README.md)** - Handmade fishing lure catalog, **Aiven for OpenSearch**, **Vertex AI (Gemini) embeddings**, semantic search and cost estimates.

```mermaid
graph LR
    P1["Part 1\nCatalog · Embeddings\nSemantic Search"]:::active
    P2["Part 2\ncoming soon"]:::planned
    P3["· · ·"]:::planned
    P1 --> P2 --> P3
    classDef active  fill:#1b4332,color:#fff,stroke:none
    classDef planned fill:#f1f3f5,color:#aaa,stroke:#ccc,stroke-dasharray:5
```

## Quick start

```bash
cd part-1
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
cp .env.example .env
# Edit .env with your Aiven service URI, GCP project, and optional OPENSEARCH_INDEX (default: lures)
set -a && source .env && set +a
gcloud auth application-default login
```

Full steps, reset-index, and doc links: see [part-1/README.md](part-1/README.md).

## Security

Do not commit credentials. See [SECURITY.md](SECURITY.md).
