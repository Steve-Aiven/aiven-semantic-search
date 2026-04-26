## Part 1 - Semantic search for handmade fishing lures

This part builds a small, reproducible semantic search demo:

- **Catalog**: 100 small-batch, handmade fishing lures (generated deterministically)
- **Embeddings**: Gemini embeddings via **Vertex AI**
- **Vector search**: Aiven for OpenSearch (`knn_vector` + k-NN query)

The goal of Part 1 is to be extremely clear and reproducible. Later parts in the series will build on this foundation (recommendations, hybrid search, and beyond).

Do not commit secrets. See [../SECURITY.md](../SECURITY.md).

## Prerequisites

- **Python 3.10+**
- **An Aiven for OpenSearch service**
- **A Google Cloud project with Vertex AI enabled**
- **Local Google auth** (Application Default Credentials)

## Setup

From the `part-1/` folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

## Configure environment variables

```bash
cp .env.example .env
set -a && source .env && set +a
```

Set **`OPENSEARCH_URI`**, **`GCP_PROJECT_ID`**, and optionally **`OPENSEARCH_INDEX`** (default in code: `lures`). Use one index name consistently; if you previously used `products`, either point `OPENSEARCH_INDEX` at that index or run `reset-index --force` after switching to avoid mixing schemas.

## Authenticate to Google Cloud (Vertex AI)

```bash
gcloud auth application-default login
```

## Run the demo

### (Optional) Start over from scratch (delete the index)

If you want to wipe all documents + vectors and restart the tutorial:

```bash
python3 -m aiven_semantic_search reset-index --force
```

### 1) Create the vector index in OpenSearch

```bash
python3 -m aiven_semantic_search create-index
```

### 2) Generate the lure catalog (100 products)

```bash
python3 -m aiven_semantic_search generate-catalog --out data/lures.jsonl --count 100 --seed 42
```

### 3) Index the catalog (embed + upsert)

```bash
python3 -m aiven_semantic_search index-catalog data/lures.jsonl --batch-size 5 --refresh
```

### 4) Run semantic searches

```bash
python3 -m aiven_semantic_search search "small jerkbait for clear water trout" --k 5
python3 -m aiven_semantic_search search "weedless soft plastic for bass in heavy cover" --k 5
python3 -m aiven_semantic_search search "topwater bait for early morning on a calm lake" --k 5
```

## Cost transparency (embeddings)

This demo's main variable cost is **Vertex AI embeddings**.

Vertex AI charges for embeddings by **input tokens**, not "per embedding".
See the pricing page: `https://cloud.google.com/vertex-ai/generative-ai/pricing`

Rough estimate (online):  
\( \text{cost} \approx \frac{\text{total input tokens}}{1000} \times 0.00015 \) USD

For a 100-item catalog, costs are typically **a few cents** unless your descriptions are very long or you chunk them aggressively.

## References (source of truth)

- **Aiven**
  - Connect to Aiven for OpenSearch with Python: `https://developer.aiven.io/docs/products/opensearch/howto/connect-with-python.md`
  - TLS/SSL certificates (project CA vs public CA): `https://aiven.io/docs/platform/concepts/tls-ssl-certificates`
- **OpenSearch (vector search)**
  - k-NN index: `https://docs.opensearch.org/2.14/search-plugins/knn/knn-index`
  - `knn_vector` field type: `https://docs.opensearch.org/2.17/field-types/supported-field-types/knn-vector/`
- **Google Cloud / Vertex AI**
  - Enable Vertex AI API: `https://cloud.google.com/vertex-ai/docs/start/cloud-environment`
  - Get text embeddings: `https://docs.cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings`
  - Text embeddings model reference: `https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api`
  - ADC overview: `https://docs.cloud.google.com/docs/authentication/application-default-credentials`
  - `gcloud auth application-default login`: `https://docs.cloud.google.com/sdk/gcloud/reference/auth/application-default/login`

