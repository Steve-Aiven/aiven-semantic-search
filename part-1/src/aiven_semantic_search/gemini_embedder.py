"""
Gemini embedding wrapper for Part 1.

What is an embedding?
----------------------
An embedding is a fixed-length list of floating-point numbers that encodes the
meaning of a piece of text. Two pieces of text that mean similar things will
have embedding vectors that are close together in vector space, even if they use
completely different words.

That is the core idea behind semantic search: instead of matching keywords, you
match meanings.

Why does the task type matter?
--------------------------------
Gemini's embedding model accepts a `task_type` parameter that tunes the vector
for a specific use case. For retrieval (search), Google recommends:

- `RETRIEVAL_DOCUMENT` when embedding text that will be stored and searched.
- `RETRIEVAL_QUERY`    when embedding the user's search query.

Using the correct task type for each side of the retrieval pair improves search
accuracy. Using the wrong one - or the same one for both - produces a subtly
worse ranking. This class enforces the distinction by exposing two separate
methods rather than one generic embed function.

Why keep a single long-lived client?
--------------------------------------
The `genai.Client` maintains an HTTP connection pool and handles OAuth token
refresh internally. Creating a new client for every embedding call throws away
those connections and, in practice, causes a "Cannot send a request, as the
client has been closed" error when the previous client is garbage-collected while
a retry is in flight. We build the client once in `__post_init__` and reuse it.

The class is a `frozen=True` dataclass, which means its attributes cannot be
reassigned after construction. We use `object.__setattr__` to bypass that
restriction for the private `_client` field, which is not part of the public
interface and is never reassigned after construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from google import genai
from google.genai.types import EmbedContentConfig


@dataclass(frozen=True)
class GeminiEmbedder:
    model: str
    project_id: str
    location: str
    output_dimensionality: int

    # Long-lived Vertex AI client. `init=False` means callers cannot pass a
    # client directly - it is always constructed internally by `__post_init__`.
    _client: genai.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.output_dimensionality <= 0:
            raise ValueError("output_dimensionality must be > 0")
        object.__setattr__(
            self,
            "_client",
            genai.Client(vertexai=True, project=self.project_id, location=self.location),
        )

    @property
    def client(self) -> genai.Client:
        return self._client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Convert a list of product descriptions into embedding vectors for storage.

        We pass all texts in a single API call (a batch). Batching is more
        efficient than one call per document, but there are per-request limits.
        The `index-catalog` command controls batch size with `--batch-size` to
        keep individual requests within those limits.

        The returned list of vectors is in the same order as the input list, so
        you can zip them together with the original documents safely.
        """
        resp = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=self.output_dimensionality,
            ),
        )
        return [e.values for e in resp.embeddings]

    def embed_query(self, text: str) -> list[float]:
        """
        Convert a search query string into an embedding vector for retrieval.

        The resulting vector is passed to OpenSearch's k-NN query, which finds
        the stored document vectors that are closest to it in vector space.
        "Closest" here means most semantically similar, not lexically similar.
        """
        resp = self.client.models.embed_content(
            model=self.model,
            contents=[text],
            config=EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self.output_dimensionality,
            ),
        )
        return resp.embeddings[0].values

