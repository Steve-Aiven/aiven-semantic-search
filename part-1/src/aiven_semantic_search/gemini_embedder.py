"""
Gemini embedding wrapper (Vertex AI) - Part 1.

We keep the embedding logic in one place so:
- indexing uses RETRIEVAL_DOCUMENT
- searching uses RETRIEVAL_QUERY

Those task types are documented in the Vertex AI text embeddings reference.
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
        resp = self.client.models.embed_content(
            model=self.model,
            contents=[text],
            config=EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self.output_dimensionality,
            ),
        )
        return resp.embeddings[0].values

