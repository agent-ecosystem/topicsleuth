"""Voyage AI embeddings. ``pip install "topicsleuth[voyage]"``."""
from __future__ import annotations

import math


class VoyageEmbedder:
    """``dimensions`` is requested from the API (``output_dimension``) and
    every returned vector is checked against it, so a store whose index was
    built for one dimensionality never receives another."""

    def __init__(self, model: str = "voyage-4", dimensions: int = 1024, client=None, api_key: str | None = None):
        try:
            import voyageai
        except ImportError as e:  # pragma: no cover
            raise ImportError('VoyageEmbedder needs the voyageai package: pip install "topicsleuth[voyage]"') from e
        self.model = model
        self.dimensions = dimensions
        self._client = client or voyageai.Client(api_key=api_key)

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embed(
            texts=texts, model=self.model, input_type=input_type, output_dimension=self.dimensions,
        )
        vectors = [list(e) for e in response.embeddings]
        if len(vectors) != len(texts):
            raise ValueError(f"Voyage returned {len(vectors)} embeddings for {len(texts)} texts")
        for v in vectors:
            if len(v) != self.dimensions:
                raise ValueError(f"Voyage returned a {len(v)}-dimensional vector; this embedder is {self.dimensions}")
            if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in v):
                raise ValueError("Voyage returned a vector with a non-finite value")
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "document")

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "query")
