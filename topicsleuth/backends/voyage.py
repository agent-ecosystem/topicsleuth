"""Voyage AI embeddings. ``pip install "topicsleuth[voyage]"``."""
from __future__ import annotations


class VoyageEmbedder:
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
        response = self._client.embed(texts=texts, model=self.model, input_type=input_type)
        return [list(e) for e in response.embeddings]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "document")

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "query")
