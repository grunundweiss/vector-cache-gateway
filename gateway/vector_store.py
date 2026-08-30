"""Local vector store: embeds documents and answers similarity queries."""

from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

_EPS = 1e-8


class LocalVectorStore:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """Initializes the vector store with a local sentence-transformer encoder."""
        self.model = SentenceTransformer(model_name)
        self.documents: list[dict[str, Any]] = []
        self.embeddings: list[np.ndarray] = []
        self._version = 0

    @property
    def version(self) -> int:
        """Increments on every corpus change, so caches can detect staleness."""
        return self._version

    def encode(self, text: str) -> np.ndarray:
        """Single entry point for embedding text.

        Callers that already hold an embedding should pass it back down rather
        than re-encoding: a forward pass dominates the cost of a query.
        """
        return self.model.encode(text, convert_to_numpy=True)

    def ingest_document(self, doc_id: str, text: str, metadata: dict | None = None) -> None:
        """Embeds a document and adds it to the index."""
        self.documents.append({"id": doc_id, "text": text, "metadata": metadata or {}})
        self.embeddings.append(self.encode(text))
        self._version += 1

    def similarity_search(
        self,
        query: str | None = None,
        top_k: int = 1,
        *,
        embedding: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Ranks indexed documents by cosine similarity.

        Accepts either raw ``query`` text or a precomputed ``embedding``. Passing
        an embedding avoids a redundant forward pass when the caller has already
        encoded the query.
        """
        if embedding is None:
            if query is None:
                raise ValueError("similarity_search requires either query or embedding")
            embedding = self.encode(query)

        if not self.embeddings:
            return []

        matrix = np.vstack(self.embeddings)

        # cosine similarity: (A . B) / (||A|| * ||B||)
        scores = np.dot(matrix, embedding) / (
            np.linalg.norm(matrix, axis=1) * np.linalg.norm(embedding) + _EPS
        )

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {"document": self.documents[idx], "score": float(scores[idx])}
            for idx in top_indices
        ]
