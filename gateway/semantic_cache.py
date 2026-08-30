"""Similarity-gated cache in front of a vector store."""

import logging

import numpy as np

from gateway.cache import DEFAULT_MAX_ENTRIES, DEFAULT_THRESHOLD, SemanticCache
from gateway.generator import Generator, RetrievalOnlyGenerator
from gateway.vector_store import LocalVectorStore

logger = logging.getLogger(__name__)

NO_MATCH = "No matching compliance documentation found."


class SemanticCacheGateway:
    def __init__(
        self,
        vector_store: LocalVectorStore,
        similarity_threshold: float = DEFAULT_THRESHOLD,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        generator: Generator | None = None,
    ):
        """Gateway managing cache state and the similarity gate.

        ``generator`` decides what a cache hit actually saves; see
        gateway/generator.py. Defaults to retrieval-only.
        """
        self.vector_store = vector_store
        self.threshold = similarity_threshold
        self.generator = generator or RetrievalOnlyGenerator()
        self.cache = SemanticCache(threshold=similarity_threshold, max_entries=max_entries)
        self._corpus_version = vector_store.version

    def process_query(self, query_text: str) -> tuple[str, str]:
        """Answers a query from cache when a near-enough entry exists.

        The query is embedded exactly once and the vector is threaded through
        every downstream step.
        """
        self._invalidate_if_corpus_changed()
        embedding = self.vector_store.encode(query_text)

        hit = self.cache.lookup(embedding)
        if hit is not None:
            answer, score = hit
            return answer, f"CACHE_HIT (Similarity: {score:.4f})"

        return self._fallback_and_cache(query_text, embedding)

    def _fallback_and_cache(self, query_text: str, embedding: np.ndarray) -> tuple[str, str]:
        """Retrieves and generates on a miss, then records the answer.

        Both steps are what a cache hit skips.
        """
        results = self.vector_store.similarity_search(top_k=1, embedding=embedding)
        if not results:
            return NO_MATCH, "CACHE_MISS"

        context = results[0]["document"]["text"]
        answer = self.generator.generate(query_text, context)

        self.cache.insert(query_text, embedding, answer)
        return answer, "CACHE_MISS"

    def _invalidate_if_corpus_changed(self) -> None:
        """Drops the whole cache when the corpus changes.

        Cached answers are derived from documents in the store, so ingesting a
        document that better answers an already-cached query would otherwise
        keep serving the stale answer forever.

        Clearing everything is deliberately blunt. Deciding which entries a new
        document actually affects means re-running the search for every cached
        query, which costs more than repopulating on demand. The trade is a
        cold cache after each ingest, which is the right default for a
        write-rarely corpus; a write-heavy one would want per-entry
        invalidation instead.
        """
        current = self.vector_store.version
        if current == self._corpus_version:
            return

        logger.info(
            "corpus changed (v%d -> v%d), dropping %d cached answers",
            self._corpus_version,
            current,
            len(self.cache),
        )
        self.cache.clear()
        self._corpus_version = current
