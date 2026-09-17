"""Similarity-gated cache in front of a vector store."""

import logging
from pathlib import Path

from gateway.cache import DEFAULT_MAX_ENTRIES, DEFAULT_THRESHOLD
from gateway.engine import CacheResult, MissOutcome, Prepared, SemanticCacheEngine
from gateway.generator import Generator, RetrievalOnlyGenerator
from gateway.partition import DEFAULT_MAX_NAMESPACES, DEFAULT_NAMESPACE
from gateway.persistence import load_snapshot, save_snapshot
from gateway.vector_store import LocalVectorStore

logger = logging.getLogger(__name__)

NO_MATCH = "No matching compliance documentation found."


class SemanticCacheGateway:
    """Retrieval-augmented answering with a semantic cache in front of it.

    The cache machinery lives in gateway/engine.py, which the chat proxy in
    gateway/middleware.py also uses. What this class adds is the miss path:
    search the vector store, generate from the retrieved passage, cache the
    result.
    """

    def __init__(
        self,
        vector_store: LocalVectorStore,
        similarity_threshold: float = DEFAULT_THRESHOLD,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        generator: Generator | None = None,
        *,
        guard="default",
        resolver=None,
        threshold_policy=None,
        ttl: float | None = None,
        index_backend: str = "auto",
        max_namespaces: int = DEFAULT_MAX_NAMESPACES,
        min_relevance: float | None = None,
        metrics=None,
        **engine_kwargs,
    ):
        """Gateway managing cache state and the similarity gate.

        ``generator`` decides what a cache hit actually saves; see
        gateway/generator.py. Defaults to retrieval-only.

        ``guard`` defaults to the lexical guard from gateway/guard.py, which
        vetoes a hit whose numbers, polarity or acronyms differ from the cached
        question's. Pass ``None`` for the pre-guard behaviour, which the
        threshold sweep measured at a 20% false-hit rate.

        ``min_relevance`` puts a floor under retrieval: below it, the best
        matching document is treated as no match at all rather than answered
        from. Off by default, because no benchmark here chose a value and a
        guessed cutoff silently drops answers.
        """
        self.vector_store = vector_store
        self.threshold = similarity_threshold
        self.generator = generator or RetrievalOnlyGenerator()
        self.min_relevance = min_relevance
        self.engine = SemanticCacheEngine(
            vector_store,
            threshold=similarity_threshold,
            max_entries=max_entries,
            max_namespaces=max_namespaces,
            guard=guard,
            resolver=resolver,
            threshold_policy=threshold_policy,
            ttl=ttl,
            index_backend=index_backend,
            metrics=metrics,
            **engine_kwargs,
        )
        self._corpus_version = vector_store.version

    @property
    def cache(self):
        return self.engine.cache

    @property
    def metrics(self):
        return self.engine.metrics

    def process_query(self, query_text: str) -> tuple[str, str]:
        """Answers a query from cache when a near-enough entry exists.

        Returns the legacy ``(answer, status)`` pair. ``process`` returns the
        same answer plus the similarity, entry id and namespace that a caller
        needs in order to report feedback or debug a hit.
        """
        result = self.process(query_text)
        if result.hit:
            return result.answer, f"CACHE_HIT (Similarity: {result.similarity:.4f})"
        return result.answer, f"CACHE_{result.status}"

    def process(
        self,
        query_text: str,
        *,
        history=(),
        namespace: str = DEFAULT_NAMESPACE,
        ttl: float | None = None,
        skip_cache: bool = False,
    ) -> CacheResult:
        """Answers a query, with conversation history and a tenant namespace.

        ``history`` is the preceding turns; a resolver decides whether they are
        needed (gateway/conversation.py). ``namespace`` partitions the cache:
        two tenants asking the same question never see each other's answers.
        """
        self._invalidate_if_corpus_changed()
        return self.engine.answer(
            query_text,
            self._on_miss,
            history=history,
            namespace=namespace,
            ttl=ttl,
            skip_cache=skip_cache,
        )

    def feedback(self, entry_id: str, helpful: bool = False) -> bool:
        """Evicts the entry a user rejected. Returns whether it was still there."""
        return self.engine.feedback(entry_id, helpful)

    def stats(self) -> dict:
        return self.engine.stats()

    def save_snapshot(self, path: str | Path) -> int:
        """Persists the cache so a restart does not start cold."""
        return save_snapshot(
            self.cache,
            path,
            model=getattr(self.vector_store, "model_name", ""),
            corpus_version=self.vector_store.version,
        )

    def load_snapshot(self, path: str | Path, *, strict: bool = True) -> int:
        """Restores a snapshot, refusing one built by a different encoder or corpus."""
        restored = load_snapshot(
            self.cache,
            path,
            model=getattr(self.vector_store, "model_name", ""),
            corpus_version=self.vector_store.version,
            strict=strict,
        )
        self._corpus_version = self.vector_store.version
        return restored

    def _on_miss(self, prepared: Prepared) -> MissOutcome:
        """Retrieves and generates. Both steps are what a cache hit skips.

        The embedding computed for the cache lookup is reused for retrieval, so
        a miss still costs exactly one forward pass.
        """
        results = self.vector_store.similarity_search(top_k=1, embedding=prepared.embedding)
        if not results:
            return MissOutcome(NO_MATCH, cacheable=False)

        best = results[0]
        if self.min_relevance is not None and best["score"] < self.min_relevance:
            logger.info(
                "best document scored %.4f, below the %.2f relevance floor; not answering",
                best["score"],
                self.min_relevance,
            )
            return MissOutcome(NO_MATCH, cacheable=False)

        context = best["document"]["text"]
        answer = self.generator.generate(prepared.resolved.query, context)
        return MissOutcome(answer)

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

        This handles the corpus going stale. It does not handle an answer going
        stale on its own -- "what is the current promotion" is wrong long before
        any document changes -- which is what per-entry TTLs are for.
        """
        current = self.vector_store.version
        if current == self._corpus_version:
            return

        dropped = len(self.cache)
        logger.info(
            "corpus changed (v%d -> v%d), dropping %d cached answers",
            self._corpus_version,
            current,
            dropped,
        )
        self.engine.clear()
        self.metrics.record_corpus_invalidation(dropped)
        self._corpus_version = current
