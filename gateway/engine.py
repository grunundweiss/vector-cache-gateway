"""The caching machinery, independent of what produces an answer on a miss.

Two things in this repo need a semantic cache: the RAG gateway in
gateway/semantic_cache.py, which answers from a vector store, and the chat proxy
in gateway/middleware.py, which answers from an LLM endpoint. They differ only
in what happens on a miss. Everything else -- resolving the turn, embedding it
once, choosing a threshold, searching the right partition, running the guard,
recording the metrics, storing the result -- is identical, and lives here.

The query is embedded exactly once per request and the vector is threaded
through every downstream step, which is the invariant tests/test_encode_budget.py
exists to defend. On a 40 ms embedding pass, a second forward call costs more
than everything else in this file put together.
"""

import logging
import time
from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np

from gateway.cache import DEFAULT_CANDIDATES, DEFAULT_MAX_ENTRIES, DEFAULT_THRESHOLD, CacheHit
from gateway.conversation import ResolvedQuery, resolve_resolver
from gateway.guard import default_guard
from gateway.metrics import CacheMetrics
from gateway.partition import DEFAULT_MAX_NAMESPACES, DEFAULT_NAMESPACE, PartitionedCache
from gateway.policy import resolve_policy

logger = logging.getLogger(__name__)


@runtime_checkable
class Encoder(Protocol):
    """Anything that turns text into a vector. `LocalVectorStore` is one."""

    def encode(self, text: str) -> np.ndarray: ...


class MissOutcome(NamedTuple):
    """What a miss handler produced, and whether it may be cached.

    ``cacheable=False`` is for answers that are correct now and wrong later, or
    that were never really answers: a "no matching document" placeholder, a
    response that came back empty, anything derived from live state.
    """

    answer: str
    cacheable: bool = True
    ttl: float | None = None


class Prepared(NamedTuple):
    """Everything a lookup needs, computed once per request."""

    resolved: ResolvedQuery
    embedding: np.ndarray
    threshold: float
    namespace: str
    started: float


class CacheResult(NamedTuple):
    """The outcome of one request through the cache."""

    answer: str
    status: str  # HIT | MISS | BYPASS
    similarity: float | None = None
    entry_id: str | None = None
    namespace: str = DEFAULT_NAMESPACE
    cache_key: str = ""
    latency_ms: float = 0.0
    resolved: ResolvedQuery | None = None

    @property
    def hit(self) -> bool:
        return self.status == "HIT"


class SemanticCacheEngine:
    """Encoder + partitioned cache + resolver + threshold policy + guard + metrics.

    Every knob has a default that matches what the benchmarks measured, except
    the two that are policy rather than measurement and say so where they are
    defined: ``threshold_policy="intent"`` and the per-entry ``ttl``.
    """

    def __init__(
        self,
        encoder: Encoder,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_namespaces: int = DEFAULT_MAX_NAMESPACES,
        guard="default",
        resolver=None,
        threshold_policy=None,
        ttl: float | None = None,
        index_backend: str = "auto",
        candidates: int = DEFAULT_CANDIDATES,
        metrics: CacheMetrics | None = None,
        time_fn=time.time,
        rephrase_feedback: bool = False,
        rephrase_window_s: float = 30.0,
        rephrase_similarity: float = 0.9,
        index_kwargs: dict | None = None,
    ):
        self.encoder = encoder
        self.resolver = resolve_resolver(resolver)
        self.policy = resolve_policy(threshold_policy, threshold)
        self.metrics = metrics if metrics is not None else CacheMetrics()
        self.time_fn = time_fn
        self.rephrase_feedback = rephrase_feedback
        self.rephrase_window_s = rephrase_window_s
        self.rephrase_similarity = rephrase_similarity

        self.cache = PartitionedCache(
            threshold=threshold,
            max_entries=max_entries,
            max_namespaces=max_namespaces,
            guard=default_guard() if guard == "default" else guard,
            ttl=ttl,
            index_backend=index_backend,
            candidates=candidates,
            metrics=self.metrics,
            time_fn=time_fn,
            index_kwargs=index_kwargs,
        )
        self._last_served: dict[str, tuple[str, np.ndarray, float]] = {}

    # -- the three steps, separately, for callers that stream ---------------

    def prepare(
        self, query: str, *, history=(), namespace: str = DEFAULT_NAMESPACE
    ) -> Prepared:
        """Resolves the turn and embeds it. The only forward pass in a request."""
        resolved = self.resolver.resolve(query, history)
        embedding = self.encoder.encode(resolved.text)
        threshold = self.policy.threshold_for(resolved.text)
        return Prepared(resolved, embedding, threshold, namespace, time.perf_counter())

    def lookup(self, prepared: Prepared) -> CacheHit | None:
        """Searches the namespace's partition, guard and all."""
        self._maybe_rephrase_feedback(prepared)
        hit = self.cache.lookup(
            prepared.embedding,
            namespace=prepared.namespace,
            query=prepared.resolved.cache_key,
            threshold=prepared.threshold,
        )
        if hit is not None:
            self._remember(prepared, hit.entry_id)
        return hit

    def store(self, prepared: Prepared, answer: str, *, ttl: float | None = None) -> str:
        entry_id = self.cache.insert(
            prepared.resolved.cache_key,
            prepared.embedding,
            answer,
            namespace=prepared.namespace,
            ttl=ttl,
        )
        self._remember(prepared, entry_id)
        return entry_id

    # -- the whole request, for callers that do not ------------------------

    def answer(
        self,
        query: str,
        on_miss,
        *,
        history=(),
        namespace: str = DEFAULT_NAMESPACE,
        ttl: float | None = None,
        skip_cache: bool = False,
    ) -> CacheResult:
        """Serves from cache, or calls ``on_miss(prepared)`` and stores the result.

        ``on_miss`` may return a plain string or a `MissOutcome`. Returning
        ``cacheable=False`` produces an answer that is served and forgotten,
        which is the right handling for anything derived from live state.

        ``skip_cache=True`` neither reads nor writes the cache and is counted as
        a bypass rather than a miss.
        """
        if skip_cache:
            # Skips the cache, not the embedding: the miss handler may need the
            # vector -- the RAG gateway retrieves with it -- and handing it an
            # empty array to save a forward pass it is about to make anyway
            # trades a real crash for an imaginary saving.
            prepared = self.prepare(query, history=history, namespace=namespace)
            outcome = _as_outcome(on_miss(prepared))
            self.metrics.record_bypass("skip_cache")
            return CacheResult(
                outcome.answer,
                "BYPASS",
                namespace=namespace,
                cache_key=prepared.resolved.cache_key,
                latency_ms=(time.perf_counter() - prepared.started) * 1000,
                resolved=prepared.resolved,
            )

        prepared = self.prepare(query, history=history, namespace=namespace)
        hit = self.lookup(prepared)
        if hit is not None:
            latency_ms = (time.perf_counter() - prepared.started) * 1000
            self.metrics.record_hit(hit.score, latency_ms)
            return CacheResult(
                hit.answer,
                "HIT",
                similarity=hit.score,
                entry_id=hit.entry_id,
                namespace=hit.namespace,
                cache_key=hit.key,
                latency_ms=latency_ms,
                resolved=prepared.resolved,
            )

        outcome = _as_outcome(on_miss(prepared))
        entry_id = None
        if outcome.cacheable:
            entry_id = self.store(prepared, outcome.answer, ttl=outcome.ttl or ttl)
        latency_ms = (time.perf_counter() - prepared.started) * 1000
        self.metrics.record_miss(latency_ms)
        return CacheResult(
            outcome.answer,
            "MISS",
            entry_id=entry_id,
            namespace=namespace,
            cache_key=prepared.resolved.cache_key,
            latency_ms=latency_ms,
            resolved=prepared.resolved,
        )

    # -- feedback ----------------------------------------------------------

    def feedback(self, entry_id: str, helpful: bool) -> bool:
        """Records a user's verdict on a served answer.

        Positive feedback is deliberately a no-op beyond the log line. A cached
        entry that was useful once is already being kept by the LRU; promoting
        it further would mean trusting a signal that arrives from whoever is
        holding the other end of the connection. Negative feedback evicts,
        because the cost of being wrong about *that* is one regenerated answer.
        """
        if helpful:
            logger.debug("positive feedback for %s (no action)", entry_id)
            return False
        return self.cache.invalidate(entry_id, feedback=True)

    def clear(self, namespace: str | None = None) -> None:
        self.cache.clear(namespace)
        self._last_served.clear()

    def purge_expired(self) -> int:
        return self.cache.purge_expired()

    def stats(self) -> dict:
        return {"metrics": self.metrics.snapshot(), "cache": self.cache.describe()}

    # -- internals ---------------------------------------------------------

    def _remember(self, prepared: Prepared, entry_id: str) -> None:
        if self.rephrase_feedback:
            self._last_served[prepared.namespace] = (
                entry_id,
                prepared.embedding,
                self.time_fn(),
            )

    def _maybe_rephrase_feedback(self, prepared: Prepared) -> None:
        """Treats an immediate rephrase as a thumbs-down on what was just served.

        A user who rewords the same question seconds after getting an answer is
        telling you the answer missed. It is a real signal and a noisy one: with
        a conversational resolver, a legitimate follow-up shares most of its
        text with the previous turn and looks exactly like a rephrase. That is
        why this is off unless asked for, and why the window is short.
        """
        if not self.rephrase_feedback:
            return
        previous = self._last_served.get(prepared.namespace)
        if previous is None:
            return
        entry_id, embedding, served_at = previous
        if self.time_fn() - served_at > self.rephrase_window_s:
            self._last_served.pop(prepared.namespace, None)
            return
        left = embedding / (np.linalg.norm(embedding) + 1e-8)
        right = prepared.embedding / (np.linalg.norm(prepared.embedding) + 1e-8)
        similarity = float(left @ right)
        if similarity >= self.rephrase_similarity and similarity < 0.9999:
            logger.info(
                "treating rephrase at %.4f as negative feedback for %s", similarity, entry_id
            )
            self.cache.invalidate(entry_id, feedback=True)
            self._last_served.pop(prepared.namespace, None)


def _as_outcome(value) -> MissOutcome:
    return value if isinstance(value, MissOutcome) else MissOutcome(str(value))
