"""Per-tenant cache partitions.

A semantic cache that serves more than one user has a data-leak shape that a
key-value cache does not. An exact-match cache can only return A's answer to B
if B sends A's exact question. A similarity cache returns A's answer to B when
the questions are merely *close* -- and "What is my account balance?" is close
to "What is my account balance?" no matter who is asking. The answer contains
A's balance.

So partitions are enforced by construction rather than by a filter: each
namespace gets its own `SemanticCache`, with its own vectors and its own index.
There is no code path that compares a vector in one namespace against a vector
in another, because they are never in the same matrix. A filter applied after
the search would be one forgotten predicate away from the failure above, and
ANN backends make post-filtering worse -- the neighbours come back already
chosen, so a filtered-out result is a hit you silently lost rather than one you
never found.

The cost is duplication: the same question asked by a thousand tenants is cached
a thousand times. That is the correct trade for answers derived from per-tenant
data, and the wrong one for a shared, public corpus. A deployment whose answers
depend only on public documents should use one namespace and say so, out loud,
in the code that chooses it.
"""

import logging
import threading
import time

import numpy as np

from gateway.cache import (
    DEFAULT_CANDIDATES,
    DEFAULT_MAX_ENTRIES,
    DEFAULT_THRESHOLD,
    CacheHit,
    SemanticCache,
)
from gateway.guard import CacheGuard

logger = logging.getLogger(__name__)

DEFAULT_NAMESPACE = "default"

# Bounded like everything else here: a namespace per tenant is a memory leak the
# moment tenants are created faster than they are retired.
DEFAULT_MAX_NAMESPACES = 128


class PartitionedCache:
    """A `SemanticCache` per namespace, bounded in both dimensions.

    Worst-case vector memory is ``max_namespaces * max_entries * dim * 4``
    bytes -- 128 namespaces of 10k 768-dim entries is 3.75 GB, which is a number
    worth reading before accepting the defaults. ``describe()`` reports what is
    actually allocated, which for most deployments is far less, because
    namespaces are created on first write and hold only what they were asked to
    hold.

    Whole namespaces are evicted least-recently-used when there are too many.
    That drops a tenant's entire cache at once, which is blunt, and correct: a
    tenant nobody has queried in a while has the coldest entries in the process
    by definition.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        *,
        max_namespaces: int = DEFAULT_MAX_NAMESPACES,
        guard: CacheGuard | None = None,
        ttl: float | None = None,
        index_backend: str = "auto",
        candidates: int = DEFAULT_CANDIDATES,
        metrics=None,
        time_fn=time.time,
        index_kwargs: dict | None = None,
    ):
        if max_namespaces < 1:
            raise ValueError("max_namespaces must be at least 1")

        self.threshold = threshold
        self.max_entries = max_entries
        self.max_namespaces = max_namespaces
        self.guard = guard
        self.ttl = ttl
        self.index_backend = index_backend
        self.candidates = candidates
        self.metrics = metrics
        self.time_fn = time_fn
        self.index_kwargs = index_kwargs

        self._lock = threading.RLock()
        self._partitions: dict[str, SemanticCache] = {}
        self._used: dict[str, int] = {}
        self._tick = 0

    def __len__(self) -> int:
        with self._lock:
            return sum(len(cache) for cache in self._partitions.values())

    def namespaces(self) -> list[str]:
        with self._lock:
            return list(self._partitions)

    def partition(self, namespace: str, *, create: bool = True) -> SemanticCache | None:
        """Returns the cache for ``namespace``, creating it on demand."""
        with self._lock:
            cache = self._partitions.get(namespace)
            if cache is None:
                if not create:
                    return None
                self._evict_namespace_if_full()
                cache = SemanticCache(
                    threshold=self.threshold,
                    max_entries=self.max_entries,
                    namespace=namespace,
                    guard=self.guard,
                    ttl=self.ttl,
                    index_backend=self.index_backend,
                    candidates=self.candidates,
                    metrics=self.metrics,
                    time_fn=self.time_fn,
                    index_kwargs=self.index_kwargs,
                )
                self._partitions[namespace] = cache
                logger.debug("created cache partition %r", namespace)
            self._tick += 1
            self._used[namespace] = self._tick
            return cache

    def lookup(
        self,
        embedding: np.ndarray,
        *,
        namespace: str = DEFAULT_NAMESPACE,
        query: str | None = None,
        threshold: float | None = None,
        now: float | None = None,
    ) -> CacheHit | None:
        cache = self.partition(namespace, create=False)
        if cache is None:
            return None
        with self._lock:
            self._tick += 1
            self._used[namespace] = self._tick
        return cache.lookup(embedding, query=query, threshold=threshold, now=now)

    def insert(
        self,
        key: str,
        embedding: np.ndarray,
        answer: str,
        *,
        namespace: str = DEFAULT_NAMESPACE,
        ttl: float | None = None,
        now: float | None = None,
    ) -> str:
        cache = self.partition(namespace)
        assert cache is not None
        return cache.insert(key, embedding, answer, ttl=ttl, now=now)

    def invalidate(self, entry_id: str, *, feedback: bool = False) -> bool:
        """Drops an entry by id, wherever it lives.

        Scans partitions rather than keeping an id-to-namespace map, because
        that map would have to be kept in step with eviction and expiry in every
        partition -- a second source of truth about which entries exist, which
        is the kind of thing that gets out of step and then deletes the wrong
        row. A few dozen dict lookups on a feedback event is not a cost worth
        optimizing.
        """
        with self._lock:
            partitions = list(self._partitions.values())
        for cache in partitions:
            if cache.invalidate(entry_id, feedback=feedback):
                return True
        return False

    def invalidate_key(self, key: str, *, namespace: str = DEFAULT_NAMESPACE) -> bool:
        cache = self.partition(namespace, create=False)
        return False if cache is None else cache.invalidate_key(key)

    def purge_expired(self, now: float | None = None) -> int:
        with self._lock:
            partitions = list(self._partitions.values())
        return sum(cache.purge_expired(now) for cache in partitions)

    def clear(self, namespace: str | None = None) -> None:
        """Clears one namespace, or drops every partition when given none."""
        with self._lock:
            if namespace is None:
                for cache in self._partitions.values():
                    cache.clear()
                self._partitions.clear()
                self._used.clear()
                return
            dropped = self._partitions.pop(namespace, None)
            self._used.pop(namespace, None)
        if dropped is not None:
            dropped.clear()

    def describe(self) -> dict:
        with self._lock:
            partitions = dict(self._partitions)
        return {
            "namespaces": len(partitions),
            "max_namespaces": self.max_namespaces,
            "entries": sum(len(cache) for cache in partitions.values()),
            "vector_bytes": sum(cache.nbytes for cache in partitions.values()),
            "partitions": [cache.describe() for cache in partitions.values()],
        }

    def _evict_namespace_if_full(self) -> None:
        if len(self._partitions) < self.max_namespaces:
            return
        victim = min(self._used, key=lambda ns: self._used[ns])
        dropped = len(self._partitions[victim])
        logger.info("namespace limit reached, dropping %r (%d entries)", victim, dropped)
        self._partitions.pop(victim).clear()
        self._used.pop(victim, None)
        if self.metrics is not None:
            self.metrics.record_invalidation(dropped)
