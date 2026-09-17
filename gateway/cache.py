"""Similarity cache backed by a matrix of pre-normalized vectors.

The lookup path is four gates in series, cheapest first:

    1. nearest neighbours   from the index backend -- exact scan or ANN
    2. exact rescore        k dot products against the stored vectors
    3. threshold            per-query, from the threshold policy
    4. guard                lexical veto on the string pair (gateway/guard.py)

Only an entry that clears all four is served. Steps 2 and 4 are the ones added
after the threshold sweep showed that step 3 alone cannot separate a paraphrase
from a near-miss: the sweep's worst pair scores 0.9792 and asks about a
different amount of money.
"""

import logging
import threading
import time
import uuid
from typing import NamedTuple

import numpy as np

from gateway.guard import CacheGuard, NullGuard
from gateway.index import VectorIndex, build_index

logger = logging.getLogger(__name__)

_EPS = 1e-8

# sentence-transformers emits float32; storing float64 would double the
# footprint to buy precision the encoder never produced.
_DTYPE = np.float32
DEFAULT_MAX_ENTRIES = 10_000

# Chosen from benchmarks/threshold_sweep.py, not by hand. At 0.85 the sweep
# measures 60% hit rate against 20% false hits -- the widest margin of any
# threshold tested. See "Choosing the threshold" in the README, including why
# 20% is still too high to run unguarded, which is what the guard is for.
DEFAULT_THRESHOLD = 0.85

# How many neighbours to pull before filtering. More than one, because the
# nearest entry may be expired, tombstoned, or vetoed by the guard while the
# second-nearest is a perfectly good answer. Small, because each one costs a
# rescore and a guard check, and past a handful the extra candidates are far
# below the threshold anyway.
DEFAULT_CANDIDATES = 5


def _normalize(vector: np.ndarray) -> np.ndarray:
    return (vector / (np.linalg.norm(vector) + _EPS)).astype(_DTYPE, copy=False)


class CacheHit(NamedTuple):
    """What the cache serves, and enough context to audit or retract it.

    ``answer`` and ``score`` come first so the original two-value unpacking
    still works. ``entry_id`` is what feedback refers to later: it identifies
    this specific cached answer, so a thumbs-down evicts the entry that was
    actually served rather than whatever currently matches the query.
    """

    answer: str
    score: float
    key: str
    entry_id: str
    namespace: str = "default"


class SemanticCache:
    """Maps query embeddings to cached answers.

    Vectors are L2-normalized on insert, so a cosine-similarity lookup reduces
    to a dot product instead of a Python loop that recomputes norms that never
    change. Which dot products get computed is the index backend's decision
    (gateway/index.py); the scoring and the gating happen here either way.

    Capacity is bounded: at ``max_entries`` the least-recently-used entry is
    overwritten in place. Without a bound this is a memory leak with a
    similarity function attached -- a 768-dim vector is ~3 KB per entry before
    Python object overhead, so an unbounded cache grows without limit for as
    long as the process runs.

    Entries removed before eviction -- expired, invalidated, thumbed down --
    become tombstones rather than holes: the slot is marked dead, handed to the
    index to forget, and put at the front of the queue for the next insert.
    Compacting instead would renumber every slot above the hole, and slot
    numbers are the labels an ANN index was built on.

    A lock guards every mutation. Two concurrent misses for the same query will
    still both generate an answer -- the lock makes that redundant, not unsafe --
    but the buffer, the free list and the index cannot be observed mid-update.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        *,
        namespace: str = "default",
        guard: CacheGuard | None = None,
        ttl: float | None = None,
        index_backend: str = "auto",
        candidates: int = DEFAULT_CANDIDATES,
        metrics=None,
        time_fn=time.time,
        index_kwargs: dict | None = None,
    ):
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        if ttl is not None and ttl <= 0:
            raise ValueError("ttl must be positive, or None for no expiry")

        self.threshold = threshold
        self.max_entries = max_entries
        self.namespace = namespace
        self.guard: CacheGuard = guard or NullGuard()
        self.ttl = ttl
        self.index_backend = index_backend
        self.candidates = candidates
        self.metrics = metrics
        self.time_fn = time_fn
        self.index_kwargs = index_kwargs or {}

        self._lock = threading.RLock()
        self._index: VectorIndex | None = None
        self._dim: int | None = None
        self._keys: list[str] = []
        self._answers: list[str] = []
        self._entry_ids: list[str] = []
        self._expires: list[float | None] = []
        self._last_used: list[int] = []
        self._alive: list[bool] = []
        self._by_key: dict[str, int] = {}
        self._by_entry: dict[str, int] = {}
        self._free: list[int] = []
        self._size = 0  # allocated slots, live or tombstoned
        self._tick = 0

    def __len__(self) -> int:
        """Live entries. Tombstoned slots are allocated but not present."""
        return self._size - len(self._free)

    @property
    def backend(self) -> str:
        return self._index.name if self._index is not None else self.index_backend

    @property
    def nbytes(self) -> int:
        """Bytes held by the vector mirror. Excludes keys, answers and the index
        structure's own copy, which `benchmarks/bench_ann.py` measures instead."""
        if self._index is None or self._dim is None:
            return 0
        return self._size * self._dim * np.dtype(_DTYPE).itemsize

    def lookup(
        self,
        embedding: np.ndarray,
        *,
        query: str | None = None,
        threshold: float | None = None,
        now: float | None = None,
    ) -> CacheHit | None:
        """Returns the nearest entry that clears every gate, else None.

        ``query`` is the text the embedding came from. Without it the guard
        cannot run -- it compares strings, not vectors -- so a caller that omits
        it is choosing the pre-guard behaviour. The gateway always passes it.
        """
        with self._lock:
            if len(self) == 0 or self._index is None:
                return None

            limit = self.threshold if threshold is None else threshold
            now = self.time_fn() if now is None else now
            normalized = _normalize(embedding)

            scored = [
                (float(self._index.vector(slot) @ normalized), slot)
                for slot in self._index.search(normalized, self.candidates)
                if self._alive[slot]
            ]
            scored.sort(reverse=True)

            if scored and logger.isEnabledFor(logging.DEBUG):
                best_score, best_slot = scored[0]
                logger.debug(
                    "closest cache match: %r | similarity: %.4f | threshold: %.2f",
                    self._keys[best_slot],
                    best_score,
                    limit,
                )

            for score, slot in scored:
                if score < limit:
                    break
                if self._is_expired(slot, now):
                    logger.debug("cache entry %r expired", self._keys[slot])
                    self._release(slot)
                    if self.metrics is not None:
                        self.metrics.record_expired()
                    continue
                if query is not None:
                    decision = self.guard.approve(query, self._keys[slot], score)
                    if not decision.ok:
                        logger.info(
                            "guard vetoed cache hit at %.4f: %r vs %r (%s)",
                            score,
                            query,
                            self._keys[slot],
                            decision.reason,
                        )
                        if self.metrics is not None:
                            self.metrics.record_guard_veto(decision.reason)
                        continue
                self._touch(slot)
                return CacheHit(
                    self._answers[slot],
                    score,
                    self._keys[slot],
                    self._entry_ids[slot],
                    self.namespace,
                )
            return None

    def insert(
        self,
        key: str,
        embedding: np.ndarray,
        answer: str,
        *,
        ttl: float | None = None,
        now: float | None = None,
        entry_id: str | None = None,
        expires_at: float | None = None,
    ) -> str:
        """Adds or replaces an entry and returns its entry id.

        ``ttl`` overrides the cache-wide default for this entry alone, which is
        how a volatile answer ("today's rate") lives alongside a stable one
        ("the retention period") without giving both the shorter lifetime.

        ``entry_id`` and ``expires_at`` exist for restoring a snapshot, which
        needs the ids and deadlines the entries already had rather than new
        ones. Normal inserts leave both alone.
        """
        with self._lock:
            normalized = _normalize(embedding)
            now = self.time_fn() if now is None else now
            expires = expires_at if expires_at is not None else self._expiry(ttl, now)

            existing = self._by_key.get(key)
            if existing is not None:
                return self._write_slot(existing, key, normalized, answer, expires, entry_id)

            if self._free:
                return self._write_slot(
                    self._free.pop(), key, normalized, answer, expires, entry_id
                )

            if self._size >= self.max_entries:
                # No free slots means every allocated slot is live, so the
                # least-recently-used one is simply the smallest tick.
                victim = int(np.argmin(self._last_used))
                logger.debug(
                    "cache full (%d entries), evicting %r", self._size, self._keys[victim]
                )
                if self.metrics is not None:
                    self.metrics.record_eviction()
                self._forget(victim)
                return self._write_slot(victim, key, normalized, answer, expires, entry_id)

            self._grow(normalized.shape[0])
            return self._write_slot(self._size - 1, key, normalized, answer, expires, entry_id)

    def invalidate(self, entry_id: str, *, feedback: bool = False) -> bool:
        """Drops one entry by id. Returns whether it was there.

        This is what negative feedback calls. It takes an entry id rather than a
        query because by the time a user reacts, the query that produced the
        answer may no longer be the nearest neighbour of anything -- and
        re-running the search to find what to delete could delete something
        else.
        """
        with self._lock:
            slot = self._by_entry.get(entry_id)
            if slot is None or not self._alive[slot]:
                return False
            logger.info("invalidating cache entry %r (%s)", self._keys[slot], entry_id)
            self._release(slot)
            if self.metrics is not None:
                self.metrics.record_invalidation(feedback=feedback)
            return True

    def invalidate_key(self, key: str) -> bool:
        """Drops the entry whose cache key is exactly ``key``."""
        with self._lock:
            slot = self._by_key.get(key)
            if slot is None or not self._alive[slot]:
                return False
            self._release(slot)
            if self.metrics is not None:
                self.metrics.record_invalidation()
            return True

    def purge_expired(self, now: float | None = None) -> int:
        """Removes every expired entry. Returns how many.

        Lookups already drop expired entries they happen to touch; this is for
        the ones nobody queries, which would otherwise hold a slot until they
        were evicted for being old -- the same outcome, arbitrarily later.
        """
        with self._lock:
            now = self.time_fn() if now is None else now
            purged = 0
            for slot in range(self._size):
                if self._alive[slot] and self._is_expired(slot, now):
                    self._release(slot)
                    purged += 1
            if purged and self.metrics is not None:
                self.metrics.record_expired(purged)
            return purged

    def clear(self) -> None:
        with self._lock:
            if self._index is not None:
                self._index.clear()
            self._keys.clear()
            self._answers.clear()
            self._entry_ids.clear()
            self._expires.clear()
            self._last_used.clear()
            self._alive.clear()
            self._by_key.clear()
            self._by_entry.clear()
            self._free.clear()
            self._size = 0

    def entries(self) -> list[dict]:
        """Live entries as plain data, for snapshots. Vectors are copies."""
        with self._lock:
            return [
                {
                    "key": self._keys[slot],
                    "answer": self._answers[slot],
                    "entry_id": self._entry_ids[slot],
                    "expires_at": self._expires[slot],
                    "vector": np.array(self._index.vector(slot), copy=True),  # type: ignore[union-attr]
                }
                for slot in range(self._size)
                if self._alive[slot]
            ]

    def describe(self) -> dict:
        with self._lock:
            return {
                "namespace": self.namespace,
                "entries": len(self),
                "allocated_slots": self._size,
                "tombstones": len(self._free),
                "max_entries": self.max_entries,
                "backend": self.backend,
                "threshold": self.threshold,
                "ttl": self.ttl,
                "guard": getattr(self.guard, "name", type(self.guard).__name__),
                "vector_bytes": self.nbytes,
            }

    # -- internals ---------------------------------------------------------

    def _expiry(self, ttl: float | None, now: float) -> float | None:
        effective = self.ttl if ttl is None else ttl
        return None if effective is None else now + effective

    def _is_expired(self, slot: int, now: float) -> bool:
        expires = self._expires[slot]
        return expires is not None and expires <= now

    def _grow(self, dim: int) -> None:
        if self._index is None:
            self._dim = dim
            self._index = build_index(dim, self.max_entries, self.index_backend,
                                      **self.index_kwargs)
        elif dim != self._dim:
            raise ValueError(f"expected {self._dim}-dim vectors, got {dim}")
        self._keys.append("")
        self._answers.append("")
        self._entry_ids.append("")
        self._expires.append(None)
        self._last_used.append(0)
        self._alive.append(False)
        self._size += 1

    def _write_slot(
        self,
        slot: int,
        key: str,
        normalized: np.ndarray,
        answer: str,
        expires: float | None,
        entry_id: str | None = None,
    ) -> str:
        # Every path into here goes through _grow first, so the index exists and
        # the slot is allocated.
        assert self._index is not None
        old_entry = self._entry_ids[slot]
        if old_entry:
            self._by_entry.pop(old_entry, None)

        entry_id = entry_id or uuid.uuid4().hex[:16]
        self._index.upsert(slot, normalized)
        self._keys[slot] = key
        self._answers[slot] = answer
        self._entry_ids[slot] = entry_id
        self._expires[slot] = expires
        self._alive[slot] = True
        self._by_key[key] = slot
        self._by_entry[entry_id] = slot
        self._touch(slot)
        return entry_id

    def _forget(self, slot: int) -> None:
        """Detaches a slot's identity without freeing it -- the caller reuses it."""
        if self._keys[slot]:
            self._by_key.pop(self._keys[slot], None)
        if self._entry_ids[slot]:
            self._by_entry.pop(self._entry_ids[slot], None)

    def _release(self, slot: int) -> None:
        """Tombstones a slot: dead here, forgotten by the index, reusable next."""
        self._forget(slot)
        self._alive[slot] = False
        self._keys[slot] = ""
        self._answers[slot] = ""
        self._entry_ids[slot] = ""
        self._expires[slot] = None
        self._last_used[slot] = 0
        if self._index is not None:
            self._index.remove(slot)
        self._free.append(slot)

    def _touch(self, slot: int) -> None:
        self._tick += 1
        self._last_used[slot] = self._tick
