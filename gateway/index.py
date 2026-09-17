"""Vector index backends: an exact scan, and ANN for when the cache outgrows it.

The cache stores L2-normalized float32 rows, so cosine similarity is a plain dot
product and "nearest neighbour" means "largest inner product". The only thing
that differs between backends is how many of those products get computed:

``exact``  every row on every lookup. O(n), recall 1.0 by construction. This is
           what the repo did everywhere before, and it is still the default at
           the default cache size -- 10k entries scan in ~5.7 ms, comfortably
           under the ~40 ms embedding pass that precedes it.
``hnsw``   hnswlib's navigable small-world graph. Sub-linear, approximate, and
           supports updating a label in place, which is what eviction needs.
``faiss``  faiss IVF-Flat. Sub-linear once trained, approximate, and needs a
           training pass over a sample of the data before it can be queried --
           until then this backend falls back to an exact scan.

Every backend keeps its own float32 copy of the vectors, so the cache can
rescore the handful of candidates an ANN search returns *exactly* before
applying the threshold. That costs k dot products and buys an important
property: the threshold means the same thing on every backend. Switching to ANN
changes which entries are *found*, never what score they are judged by.

The cost of that copy is memory: an ANN backend holds the vectors twice, once in
the numpy mirror and once in its own structure. `benchmarks/bench_ann.py`
measures both the recall that approximation costs and the memory it adds.
"""

from typing import Any, Protocol, runtime_checkable

import numpy as np

_DTYPE = np.float32
_INITIAL_CAPACITY = 64

# Above this many entries `build_index(..., "auto")` prefers an ANN backend if
# one is installed. It is the bound the scaling benchmark calls the limit of a
# defensible full scan, so the default 10k cache stays exact and only a cache
# explicitly configured larger trades recall for speed.
ANN_AUTO_ABOVE = 10_000


@runtime_checkable
class VectorIndex(Protocol):
    """Finds candidate slots for a query vector.

    Slots are dense integers owned by the cache: it decides where an entry
    lives, the index only has to answer where to look. ``search`` returns
    candidates, not decisions -- the caller rescores them against the stored
    vectors and applies the threshold itself.
    """

    name: str

    def upsert(self, slot: int, vector: np.ndarray) -> None:
        """Stores ``vector`` at ``slot``, replacing whatever was there."""
        ...

    def remove(self, slot: int) -> None:
        """Drops ``slot`` so it stops appearing in search results."""
        ...

    def search(self, vector: np.ndarray, k: int) -> list[int]:
        """Returns up to ``k`` candidate slots, best first."""
        ...

    def vector(self, slot: int) -> np.ndarray:
        """Returns the stored vector for ``slot``."""
        ...

    def clear(self) -> None: ...


class _Mirror:
    """The float32 copy every backend keeps, grown geometrically."""

    def __init__(self, dim: int, capacity: int):
        self.dim = dim
        self.capacity = capacity
        self._rows = np.zeros((min(_INITIAL_CAPACITY, capacity), dim), dtype=_DTYPE)
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def write(self, slot: int, vector: np.ndarray) -> None:
        if slot >= self._rows.shape[0]:
            grown_rows = min(max(self._rows.shape[0] * 2, slot + 1), self.capacity)
            grown = np.zeros((grown_rows, self.dim), dtype=_DTYPE)
            grown[: self._size] = self._rows[: self._size]
            self._rows = grown
        self._rows[slot] = vector
        self._size = max(self._size, slot + 1)

    def zero(self, slot: int) -> None:
        if slot < self._rows.shape[0]:
            self._rows[slot] = 0.0

    @property
    def matrix(self) -> np.ndarray:
        return self._rows[: self._size]

    def row(self, slot: int) -> np.ndarray:
        return self._rows[slot]

    def clear(self) -> None:
        self._rows = np.zeros((min(_INITIAL_CAPACITY, self.capacity), self.dim), dtype=_DTYPE)
        self._size = 0


def _top_k(scores: np.ndarray, k: int) -> list[int]:
    """Indices of the k largest scores, descending. argpartition, not a sort."""
    if k >= scores.shape[0]:
        return [int(i) for i in np.argsort(scores)[::-1]]
    candidates = np.argpartition(scores, -k)[-k:]
    return [int(i) for i in candidates[np.argsort(scores[candidates])[::-1]]]


class ExactIndex:
    """Full cosine scan over every stored vector.

    One matmul, no approximation, no tuning parameters, and no second copy of
    the data in a foreign structure. It is the right answer until the scan cost
    approaches the embedding pass that precedes it -- see the scaling table in
    `results/benchmarks.md` for where that is.
    """

    name = "exact"

    def __init__(self, dim: int, capacity: int):
        self._mirror = _Mirror(dim, capacity)

    def upsert(self, slot: int, vector: np.ndarray) -> None:
        self._mirror.write(slot, vector)

    def remove(self, slot: int) -> None:
        # A zero row scores 0.0 against every query, so a removed slot can never
        # clear a positive threshold or crowd a live entry out of the top k.
        self._mirror.zero(slot)

    def search(self, vector: np.ndarray, k: int) -> list[int]:
        matrix = self._mirror.matrix
        if matrix.shape[0] == 0:
            return []
        return _top_k(matrix @ vector, k)

    def vector(self, slot: int) -> np.ndarray:
        return self._mirror.row(slot)

    def clear(self) -> None:
        self._mirror.clear()


class HnswIndex:
    """hnswlib-backed approximate index.

    Chosen over faiss for the default ANN backend because it updates a label in
    place, which is exactly what LRU eviction does to a slot. ``ef_search``
    trades recall for latency at query time and can be changed after the fact;
    ``m`` and ``ef_construction`` are fixed when the graph is built.
    """

    name = "hnsw"

    def __init__(
        self,
        dim: int,
        capacity: int,
        *,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
    ):
        import hnswlib

        self._hnswlib = hnswlib
        self._dim = dim
        self._capacity = capacity
        self._m = m
        self._ef_construction = ef_construction
        self._ef_search = ef_search
        self._mirror = _Mirror(dim, capacity)
        self._live: set[int] = set()
        self._deleted: set[int] = set()
        self._index = self._new_index(min(_INITIAL_CAPACITY, capacity))

    def _new_index(self, size: int):
        index = self._hnswlib.Index(space="cosine", dim=self._dim)
        index.init_index(
            max_elements=max(size, 1), M=self._m, ef_construction=self._ef_construction
        )
        index.set_ef(self._ef_search)
        return index

    @property
    def ef_search(self) -> int:
        return self._ef_search

    @ef_search.setter
    def ef_search(self, value: int) -> None:
        self._ef_search = value
        self._index.set_ef(value)

    def upsert(self, slot: int, vector: np.ndarray) -> None:
        self._mirror.write(slot, vector)
        if slot >= self._index.get_max_elements():
            grown = min(max(self._index.get_max_elements() * 2, slot + 1), self._capacity)
            self._index.resize_index(max(grown, slot + 1))
        if slot in self._deleted:
            self._index.unmark_deleted(slot)
            self._deleted.discard(slot)
        # hnswlib updates an existing label in place, so this is both insert and
        # overwrite -- the two things eviction needs it to be.
        self._index.add_items(vector.reshape(1, -1), np.array([slot], dtype=np.int64))
        self._live.add(slot)

    def remove(self, slot: int) -> None:
        if slot not in self._live:
            return
        self._index.mark_deleted(slot)
        self._deleted.add(slot)
        self._live.discard(slot)
        self._mirror.zero(slot)

    def search(self, vector: np.ndarray, k: int) -> list[int]:
        if not self._live:
            return []
        k = min(k, len(self._live))
        # ef must be at least k or hnswlib refuses the query.
        if self._ef_search < k:
            self._index.set_ef(k)
        labels, _ = self._index.knn_query(vector.reshape(1, -1), k=k)
        return [int(label) for label in labels[0]]

    def vector(self, slot: int) -> np.ndarray:
        return self._mirror.row(slot)

    def clear(self) -> None:
        self._mirror.clear()
        self._live.clear()
        self._deleted.clear()
        self._index = self._new_index(min(_INITIAL_CAPACITY, self._capacity))


class FaissIvfIndex:
    """faiss IVF-Flat, with an exact scan until there is enough data to train it.

    IVF partitions the vectors into ``nlist`` cells and searches ``nprobe`` of
    them, so lookup touches a fraction of the data. The catch is the training
    pass: a coarse quantizer has to be fitted before the index can be queried at
    all, and fitting it on a handful of vectors produces a bad partition. This
    backend therefore stays exact until ``train_after`` vectors have arrived,
    trains once, and adds everything it was holding. A cache that never gets
    that big never leaves the exact path, which is the correct outcome rather
    than a fallback.
    """

    name = "faiss-ivf"

    def __init__(
        self,
        dim: int,
        capacity: int,
        *,
        nlist: int | None = None,
        nprobe: int = 8,
        train_after: int | None = None,
    ):
        import faiss

        self._faiss = faiss
        self._dim = dim
        self._mirror = _Mirror(dim, capacity)
        self._live: set[int] = set()
        self._nlist = nlist or int(np.clip(int(np.sqrt(max(capacity, 1))), 8, 4096))
        self._nprobe = nprobe
        # faiss itself warns below ~39 training points per centroid.
        self._train_after = train_after or self._nlist * 39
        self._index: Any = None

    def _maybe_train(self) -> None:
        if self._index is not None or len(self._live) < self._train_after:
            return
        quantizer = self._faiss.IndexFlatIP(self._dim)
        index = self._faiss.IndexIVFFlat(
            quantizer, self._dim, self._nlist, self._faiss.METRIC_INNER_PRODUCT
        )
        slots = np.fromiter(sorted(self._live), dtype=np.int64)
        training = np.ascontiguousarray(self._mirror.matrix[slots])
        index.train(training)
        index.add_with_ids(training, slots)
        index.nprobe = self._nprobe
        self._index = index

    def upsert(self, slot: int, vector: np.ndarray) -> None:
        self._mirror.write(slot, vector)
        self._live.add(slot)
        if self._index is not None:
            ids = np.array([slot], dtype=np.int64)
            self._index.remove_ids(self._faiss.IDSelectorBatch(ids))
            self._index.add_with_ids(
                np.ascontiguousarray(vector.reshape(1, -1), dtype=_DTYPE), ids
            )
            return
        self._maybe_train()

    def remove(self, slot: int) -> None:
        if slot not in self._live:
            return
        self._live.discard(slot)
        self._mirror.zero(slot)
        if self._index is not None:
            self._index.remove_ids(self._faiss.IDSelectorBatch(np.array([slot], dtype=np.int64)))

    def search(self, vector: np.ndarray, k: int) -> list[int]:
        if not self._live:
            return []
        k = min(k, len(self._live))
        if self._index is None:
            return _top_k(self._mirror.matrix @ vector, k)
        _, labels = self._index.search(
            np.ascontiguousarray(vector.reshape(1, -1), dtype=_DTYPE), k
        )
        return [int(label) for label in labels[0] if label >= 0]

    def vector(self, slot: int) -> np.ndarray:
        return self._mirror.row(slot)

    @property
    def trained(self) -> bool:
        return self._index is not None

    def clear(self) -> None:
        self._mirror.clear()
        self._live.clear()
        self._index = None


def available_backends() -> list[str]:
    """Backend names that can actually be constructed in this environment."""
    names = ["exact"]
    for name, module in (("hnsw", "hnswlib"), ("faiss-ivf", "faiss")):
        try:
            __import__(module)
        except ImportError:
            continue
        names.append(name)
    return names


def build_index(dim: int, capacity: int, backend: str = "auto", **kwargs) -> VectorIndex:
    """Constructs an index backend by name.

    ``auto`` keeps the exact scan at the default cache size and reaches for ANN
    only above ``ANN_AUTO_ABOVE``, where the scan stops being cheap relative to
    the embedding pass. An explicitly named backend that is not installed is an
    error rather than a silent downgrade: a cache that quietly stopped being the
    thing you configured is worse than one that refuses to start.
    """
    if backend == "auto":
        if capacity > ANN_AUTO_ABOVE:
            for name in ("hnsw", "faiss-ivf"):
                if name in available_backends():
                    return build_index(dim, capacity, name, **kwargs)
        return ExactIndex(dim, capacity)
    if backend == "exact":
        return ExactIndex(dim, capacity)
    if backend == "hnsw":
        return HnswIndex(dim, capacity, **kwargs)
    if backend == "faiss-ivf":
        return FaissIvfIndex(dim, capacity, **kwargs)
    raise ValueError(f"unknown index backend {backend!r}; available: {available_backends()}")
