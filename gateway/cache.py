"""Similarity cache backed by a matrix of pre-normalized vectors."""

import logging

import numpy as np

logger = logging.getLogger(__name__)

_EPS = 1e-8
_INITIAL_CAPACITY = 64

# sentence-transformers emits float32; storing float64 would double the
# footprint to buy precision the encoder never produced.
_DTYPE = np.float32
DEFAULT_MAX_ENTRIES = 10_000

# Chosen from benchmarks/threshold_sweep.py, not by hand. At 0.85 the sweep
# measures 60% hit rate against 20% false hits -- the widest margin of any
# threshold tested. See "Choosing the threshold" in the README, including why
# 20% is still too high to run unguarded.
DEFAULT_THRESHOLD = 0.85


def _normalize(vector: np.ndarray) -> np.ndarray:
    return (vector / (np.linalg.norm(vector) + _EPS)).astype(_DTYPE, copy=False)


class SemanticCache:
    """Maps query embeddings to cached answers.

    Vectors are L2-normalized on insert, so a cosine-similarity lookup reduces
    to a single matrix-vector product instead of a Python loop that recomputes
    norms that never change.

    Capacity is bounded: at ``max_entries`` the least-recently-used entry is
    overwritten in place. Without a bound this is a memory leak with a
    similarity function attached -- a 768-dim vector is ~3 KB per entry before
    Python object overhead, so an unbounded cache grows without limit for as
    long as the process runs.

    Vectors are stored as float32, which is what sentence-transformers emits.
    float64 would double the footprint to buy precision the encoder never
    produced.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ):
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")

        self.threshold = threshold
        self.max_entries = max_entries

        self._buffer: np.ndarray | None = None  # (capacity, dim), normalized rows
        self._answers: list[str] = []
        self._keys: list[str] = []
        self._index: dict[str, int] = {}
        self._last_used: list[int] = []
        self._size = 0
        self._tick = 0

    def __len__(self) -> int:
        return self._size

    @property
    def _matrix(self) -> np.ndarray | None:
        """The populated rows of the buffer."""
        if self._buffer is None or self._size == 0:
            return None
        return self._buffer[: self._size]

    def lookup(self, embedding: np.ndarray) -> tuple[str, float] | None:
        """Returns (answer, score) for the nearest entry above threshold, else None."""
        matrix = self._matrix
        if matrix is None:
            return None

        scores = matrix @ _normalize(embedding)
        best = int(np.argmax(scores))
        best_score = float(scores[best])

        logger.debug(
            "closest cache match: %r | similarity: %.4f | threshold: %.2f",
            self._keys[best],
            best_score,
            self.threshold,
        )

        if best_score >= self.threshold:
            self._touch(best)
            return self._answers[best], best_score
        return None

    def insert(self, key: str, embedding: np.ndarray, answer: str) -> None:
        """Adds or updates an entry, evicting the least-recently-used one if full."""
        normalized = _normalize(embedding)

        existing = self._index.get(key)
        if existing is not None:
            self._write_slot(existing, key, normalized, answer)
            return

        if self._size >= self.max_entries:
            victim = int(np.argmin(self._last_used))
            logger.debug("cache full (%d entries), evicting %r", self._size, self._keys[victim])
            del self._index[self._keys[victim]]
            self._write_slot(victim, key, normalized, answer)
            return

        self._ensure_capacity(normalized.shape[0])
        self._keys.append(key)
        self._answers.append(answer)
        self._last_used.append(0)
        self._size += 1
        self._write_slot(self._size - 1, key, normalized, answer)

    def clear(self) -> None:
        self._buffer = None
        self._answers.clear()
        self._keys.clear()
        self._index.clear()
        self._last_used.clear()
        self._size = 0

    def _write_slot(self, idx: int, key: str, normalized: np.ndarray, answer: str) -> None:
        self._buffer[idx] = normalized  # type: ignore[index]
        self._keys[idx] = key
        self._answers[idx] = answer
        self._index[key] = idx
        self._touch(idx)

    def _touch(self, idx: int) -> None:
        self._tick += 1
        self._last_used[idx] = self._tick

    def _ensure_capacity(self, dim: int) -> None:
        """Grows the backing buffer geometrically so inserts amortize to O(1)."""
        capacity = min(_INITIAL_CAPACITY, self.max_entries)
        if self._buffer is None:
            self._buffer = np.empty((capacity, dim), dtype=_DTYPE)
            return
        if self._size >= self._buffer.shape[0]:
            grown_rows = min(self._buffer.shape[0] * 2, self.max_entries)
            grown = np.empty((grown_rows, dim), dtype=_DTYPE)
            grown[: self._size] = self._buffer[: self._size]
            self._buffer = grown
