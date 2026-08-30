"""The cache is bounded and evicts least-recently-used entries."""

import numpy as np
import pytest

from gateway.cache import SemanticCache


def vec(seed: int, dim: int = 8) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=dim)


def test_cache_never_exceeds_max_entries():
    cache = SemanticCache(threshold=0.75, max_entries=3)

    for i in range(50):
        cache.insert(f"query-{i}", vec(i), f"answer-{i}")

    assert len(cache) == 3


def test_reinserting_same_key_updates_in_place():
    cache = SemanticCache(threshold=0.75, max_entries=10)

    cache.insert("q", vec(1), "first answer")
    cache.insert("q", vec(1), "second answer")

    assert len(cache) == 1
    hit = cache.lookup(vec(1))
    assert hit is not None
    assert hit[0] == "second answer"


def test_eviction_removes_least_recently_used_not_oldest():
    cache = SemanticCache(threshold=0.99, max_entries=2)
    cache.insert("a", vec(1), "answer-a")
    cache.insert("b", vec(2), "answer-b")

    # Touch "a" so "b" becomes least-recently-used.
    assert cache.lookup(vec(1)) is not None

    cache.insert("c", vec(3), "answer-c")

    assert cache.lookup(vec(1)) is not None, "recently used entry should survive"
    assert cache.lookup(vec(2)) is None, "least-recently-used entry should be evicted"
    assert cache.lookup(vec(3)) is not None


def test_max_entries_must_be_positive():
    with pytest.raises(ValueError):
        SemanticCache(max_entries=0)
