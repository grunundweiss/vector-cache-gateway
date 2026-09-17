"""Answers go stale on their own, not only when the corpus changes.

Corpus-change invalidation catches "the document was updated". It does not catch
"this answer was only ever true for an hour" -- store hours, a current
promotion, today's rate. Those expire on a clock nobody else is watching.
"""

import numpy as np
import pytest

from gateway.cache import SemanticCache


def vec(seed: int, dim: int = 8) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=dim)


@pytest.fixture
def clock():
    return [1_000.0]


def cache_at(clock, **kwargs) -> SemanticCache:
    return SemanticCache(threshold=0.9, max_entries=10, time_fn=lambda: clock[0], **kwargs)


def test_entry_is_served_before_its_ttl_elapses(clock):
    cache = cache_at(clock, ttl=60)
    cache.insert("q", vec(1), "answer")

    clock[0] += 59

    assert cache.lookup(vec(1)) is not None


def test_entry_is_not_served_after_its_ttl_elapses(clock):
    cache = cache_at(clock, ttl=60)
    cache.insert("q", vec(1), "answer")

    clock[0] += 61

    assert cache.lookup(vec(1)) is None


def test_expired_entry_frees_its_slot_for_reuse(clock):
    """An expired entry that keeps its slot is a leak with a deadline."""
    cache = cache_at(clock, ttl=60)
    cache.insert("q", vec(1), "answer")

    clock[0] += 61
    cache.lookup(vec(1))

    assert len(cache) == 0
    cache.insert("q2", vec(2), "another")
    assert cache.describe()["allocated_slots"] == 1


def test_per_entry_ttl_overrides_the_cache_default(clock):
    """A volatile answer must not force a short lifetime on a stable one."""
    cache = cache_at(clock, ttl=3600)
    cache.insert("stable", vec(1), "5 years")
    cache.insert("volatile", vec(2), "today's rate", ttl=10)

    clock[0] += 11

    assert cache.lookup(vec(1)) is not None
    assert cache.lookup(vec(2)) is None


def test_no_ttl_means_no_expiry(clock):
    cache = cache_at(clock)
    cache.insert("q", vec(1), "answer")

    clock[0] += 10_000_000

    assert cache.lookup(vec(1)) is not None


def test_purge_expired_collects_entries_nobody_queried(clock):
    cache = cache_at(clock, ttl=60)
    for i in range(5):
        cache.insert(f"q{i}", vec(i), f"a{i}")

    clock[0] += 61

    assert cache.purge_expired() == 5
    assert len(cache) == 0


def test_negative_ttl_is_rejected():
    with pytest.raises(ValueError, match="ttl must be positive"):
        SemanticCache(ttl=0)
