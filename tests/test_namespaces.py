"""Cache partitions are a data boundary, not a tuning parameter.

An exact-match cache can only return one user's answer to another if they send
the same string. A similarity cache will do it for two different strings that
mean the same thing -- which is precisely what "what's my balance" is across
every user of a system.
"""

import numpy as np
import pytest

from gateway.partition import PartitionedCache


def vec(seed: int, dim: int = 8) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=dim)


@pytest.fixture
def cache():
    return PartitionedCache(threshold=0.9, max_entries=10, max_namespaces=3)


def test_one_tenants_answer_is_not_served_to_another(cache):
    cache.insert("What is my balance?", vec(1), "tenant-a: 120 NOK", namespace="tenant-a")

    assert cache.lookup(vec(1), namespace="tenant-b") is None
    assert cache.lookup(vec(1), namespace="tenant-a").answer == "tenant-a: 120 NOK"


def test_identical_questions_are_cached_separately_per_tenant(cache):
    cache.insert("balance", vec(1), "a", namespace="tenant-a")
    cache.insert("balance", vec(1), "b", namespace="tenant-b")

    assert cache.lookup(vec(1), namespace="tenant-a").answer == "a"
    assert cache.lookup(vec(1), namespace="tenant-b").answer == "b"
    assert len(cache) == 2


def test_unknown_namespace_does_not_create_a_partition_on_lookup(cache):
    cache.lookup(vec(1), namespace="never-written-to")

    assert cache.namespaces() == []


def test_namespace_count_is_bounded_least_recently_used_first(cache):
    for name in ("a", "b", "c"):
        cache.insert("q", vec(1), name, namespace=name)
    cache.lookup(vec(1), namespace="a")  # touch "a" so "b" is coldest

    cache.insert("q", vec(1), "d", namespace="d")

    assert sorted(cache.namespaces()) == ["a", "c", "d"]


def test_clearing_one_namespace_leaves_the_others(cache):
    cache.insert("q", vec(1), "a", namespace="tenant-a")
    cache.insert("q", vec(1), "b", namespace="tenant-b")

    cache.clear("tenant-a")

    assert cache.lookup(vec(1), namespace="tenant-a") is None
    assert cache.lookup(vec(1), namespace="tenant-b") is not None


def test_invalidate_finds_an_entry_in_whichever_namespace_holds_it(cache):
    cache.insert("q", vec(1), "a", namespace="tenant-a")
    entry_id = cache.insert("q", vec(2), "b", namespace="tenant-b")

    assert cache.invalidate(entry_id, feedback=True)
    assert cache.lookup(vec(2), namespace="tenant-b") is None
    assert cache.lookup(vec(1), namespace="tenant-a") is not None


def test_describe_reports_per_partition_state(cache):
    cache.insert("q", vec(1), "a", namespace="tenant-a")

    described = cache.describe()

    assert described["namespaces"] == 1
    assert described["entries"] == 1
    assert described["partitions"][0]["namespace"] == "tenant-a"
