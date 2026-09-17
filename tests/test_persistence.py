"""A cache that empties on every deploy cannot save anything over time.

The snapshot is also the place where a cache stops being ephemeral: it puts user
questions and the answers derived from them on disk. These tests cover the
mechanics; the handling requirements are in gateway/persistence.py.
"""

import numpy as np
import pytest

from gateway.partition import PartitionedCache
from gateway.persistence import (
    SnapshotMismatch,
    load_snapshot,
    read_header,
    save_snapshot,
)
from gateway.semantic_cache import SemanticCacheGateway
from gateway.vector_store import LocalVectorStore


def vec(seed: int, dim: int = 8) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=dim)


@pytest.fixture
def snapshot_path(tmp_path):
    return tmp_path / "cache.npz"


@pytest.fixture
def populated():
    cache = PartitionedCache(threshold=0.9, max_entries=10)
    cache.insert("q1", vec(1), "answer one", namespace="tenant-a")
    cache.insert("q2", vec(2), "answer two", namespace="tenant-b")
    return cache


def test_entries_survive_a_round_trip(populated, snapshot_path):
    save_snapshot(populated, snapshot_path, model="fake", corpus_version=1)

    restored = PartitionedCache(threshold=0.9, max_entries=10)
    count = load_snapshot(restored, snapshot_path, model="fake", corpus_version=1)

    assert count == 2
    assert restored.lookup(vec(1), namespace="tenant-a").answer == "answer one"


def test_namespaces_survive_the_round_trip(populated, snapshot_path):
    """A snapshot that merged tenants would be a data leak with a restart in
    the middle of it."""
    save_snapshot(populated, snapshot_path, model="fake", corpus_version=1)
    restored = PartitionedCache(threshold=0.9, max_entries=10)
    load_snapshot(restored, snapshot_path, model="fake", corpus_version=1)

    assert restored.lookup(vec(1), namespace="tenant-b") is None
    assert sorted(restored.namespaces()) == ["tenant-a", "tenant-b"]


def test_entry_ids_survive_so_feedback_still_works(populated, snapshot_path):
    entry_id = populated.lookup(vec(1), namespace="tenant-a").entry_id
    save_snapshot(populated, snapshot_path, model="fake")

    restored = PartitionedCache(threshold=0.9, max_entries=10)
    load_snapshot(restored, snapshot_path, model="fake")

    assert restored.invalidate(entry_id)


def test_a_snapshot_from_another_encoder_is_refused(populated, snapshot_path):
    """Vectors from a different model are not comparable to these ones. Serving
    them would be a permanent, silent false-hit generator."""
    save_snapshot(populated, snapshot_path, model="all-mpnet-base-v2")

    with pytest.raises(SnapshotMismatch, match="encoder"):
        load_snapshot(PartitionedCache(), snapshot_path, model="all-MiniLM-L6-v2")


def test_a_snapshot_from_another_corpus_version_is_refused(populated, snapshot_path):
    save_snapshot(populated, snapshot_path, model="fake", corpus_version=1)

    with pytest.raises(SnapshotMismatch, match="corpus version"):
        load_snapshot(PartitionedCache(), snapshot_path, model="fake", corpus_version=2)


def test_non_strict_load_starts_cold_instead_of_failing(populated, snapshot_path):
    save_snapshot(populated, snapshot_path, model="fake")

    restored = PartitionedCache()
    count = load_snapshot(restored, snapshot_path, model="other", strict=False)

    assert count == 0
    assert len(restored) == 0


def test_already_expired_entries_are_dropped_on_load(snapshot_path):
    """Deadlines are absolute, so a snapshot that sat on disk over a weekend
    comes back mostly empty. That is the point of having set a TTL."""
    cache = PartitionedCache(threshold=0.9, max_entries=10)
    cache.insert("fresh", vec(1), "a", ttl=10_000, now=1000)
    cache.insert("stale", vec(2), "b", ttl=10, now=1000)
    save_snapshot(cache, snapshot_path, model="fake")

    restored = PartitionedCache(threshold=0.9, max_entries=10)
    count = load_snapshot(restored, snapshot_path, model="fake", now=2000)

    assert count == 1
    assert restored.lookup(vec(1), now=2000) is not None


def test_header_can_be_read_without_loading_vectors(populated, snapshot_path):
    save_snapshot(populated, snapshot_path, model="fake", corpus_version=7)

    header = read_header(snapshot_path)

    assert header["model"] == "fake"
    assert header["corpus_version"] == 7
    assert header["entries"] == 2


def test_empty_cache_snapshots_cleanly(snapshot_path):
    assert save_snapshot(PartitionedCache(), snapshot_path, model="fake") == 0
    assert load_snapshot(PartitionedCache(), snapshot_path, model="fake") == 0


def test_gateway_round_trips_through_a_snapshot(tmp_path):
    store = LocalVectorStore()
    store.ingest_document("doc-1", "KYC records must be retained for 5 years.")
    gateway = SemanticCacheGateway(vector_store=store, similarity_threshold=0.75)
    gateway.process_query("How long must KYC records be kept?")

    path = tmp_path / "gateway.npz"
    assert gateway.save_snapshot(path) == 1

    restarted = SemanticCacheGateway(vector_store=store, similarity_threshold=0.75)
    assert restarted.load_snapshot(path) == 1

    answer, status = restarted.process_query("How long must KYC records be retained?")

    assert status.startswith("CACHE_HIT"), "a restarted process should not start cold"
    assert answer == "KYC records must be retained for 5 years."


def test_snapshot_write_is_atomic(populated, tmp_path):
    """A snapshot half-written by a process that died mid-deploy is a file that
    loads, validates, and serves truncated nonsense."""
    path = tmp_path / "cache.npz"
    save_snapshot(populated, path, model="fake")

    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp-" in p.name]

    assert leftovers == []
    assert path.exists()
