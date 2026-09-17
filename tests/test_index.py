"""Index backends must agree about what is in the cache, not about speed.

An ANN backend is allowed to miss a neighbour -- that is what approximate means.
It is not allowed to disagree about a vector's score, return a slot that was
evicted, or fail to notice that a slot was overwritten. Those would turn a
tuning decision into a correctness one.
"""

import numpy as np
import pytest

from gateway.index import (
    ANN_AUTO_ABOVE,
    ExactIndex,
    available_backends,
    build_index,
)

DIM = 16


def unit_vectors(count: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(count, dim)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@pytest.fixture(params=available_backends())
def backend(request):
    return request.param


def build(backend: str, capacity: int = 256):
    return build_index(DIM, capacity, backend)


def test_search_finds_the_vector_that_was_stored(backend):
    index = build(backend)
    vectors = unit_vectors(50)
    for slot, vector in enumerate(vectors):
        index.upsert(slot, vector)

    assert index.search(vectors[7], 1)[0] == 7


def test_removed_slot_stops_being_returned(backend):
    index = build(backend)
    vectors = unit_vectors(20)
    for slot, vector in enumerate(vectors):
        index.upsert(slot, vector)

    index.remove(5)

    assert 5 not in index.search(vectors[5], 5)


def test_slot_can_be_reused_after_removal(backend):
    """Eviction overwrites a slot in place. The index has to follow."""
    index = build(backend)
    vectors = unit_vectors(20)
    for slot, vector in enumerate(vectors):
        index.upsert(slot, vector)

    index.remove(5)
    index.upsert(5, vectors[11])

    assert index.search(vectors[11], 2)[:1] in ([5], [11])
    assert set(index.search(vectors[11], 2)) == {5, 11}


def test_stored_vector_round_trips(backend):
    index = build(backend)
    vectors = unit_vectors(5)
    for slot, vector in enumerate(vectors):
        index.upsert(slot, vector)

    assert np.allclose(index.vector(3), vectors[3], atol=1e-6)


def test_search_on_empty_index_returns_nothing(backend):
    assert build(backend).search(unit_vectors(1)[0], 5) == []


def test_clear_empties_the_index(backend):
    index = build(backend)
    vectors = unit_vectors(10)
    for slot, vector in enumerate(vectors):
        index.upsert(slot, vector)

    index.clear()

    assert index.search(vectors[0], 3) == []


def test_exact_index_is_exact():
    """Recall 1.0 by construction is the property the default rests on."""
    index = ExactIndex(DIM, 512)
    vectors = unit_vectors(300, seed=3)
    for slot, vector in enumerate(vectors):
        index.upsert(slot, vector)

    probes = unit_vectors(50, seed=99)
    for probe in probes:
        expected = int(np.argmax(vectors @ probe))
        assert index.search(probe, 1)[0] == expected


def test_ann_recall_is_high_enough_to_be_useful():
    """Not a guarantee of exactness -- a floor under how approximate it gets.

    Recall below this would mean the cache is silently losing hits it holds,
    which looks exactly like a threshold that is too strict.
    """
    if "hnsw" not in available_backends():
        pytest.skip("hnswlib not installed")

    index = build_index(DIM, 2000, "hnsw")
    vectors = unit_vectors(1000, seed=5)
    for slot, vector in enumerate(vectors):
        index.upsert(slot, vector)

    probes = unit_vectors(200, seed=7)
    found = sum(
        int(np.argmax(vectors @ probe)) in index.search(probe, 5) for probe in probes
    )

    assert found / len(probes) >= 0.95


def test_auto_keeps_the_exact_scan_at_the_default_size():
    assert build_index(DIM, ANN_AUTO_ABOVE, "auto").name == "exact"


def test_auto_reaches_for_ann_above_the_bound():
    expected = "exact" if available_backends() == ["exact"] else "hnsw"
    assert build_index(DIM, ANN_AUTO_ABOVE * 2, "auto").name == expected


def test_unknown_backend_is_an_error_not_a_downgrade():
    with pytest.raises(ValueError, match="unknown index backend"):
        build_index(DIM, 10, "annoy")


def test_faiss_trains_and_keeps_answering():
    """Before training it scans exactly; after training it uses the IVF lists."""
    if "faiss-ivf" not in available_backends():
        pytest.skip("faiss not installed")

    index = build_index(DIM, 4096, "faiss-ivf", nlist=8, train_after=200)
    vectors = unit_vectors(400, seed=11)

    for slot, vector in enumerate(vectors[:100]):
        index.upsert(slot, vector)
    assert not index.trained
    assert index.search(vectors[42], 1)[0] == 42

    for slot, vector in enumerate(vectors[100:], start=100):
        index.upsert(slot, vector)
    assert index.trained
    assert 42 in index.search(vectors[42], 10)


def test_cache_behaves_the_same_on_every_backend(backend):
    """The backend decides what is found, never what a hit means.

    Same inserts, same lookups, same thresholds -- the only thing an ANN backend
    is allowed to change is recall, and at this size it does not.
    """
    from gateway.cache import SemanticCache

    cache = SemanticCache(threshold=0.9, max_entries=64, index_backend=backend)
    vectors = unit_vectors(20, seed=13)
    for i, vector in enumerate(vectors):
        cache.insert(f"q-{i}", vector, f"a-{i}")

    hit = cache.lookup(vectors[7])

    assert hit is not None
    assert hit.answer == "a-7"
    assert hit.score == pytest.approx(1.0, abs=1e-5)


def test_eviction_works_on_every_backend(backend):
    """Eviction overwrites a slot the index already holds. An index that kept
    the old vector would serve the evicted answer under the new key."""
    from gateway.cache import SemanticCache

    cache = SemanticCache(threshold=0.9, max_entries=4, index_backend=backend)
    vectors = unit_vectors(12, seed=17)
    for i, vector in enumerate(vectors):
        cache.insert(f"q-{i}", vector, f"a-{i}")

    assert len(cache) == 4
    assert cache.lookup(vectors[0]) is None, "the first entry should be long gone"

    hit = cache.lookup(vectors[11])
    assert hit is not None and hit.answer == "a-11"


def test_invalidated_entry_disappears_on_every_backend(backend):
    from gateway.cache import SemanticCache

    cache = SemanticCache(threshold=0.9, max_entries=16, index_backend=backend)
    vectors = unit_vectors(8, seed=19)
    ids = [cache.insert(f"q-{i}", v, f"a-{i}") for i, v in enumerate(vectors)]

    assert cache.invalidate(ids[3], feedback=True)

    assert cache.lookup(vectors[3]) is None
    assert cache.lookup(vectors[4]) is not None
