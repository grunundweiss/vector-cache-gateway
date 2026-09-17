"""Without these counters, a threshold that is too loose and one that is too
tight look identical from the outside: a service that is up.
"""

import pytest

from gateway.metrics import CacheMetrics
from gateway.semantic_cache import SemanticCacheGateway
from gateway.vector_store import LocalVectorStore


@pytest.fixture
def metrics():
    return CacheMetrics()


def test_hit_rate_excludes_traffic_that_was_never_cacheable(metrics):
    """A tool call is not a miss. Counting it as one makes the cache look worse
    than it is, and hides how much traffic it never saw."""
    metrics.record_hit(0.9, 10)
    metrics.record_miss(500)
    metrics.record_bypass("tools_offered")

    snapshot = metrics.snapshot()

    assert snapshot["hit_rate"] == 0.5
    assert snapshot["queries"] == 3
    assert snapshot["bypasses"] == 1


def test_hit_similarity_is_summarized_not_just_counted(metrics):
    """The mean and the minimum are what say whether the threshold is right.
    A hit rate alone cannot distinguish good matches from barely-passing ones."""
    for score in (0.86, 0.91, 0.99):
        metrics.record_hit(score, 10)

    similarity = metrics.snapshot()["hit_similarity"]

    assert similarity["mean"] == pytest.approx(0.92, abs=0.01)
    assert similarity["min"] == pytest.approx(0.86)


def test_estimated_saving_is_hits_times_the_gap(metrics):
    metrics.record_miss(500)
    metrics.record_hit(0.9, 100)
    metrics.record_hit(0.9, 100)

    assert metrics.snapshot()["estimated_ms_saved"] == pytest.approx(800)


def test_saving_is_never_negative(metrics):
    """With a retrieval-only backend a hit is not faster than a miss. The
    estimate must report zero rather than a negative saving."""
    metrics.record_miss(40)
    metrics.record_hit(0.9, 48)

    assert metrics.snapshot()["estimated_ms_saved"] == 0.0


def test_guard_vetoes_are_attributed_to_a_guard(metrics):
    metrics.record_guard_veto("numeric: literals differ: [5] vs [7]")
    metrics.record_guard_veto("polarity: mutually exclusive terms")
    metrics.record_guard_veto("numeric: literals differ: [1] vs [2]")

    assert metrics.snapshot()["veto_reasons"] == {"numeric": 2, "polarity": 1}


def test_samples_are_bounded(metrics):
    small = CacheMetrics(sample_size=10)

    for _ in range(1000):
        small.record_hit(0.9, 5)

    assert small.snapshot()["hit_similarity"]["n"] == 10
    assert small.hits == 1000, "the counter is exact even though the sample is bounded"


def test_prometheus_exposition_is_parseable(metrics):
    metrics.record_hit(0.9, 10)

    lines = [line for line in metrics.prometheus().splitlines() if not line.startswith("#")]
    values = dict(line.split(" ", 1) for line in lines)

    assert values["semantic_cache_hits_total"] == "1"
    assert float(values["semantic_cache_hit_rate"]) == 1.0


def test_gateway_records_its_own_traffic():
    store = LocalVectorStore()
    store.ingest_document("doc-1", "KYC records must be retained for 5 years.")
    gateway = SemanticCacheGateway(vector_store=store, similarity_threshold=0.75)

    gateway.process_query("How long must KYC records be kept?")
    gateway.process_query("How long must KYC records be retained?")

    snapshot = gateway.stats()["metrics"]

    assert snapshot["hits"] == 1
    assert snapshot["misses"] == 1
    assert snapshot["hit_similarity"]["n"] == 1


def test_corpus_invalidation_is_counted_not_silent():
    store = LocalVectorStore()
    store.ingest_document("doc-1", "KYC records must be retained for 5 years.")
    gateway = SemanticCacheGateway(vector_store=store, similarity_threshold=0.75)
    gateway.process_query("How long must KYC records be kept?")

    store.ingest_document("doc-2", "Updated policy: 7 years.")
    gateway.process_query("How long must KYC records be kept?")

    snapshot = gateway.stats()["metrics"]

    assert snapshot["corpus_invalidations"] == 1
    assert snapshot["invalidations"] == 1
