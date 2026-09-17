"""Negative feedback is the only signal that a false hit happened at all.

A semantic cache fails silently: the wrong answer is returned confidently, with
a plausible similarity score, and nothing in the system knows. The user does.
"""

import pytest

from gateway.engine import SemanticCacheEngine
from gateway.semantic_cache import SemanticCacheGateway
from gateway.vector_store import LocalVectorStore

DOC = "KYC records must be retained for 5 years."
QUESTION = "How long must KYC records be kept?"
PARAPHRASE = "How long must KYC records be retained?"


@pytest.fixture
def gateway():
    store = LocalVectorStore()
    store.ingest_document("doc-1", DOC)
    return SemanticCacheGateway(vector_store=store, similarity_threshold=0.75)


def test_thumbs_down_evicts_the_entry_that_was_served(gateway):
    first = gateway.process(QUESTION)

    assert gateway.feedback(first.entry_id, helpful=False)

    assert gateway.process(PARAPHRASE).status == "MISS"


def test_thumbs_up_does_not_change_the_cache(gateway):
    """Positive feedback arrives from whoever is holding the other end of the
    connection. It is not a reason to keep an entry longer than the LRU would."""
    first = gateway.process(QUESTION)

    assert not gateway.feedback(first.entry_id, helpful=True)
    assert gateway.process(PARAPHRASE).status == "HIT"


def test_feedback_on_an_unknown_entry_is_a_no_op(gateway):
    gateway.process(QUESTION)

    assert not gateway.feedback("not-an-entry-id", helpful=False)


def test_feedback_targets_the_entry_not_the_query(gateway):
    """By the time a user reacts, the query that produced the answer may no
    longer be the nearest neighbour of anything -- searching again to decide
    what to delete could delete a different entry."""
    served = gateway.process(QUESTION)
    gateway.process("An entirely unrelated question about tax filings")

    gateway.feedback(served.entry_id, helpful=False)

    assert len(gateway.cache) == 1


def test_feedback_eviction_is_counted_separately(gateway):
    served = gateway.process(QUESTION)

    gateway.feedback(served.entry_id, helpful=False)

    snapshot = gateway.stats()["metrics"]
    assert snapshot["feedback_evictions"] == 1
    assert snapshot["invalidations"] == 1


def test_immediate_rephrase_can_be_treated_as_negative_feedback():
    """Opt-in, because with a conversational resolver a legitimate follow-up
    looks exactly like a rephrase."""
    clock = [1000.0]
    engine = SemanticCacheEngine(
        LocalVectorStore(),
        threshold=0.75,
        rephrase_feedback=True,
        rephrase_window_s=30,
        rephrase_similarity=0.8,
        time_fn=lambda: clock[0],
    )
    engine.answer(QUESTION, lambda prepared: "five years")

    clock[0] += 5
    result = engine.answer(PARAPHRASE, lambda prepared: "five years, from account closure")

    assert engine.metrics.snapshot()["feedback_evictions"] == 1
    assert result.status == "MISS", "the rephrase must not be served the entry it rejected"


def test_rephrase_outside_the_window_is_just_another_query():
    clock = [1000.0]
    engine = SemanticCacheEngine(
        LocalVectorStore(),
        threshold=0.75,
        rephrase_feedback=True,
        rephrase_window_s=30,
        rephrase_similarity=0.8,
        time_fn=lambda: clock[0],
    )
    engine.answer(QUESTION, lambda prepared: "five years")

    clock[0] += 31
    result = engine.answer(PARAPHRASE, lambda prepared: "should not be called")

    assert result.status == "HIT"
    assert engine.metrics.snapshot()["feedback_evictions"] == 0


def test_rephrase_detection_is_off_by_default():
    engine = SemanticCacheEngine(LocalVectorStore(), threshold=0.75)
    engine.answer(QUESTION, lambda prepared: "five years")

    result = engine.answer(PARAPHRASE, lambda prepared: "should not be called")

    assert result.status == "HIT"
