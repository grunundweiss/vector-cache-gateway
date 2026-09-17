"""Coverage for the similarity gate itself: what should hit, and what must not.

These exercise the behaviour the repo exists to provide. They run against the
lexical fake encoder from conftest, so read the limitation test at the bottom
before trusting any of these numbers as semantic results -- the real-encoder
picture is measured in benchmarks/threshold_sweep.py.
"""

import pytest

from gateway.semantic_cache import SemanticCacheGateway
from gateway.vector_store import LocalVectorStore

DOC = "KYC records must be retained for a minimum of 5 years."

# Same question, different words. These should be served from cache.
PARAPHRASES = [
    (
        "How long must KYC records be kept?",
        "How long must KYC records be retained?",
    ),
    (
        "What is the retention limit for KYC records?",
        "What is the retention period for KYC records?",
    ),
    (
        "Does the protocol require multi-factor authentication for large transfers?",
        "Does the protocol require multi-factor authentication for big transfers?",
    ),
]

# Different questions that share surface form. A hit here returns the wrong
# answer with full confidence, which is the failure mode that makes semantic
# caching hard. These are the valuable half of the suite.
NEAR_MISSES = [
    (
        "What is the retention limit for KYC records?",
        "Who approves KYC exemptions?",
    ),
    (
        "How long must KYC records be kept?",
        "What is the capital of France?",
    ),
]


def make_gateway(guard="default"):
    store = LocalVectorStore()
    store.ingest_document("doc-1", DOC)
    return SemanticCacheGateway(
        vector_store=store, similarity_threshold=0.75, guard=guard
    )


@pytest.fixture
def gateway():
    return make_gateway()


@pytest.mark.parametrize("first,second", PARAPHRASES)
def test_paraphrase_hits(gateway, first, second):
    first_answer, first_status = gateway.process_query(first)
    assert first_status == "CACHE_MISS"

    second_answer, second_status = gateway.process_query(second)

    assert second_status.startswith("CACHE_HIT"), (
        f"{second!r} should have been served from the cache entry for {first!r}"
    )
    assert second_answer == first_answer


@pytest.mark.parametrize("first,second", NEAR_MISSES)
def test_near_miss_does_not_hit(gateway, first, second):
    gateway.process_query(first)

    _, status = gateway.process_query(second)

    assert status == "CACHE_MISS", (
        f"{second!r} is a different question from {first!r} and must not be "
        "answered from its cache entry"
    )


@pytest.mark.parametrize(
    "first,second",
    [
        (
            "What is the retention limit for KYC records?",
            "What is the retention limit for AML records?",
        ),
        (
            "How long must KYC records be kept?",
            "How long must tax records be kept?",
        ),
    ],
)
def test_topic_swaps_clear_the_gate_and_are_stopped_by_the_guard(first, second):
    """The similarity gate cannot separate these. The guard can.

    Swapping one topic token (KYC -> AML) leaves every other word identical, so
    a lexical measure scores it 0.86-0.88 -- at or above what genuine
    paraphrases score, and the real encoder agrees: the sweep measures 0.8608
    for this pair. No threshold separates these two classes, because the signal
    that distinguishes them is not a matter of degree.

    Both halves are asserted on purpose. The first is the limitation the
    threshold sweep documents; the second is the guard that answers it. If the
    first assertion ever fails, the fake encoder has changed and this test is
    measuring something else.
    """
    unguarded = make_gateway(guard=None)
    unguarded.process_query(first)
    _, status = unguarded.process_query(second)
    assert status.startswith("CACHE_HIT"), (
        "expected the similarity gate alone to conflate these; if it no longer "
        "does, the encoder has changed and this test is obsolete"
    )

    guarded = make_gateway()
    guarded.process_query(first)
    _, status = guarded.process_query(second)
    assert status == "CACHE_MISS", "the guard must veto a swapped topic token"
