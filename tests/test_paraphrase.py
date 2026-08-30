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


@pytest.fixture
def gateway():
    store = LocalVectorStore()
    store.ingest_document("doc-1", DOC)
    return SemanticCacheGateway(vector_store=store, similarity_threshold=0.75)


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
def test_single_token_topic_swaps_are_not_separable_lexically(gateway, first, second):
    """Characterization test: a known limitation, not desired behaviour.

    Swapping one topic token (KYC -> AML) leaves every other word identical, so
    a lexical measure scores it 0.86-0.88 -- at or above what genuine
    paraphrases score. No threshold can separate these two classes here,
    because the signal that distinguishes them is semantic, not lexical.

    This is precisely why the shipped threshold is chosen from a sweep against
    the real encoder rather than picked by hand. If a future change makes the
    fake separate these, this test should start failing and be deleted.
    """
    gateway.process_query(first)

    _, status = gateway.process_query(second)

    assert status.startswith("CACHE_HIT"), (
        "expected the lexical fake to conflate these; if it no longer does, "
        "the fake has improved and this characterization test is obsolete"
    )
