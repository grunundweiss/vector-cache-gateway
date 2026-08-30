"""Locks in the encode-once invariant.

Embedding is the dominant cost of a query, so a cache that embeds the same
string more than once per lookup defeats its own purpose. These tests fail
loudly if that regresses.
"""

import pytest

from gateway.semantic_cache import SemanticCacheGateway
from gateway.vector_store import LocalVectorStore


@pytest.fixture
def counted_gateway():
    store = LocalVectorStore()
    store.ingest_document("doc-1", "KYC records must be retained for 5 years.")

    calls = {"n": 0}
    original_encode = store.encode

    def counting_encode(text):
        calls["n"] += 1
        return original_encode(text)

    store.encode = counting_encode
    gateway = SemanticCacheGateway(vector_store=store, similarity_threshold=0.75)
    return gateway, calls


def test_first_query_encodes_exactly_once(counted_gateway):
    gateway, calls = counted_gateway

    gateway.process_query("How long must KYC records be kept?")

    assert calls["n"] == 1


def test_subsequent_miss_encodes_exactly_once(counted_gateway):
    gateway, calls = counted_gateway
    gateway.process_query("How long must KYC records be kept?")

    calls["n"] = 0
    gateway.process_query("An entirely unrelated question about tax law")

    assert calls["n"] == 1


def test_cache_hit_encodes_exactly_once(counted_gateway):
    gateway, calls = counted_gateway
    query = "How long must KYC records be kept?"
    gateway.process_query(query)

    calls["n"] = 0
    _, status = gateway.process_query(query)

    assert status.startswith("CACHE_HIT")
    assert calls["n"] == 1
