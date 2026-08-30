"""Cached answers must not outlive the corpus they were derived from."""

from gateway.semantic_cache import SemanticCacheGateway
from gateway.vector_store import LocalVectorStore

QUERY = "How long must KYC records be kept?"
OLD_DOC = "KYC records must be retained for 5 years."
NEW_DOC = "How long must KYC records be kept? Updated policy: 7 years."


def test_ingest_bumps_corpus_version():
    store = LocalVectorStore()
    assert store.version == 0

    store.ingest_document("doc-1", OLD_DOC)

    assert store.version == 1


def test_cached_answer_is_dropped_after_new_document_is_ingested():
    store = LocalVectorStore()
    store.ingest_document("doc-1", OLD_DOC)
    gateway = SemanticCacheGateway(vector_store=store, similarity_threshold=0.75)

    answer, status = gateway.process_query(QUERY)
    assert status == "CACHE_MISS"
    assert answer == OLD_DOC

    # A better-matching document arrives after the answer was cached.
    store.ingest_document("doc-2", NEW_DOC)

    answer, status = gateway.process_query(QUERY)
    assert status == "CACHE_MISS", "corpus changed, so the cache must not serve a stale hit"
    assert answer == NEW_DOC


def test_cache_still_hits_when_corpus_is_unchanged():
    store = LocalVectorStore()
    store.ingest_document("doc-1", OLD_DOC)
    gateway = SemanticCacheGateway(vector_store=store, similarity_threshold=0.75)

    gateway.process_query(QUERY)
    _, status = gateway.process_query(QUERY)

    assert status.startswith("CACHE_HIT")
