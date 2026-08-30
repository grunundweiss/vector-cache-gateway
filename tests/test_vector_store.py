import pytest

from gateway.vector_store import LocalVectorStore


def test_similarity_search_on_empty_store_returns_empty_list():
    store = LocalVectorStore()
    assert store.similarity_search("anything") == []


def test_ingest_document_stores_text_and_metadata():
    store = LocalVectorStore()
    store.ingest_document("doc-1", "hello world", metadata={"source": "unit-test"})

    assert len(store.documents) == 1
    assert store.documents[0]["id"] == "doc-1"
    assert store.documents[0]["text"] == "hello world"
    assert store.documents[0]["metadata"] == {"source": "unit-test"}


def test_similarity_search_ranks_exact_match_first():
    store = LocalVectorStore()
    store.ingest_document("doc-1", "the quick brown fox")
    store.ingest_document("doc-2", "completely unrelated content")

    results = store.similarity_search("the quick brown fox", top_k=2)

    assert results[0]["document"]["id"] == "doc-1"
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[0]["score"] > results[1]["score"]


def test_similarity_search_respects_top_k():
    store = LocalVectorStore()
    for i in range(5):
        store.ingest_document(f"doc-{i}", f"document number {i}")

    results = store.similarity_search("document number 0", top_k=2)

    assert len(results) == 2
