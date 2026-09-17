from gateway.semantic_cache import SemanticCacheGateway
from gateway.vector_store import LocalVectorStore


def make_gateway(threshold=0.75):
    store = LocalVectorStore()
    store.ingest_document("doc-1", "KYC records must be retained for 5 years.")
    return SemanticCacheGateway(vector_store=store, similarity_threshold=threshold)


def test_first_query_is_a_cache_miss():
    gateway = make_gateway()

    answer, status = gateway.process_query("How long must KYC records be kept?")

    assert status == "CACHE_MISS"
    assert answer == "KYC records must be retained for 5 years."


def test_repeated_identical_query_is_a_cache_hit():
    gateway = make_gateway()
    query = "How long must KYC records be kept?"

    gateway.process_query(query)
    answer, status = gateway.process_query(query)

    assert status.startswith("CACHE_HIT")
    assert answer == "KYC records must be retained for 5 years."


def test_unrelated_query_is_a_cache_miss():
    gateway = make_gateway()

    gateway.process_query("How long must KYC records be kept?")
    _, status = gateway.process_query("What is the capital of France?")

    assert status == "CACHE_MISS"


def test_query_on_empty_vector_store_reports_no_match():
    gateway = SemanticCacheGateway(vector_store=LocalVectorStore())

    answer, status = gateway.process_query("anything at all")

    assert status == "CACHE_MISS"
    assert answer == "No matching compliance documentation found."


def test_skip_cache_neither_reads_nor_writes():
    """A request marked uncacheable still has to be answered.

    It skips the cache, not the embedding -- the miss path retrieves with that
    vector, so there is nothing to save by withholding it.
    """
    gateway = make_gateway()

    first = gateway.process("How long must KYC records be kept?", skip_cache=True)
    second = gateway.process("How long must KYC records be kept?", skip_cache=True)

    assert first.status == second.status == "BYPASS"
    assert first.answer == "KYC records must be retained for 5 years."
    assert len(gateway.cache) == 0, "a bypassed request must not populate the cache"
    assert gateway.stats()["metrics"]["bypasses"] == 2
