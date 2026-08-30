"""A cache hit must skip generation, not just retrieval."""

import pytest

from gateway.generator import InferenceEngineGenerator, RetrievalOnlyGenerator
from gateway.semantic_cache import SemanticCacheGateway
from gateway.vector_store import LocalVectorStore

DOC = "KYC records must be retained for a minimum of 5 years."


class SpyEngine:
    """Stands in for an LLM serving engine, counting generation calls."""

    def __init__(self):
        self.calls = 0

    def generate_inference(self, prompt, max_tokens=128):
        self.calls += 1
        return f"generated answer #{self.calls}"


@pytest.fixture
def store():
    store = LocalVectorStore()
    store.ingest_document("doc-1", DOC)
    return store


def test_default_generator_returns_retrieved_text(store):
    gateway = SemanticCacheGateway(vector_store=store, similarity_threshold=0.75)

    answer, status = gateway.process_query("How long must KYC records be kept?")

    assert status == "CACHE_MISS"
    assert answer == DOC
    assert isinstance(gateway.generator, RetrievalOnlyGenerator)


def test_cache_hit_skips_generation(store):
    engine = SpyEngine()
    gateway = SemanticCacheGateway(
        vector_store=store,
        similarity_threshold=0.75,
        generator=InferenceEngineGenerator(engine),
    )

    first, first_status = gateway.process_query("How long must KYC records be kept?")
    assert first_status == "CACHE_MISS"
    assert engine.calls == 1

    second, second_status = gateway.process_query("How long must KYC records be retained?")

    assert second_status.startswith("CACHE_HIT")
    assert engine.calls == 1, "a cache hit must not invoke the generator"
    assert second == first


def test_miss_invokes_generation(store):
    engine = SpyEngine()
    gateway = SemanticCacheGateway(
        vector_store=store,
        similarity_threshold=0.75,
        generator=InferenceEngineGenerator(engine),
    )

    gateway.process_query("How long must KYC records be kept?")
    gateway.process_query("Who approves KYC exemptions?")

    assert engine.calls == 2


def test_adapter_rejects_object_without_generate_inference():
    with pytest.raises(TypeError):
        InferenceEngineGenerator(object())
