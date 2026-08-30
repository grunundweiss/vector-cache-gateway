"""What produces an answer on a cache miss.

The cache sits in front of whatever this is. What a cache hit actually saves
depends entirely on which generator is installed:

``RetrievalOnlyGenerator``   a hit saves one embedding pass and one vector
                             search. Real, but small -- microseconds.
``InferenceEngineGenerator`` a hit saves an LLM generation call. That is the
                             two-orders-of-magnitude saving people mean when
                             they say a semantic cache "reduces inference
                             costs".

The default is retrieval-only, so the repo runs standalone with no model
server. The README states which one its numbers were measured against.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Generator(Protocol):
    """Produces an answer from a query and retrieved context."""

    def generate(self, query: str, context: str) -> str: ...


class RetrievalOnlyGenerator:
    """Returns the retrieved passage verbatim.

    No model and no inference cost, so a cache hit here saves retrieval work
    only. This is what the repo does out of the box.
    """

    def generate(self, query: str, context: str) -> str:
        return context


class InferenceEngineGenerator:
    """Adapter for an external LLM serving engine.

    Deliberately duck-typed rather than importing a specific project: it wraps
    any object exposing ``generate_inference(prompt, ...) -> str``, which is the
    interface exposed by the sibling ``secure-ai-platform`` inference engine.
    Passing one in is what makes the "saves an inference call" claim true.

        from model_platform.inference_engine import InferenceEngine
        gateway = SemanticCacheGateway(
            vector_store=store,
            generator=InferenceEngineGenerator(InferenceEngine()),
        )
    """

    def __init__(self, engine, max_tokens: int = 128):
        if not hasattr(engine, "generate_inference"):
            raise TypeError(
                "engine must expose generate_inference(prompt); "
                f"got {type(engine).__name__}"
            )
        self.engine = engine
        self.max_tokens = max_tokens

    def generate(self, query: str, context: str) -> str:
        prompt = (
            "Answer the question using only the context below.\n\n"
            f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        )
        return self.engine.generate_inference(prompt, max_tokens=self.max_tokens)
