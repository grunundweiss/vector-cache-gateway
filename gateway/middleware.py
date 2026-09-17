"""A cache that sits in front of a chat endpoint, rather than beside one.

Everything else in this repo is a library you call. This is the shape you
actually deploy: intercept the request, look it up, return immediately on a hit,
call the model on a miss and store what comes back. It is transport-agnostic on
purpose -- it wraps a function, so the same object works behind FastAPI,
Starlette, a Lambda handler, or a test with a fake. See `examples/fastapi_proxy.py`
for the HTTP wiring.

Two things trip people up here, and both are handled below:

**Streaming.** A cached answer has no upstream stream to forward. Returning it
as a single blob breaks clients that expect chunks, so a hit is re-chunked into
a synthetic stream. It arrives all at once because it is already complete --
that is the saving, visible.

**Tool calls.** A turn that calls a tool is not a question with a stable answer;
it is a request to do something with live state. "Cancel my order" resolves
against whichever order exists right now. Caching either the tool call or its
result serves a stale decision about a changed world, so any request that offers
tools, contains a tool result, or comes back as a tool call is never cached --
and the bypass is counted, so the metrics show how much traffic the cache never
saw rather than hiding it in the miss rate.
"""

import logging
import time
from collections.abc import Iterable, Iterator

from gateway.conversation import turns_from_messages
from gateway.engine import CacheResult, MissOutcome, SemanticCacheEngine
from gateway.partition import DEFAULT_NAMESPACE

logger = logging.getLogger(__name__)

# Request keys whose presence means the model may decide to call a tool.
_TOOL_KEYS = ("tools", "functions", "tool_choice", "function_call")


def is_cacheable(messages, **kwargs) -> tuple[bool, str]:
    """Whether this request may be served from, or written to, the cache.

    Returns (cacheable, reason). The reason is recorded as a bypass, because
    "why is the hit rate low" is answerable only if the traffic that was never
    eligible is visible separately from the traffic that missed.
    """
    if kwargs.get("no_cache"):
        return False, "no_cache"
    if any(kwargs.get(key) for key in _TOOL_KEYS):
        return False, "tools_offered"
    for message in messages:
        role = message.get("role")
        if role in ("tool", "function"):
            return False, "tool_result"
        if message.get("tool_calls") or message.get("function_call"):
            return False, "tool_call"
        if not isinstance(message.get("content"), str) and role in ("user", "assistant"):
            # Content that is a list of parts means images or audio; the text
            # embedding does not represent it, so a match on the text alone
            # would be a match on the wrong thing.
            return False, "multimodal"
    if not messages or messages[-1].get("role") != "user":
        return False, "no_user_turn"
    return True, ""


class ChatCacheMiddleware:
    """Semantic cache around a chat-completions style function.

    ``chat_fn(messages, **kwargs) -> str`` is the upstream call on a miss.
    ``stream_fn(messages, **kwargs) -> Iterator[str]`` is its streaming
    equivalent, required only if `stream` is used.

    ``namespace_fn(messages, kwargs) -> str`` decides the cache partition. The
    default is a single shared namespace, which is correct only when answers
    depend on nothing user-specific. Anything that touches account state should
    pass a function returning the user or tenant id -- see gateway/partition.py
    for why that is a data-leak boundary rather than a tuning parameter.
    """

    def __init__(
        self,
        engine: SemanticCacheEngine,
        chat_fn,
        *,
        stream_fn=None,
        namespace_fn=None,
        ttl: float | None = None,
        chunk_words: int = 8,
    ):
        self.engine = engine
        self.chat_fn = chat_fn
        self.stream_fn = stream_fn
        self.namespace_fn = namespace_fn
        self.ttl = ttl
        self.chunk_words = chunk_words

    @property
    def metrics(self):
        return self.engine.metrics

    def complete(self, messages, **kwargs) -> CacheResult:
        """Answers a chat request, from cache when it can.

        Returns a `CacheResult` rather than a bare string: the caller needs the
        entry id to attach feedback to, and the status to put in a response
        header. A proxy that returns only text cannot tell its client -- or its
        own dashboard -- which answers were cached.
        """
        cacheable, reason = is_cacheable(messages, **kwargs)
        namespace = self._namespace(messages, kwargs)
        history, query = turns_from_messages(messages)

        if not cacheable:
            logger.debug("bypassing cache: %s", reason)
            self.metrics.record_bypass(reason)
            answer = self.chat_fn(messages, **self._upstream_kwargs(kwargs))
            return CacheResult(answer, "BYPASS", namespace=namespace)

        def on_miss(prepared) -> MissOutcome:
            answer = self.chat_fn(messages, **self._upstream_kwargs(kwargs))
            # An empty completion is a failure, not an answer. Caching it would
            # serve the failure to everyone who asks a similar question next.
            return MissOutcome(answer, cacheable=bool(answer and answer.strip()))

        return self.engine.answer(
            query, on_miss, history=history, namespace=namespace, ttl=self.ttl
        )

    def stream(self, messages, **kwargs) -> Iterator[str]:
        """Yields the answer in chunks, from cache when it can.

        A cache hit is re-chunked into a synthetic stream so the client sees the
        same shape of response either way. A miss forwards the upstream stream
        chunk by chunk and stores the assembled text **only if the stream
        finishes** -- a response the client disconnected from halfway is a
        truncated answer, and caching it would serve that truncation to everyone
        who asks a similar question next.
        """
        if self.stream_fn is None:
            raise RuntimeError("stream() needs a stream_fn; none was configured")

        cacheable, reason = is_cacheable(messages, **kwargs)
        namespace = self._namespace(messages, kwargs)
        history, query = turns_from_messages(messages)
        upstream_kwargs = self._upstream_kwargs(kwargs)

        if not cacheable:
            logger.debug("bypassing cache: %s", reason)
            self.metrics.record_bypass(reason)
            yield from self.stream_fn(messages, **upstream_kwargs)
            return

        started = time.perf_counter()
        prepared = self.engine.prepare(query, history=history, namespace=namespace)
        hit = self.engine.lookup(prepared)
        if hit is not None:
            self.metrics.record_hit(hit.score, (time.perf_counter() - started) * 1000)
            yield from self._fake_stream(hit.answer)
            return

        chunks: list[str] = []
        completed = False
        try:
            for chunk in self.stream_fn(messages, **upstream_kwargs):
                chunks.append(chunk)
                yield chunk
            completed = True
        finally:
            answer = "".join(chunks)
            if completed and answer.strip():
                self.engine.store(prepared, answer, ttl=self.ttl)
            elif not completed:
                logger.info("upstream stream did not complete; not caching %d chunks", len(chunks))
            self.metrics.record_miss((time.perf_counter() - started) * 1000)

    def feedback(self, entry_id: str, helpful: bool = False) -> bool:
        """Evicts the entry behind an answer the user rejected."""
        return self.engine.feedback(entry_id, helpful)

    def stats(self) -> dict:
        return self.engine.stats()

    def _fake_stream(self, answer: str) -> Iterable[str]:
        words = answer.split(" ")
        for i in range(0, len(words), self.chunk_words):
            chunk = " ".join(words[i : i + self.chunk_words])
            yield chunk if i + self.chunk_words >= len(words) else chunk + " "

    def _namespace(self, messages, kwargs) -> str:
        if self.namespace_fn is not None:
            return self.namespace_fn(messages, kwargs)
        return str(kwargs.get("namespace") or DEFAULT_NAMESPACE)

    def _upstream_kwargs(self, kwargs: dict) -> dict:
        """Strips the keys this middleware owns before calling upstream."""
        return {k: v for k, v in kwargs.items() if k not in ("no_cache", "namespace")}
