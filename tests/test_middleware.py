"""The proxy shape: intercept, look up, return or call the model.

The two things that make a semantic cache wrong in front of a real chat endpoint
are both here: a cache hit has to still look like a stream, and a turn that
touches live state must never be cached at all.
"""

import pytest

from gateway.engine import SemanticCacheEngine
from gateway.middleware import ChatCacheMiddleware, is_cacheable
from gateway.vector_store import LocalVectorStore

QUESTION = "How long must KYC records be kept?"
PARAPHRASE = "How long must KYC records be retained?"


class SpyChat:
    """Stands in for the upstream model, counting calls."""

    def __init__(self, reply="records are kept for five years"):
        self.calls = 0
        self.reply = reply

    def __call__(self, messages, **kwargs):
        self.calls += 1
        return self.reply

    def stream(self, messages, **kwargs):
        self.calls += 1
        for word in self.reply.split(" "):
            yield word + " "


def user(text):
    return [{"role": "user", "content": text}]


@pytest.fixture
def chat():
    return SpyChat()


@pytest.fixture
def middleware(chat):
    engine = SemanticCacheEngine(LocalVectorStore(), threshold=0.75)
    return ChatCacheMiddleware(engine, chat, stream_fn=chat.stream)


def test_first_request_calls_the_model(middleware, chat):
    result = middleware.complete(user(QUESTION))

    assert result.status == "MISS"
    assert chat.calls == 1


def test_paraphrase_is_served_from_cache_without_calling_the_model(middleware, chat):
    middleware.complete(user(QUESTION))

    result = middleware.complete(user(PARAPHRASE))

    assert result.status == "HIT"
    assert chat.calls == 1, "a cache hit must not reach the model"
    assert result.entry_id


def test_empty_completion_is_not_cached():
    """An empty response is a failure. Caching it serves the failure to everyone."""
    chat = SpyChat(reply="")
    engine = SemanticCacheEngine(LocalVectorStore(), threshold=0.75)
    middleware = ChatCacheMiddleware(engine, chat)

    middleware.complete(user(QUESTION))
    middleware.complete(user(QUESTION))

    assert chat.calls == 2


@pytest.mark.parametrize(
    "messages,kwargs,reason",
    [
        (
            [{"role": "user", "content": "cancel my order"}],
            {"tools": [{"name": "cancel_order"}]},
            "tools_offered",
        ),
        (
            [
                {"role": "user", "content": "cancel my order"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
                {"role": "tool", "content": "cancelled"},
            ],
            {},
            "tool_call",
        ),
        (
            [{"role": "user", "content": [{"type": "image_url", "url": "..."}]}],
            {},
            "multimodal",
        ),
        ([{"role": "user", "content": "hello"}], {"no_cache": True}, "no_cache"),
        ([{"role": "assistant", "content": "hello"}], {}, "no_user_turn"),
    ],
)
def test_requests_that_must_never_be_cached(messages, kwargs, reason):
    cacheable, why = is_cacheable(messages, **kwargs)

    assert not cacheable
    assert why == reason


def test_tool_requests_bypass_the_cache_entirely(middleware, chat):
    """Twice, because a bypass must not populate the cache either.

    "Cancel my order" resolves against whichever order exists right now. A
    cached tool call is a stale decision about a changed world.
    """
    messages = user("cancel my order")

    first = middleware.complete(messages, tools=[{"name": "cancel_order"}])
    second = middleware.complete(messages, tools=[{"name": "cancel_order"}])

    assert first.status == second.status == "BYPASS"
    assert chat.calls == 2
    assert middleware.metrics.snapshot()["bypasses"] == 2


def test_bypasses_are_not_counted_as_misses(middleware):
    middleware.complete(user("cancel my order"), tools=[{"name": "cancel_order"}])

    snapshot = middleware.metrics.snapshot()

    assert snapshot["misses"] == 0
    assert snapshot["bypasses"] == 1


def test_middleware_strips_its_own_kwargs_before_calling_upstream():
    seen = {}

    def chat_fn(messages, **kwargs):
        seen.update(kwargs)
        return "answer"

    engine = SemanticCacheEngine(LocalVectorStore(), threshold=0.75)
    ChatCacheMiddleware(engine, chat_fn).complete(
        user(QUESTION), namespace="tenant-a", temperature=0.2
    )

    assert seen == {"temperature": 0.2}


def test_streaming_miss_forwards_chunks_and_caches_the_assembled_text(middleware, chat):
    chunks = list(middleware.stream(user(QUESTION)))

    assert "".join(chunks).strip() == chat.reply
    assert chat.calls == 1

    again = list(middleware.stream(user(PARAPHRASE)))

    assert "".join(again).strip() == chat.reply
    assert chat.calls == 1, "the second stream must come from cache"


def test_streaming_hit_is_chunked_not_returned_whole(middleware, chat):
    middleware.complete(user(QUESTION))

    chunks = list(middleware.stream(user(PARAPHRASE)))

    assert len(chunks) >= 1
    assert "".join(chunks) == chat.reply


def test_abandoned_stream_is_not_cached(middleware, chat):
    """A response the client disconnected from is truncated, not an answer."""
    stream = middleware.stream(user(QUESTION))
    next(stream)
    stream.close()

    list(middleware.stream(user(QUESTION)))

    assert chat.calls == 2, "the abandoned stream must not have been cached"


def test_stream_without_a_stream_fn_is_an_error(chat):
    engine = SemanticCacheEngine(LocalVectorStore(), threshold=0.75)
    middleware = ChatCacheMiddleware(engine, chat)

    with pytest.raises(RuntimeError, match="stream_fn"):
        list(middleware.stream(user(QUESTION)))


def test_namespace_fn_partitions_the_cache(chat):
    engine = SemanticCacheEngine(LocalVectorStore(), threshold=0.75)
    middleware = ChatCacheMiddleware(
        engine, chat, namespace_fn=lambda messages, kwargs: kwargs.get("user_id", "anon")
    )

    middleware.complete(user(QUESTION), user_id="alice")
    result = middleware.complete(user(QUESTION), user_id="bob")

    assert result.status == "MISS", "bob must not be served alice's cached answer"
    assert chat.calls == 2


def test_negative_feedback_evicts_the_answer_that_was_served(middleware, chat):
    first = middleware.complete(user(QUESTION))

    assert middleware.feedback(first.entry_id, helpful=False)

    middleware.complete(user(QUESTION))
    assert chat.calls == 2, "the rejected answer must not be served again"


def test_follow_up_turns_reach_the_resolver(chat):
    """The middleware has the history; it has to hand it over."""
    engine = SemanticCacheEngine(LocalVectorStore(), threshold=0.75, resolver="context")
    middleware = ChatCacheMiddleware(engine, chat)

    middleware.complete(
        [
            {"role": "user", "content": "What is the KYC retention period?"},
            {"role": "assistant", "content": "Five years."},
            {"role": "user", "content": "And the limit?"},
        ]
    )
    result = middleware.complete(
        [
            {"role": "user", "content": "What is the AML retention period?"},
            {"role": "assistant", "content": "Seven years."},
            {"role": "user", "content": "And the limit?"},
        ]
    )

    assert result.status == "MISS"
