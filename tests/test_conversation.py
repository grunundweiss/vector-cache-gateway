"""A turn in a conversation is not always a question.

"What about the pro plan?" has no subject. Embedded alone it sits near every
other elliptical follow-up ever asked, about anything. These tests cover both
halves: that a dependent turn picks up its context, and that an independent one
is left alone -- because prepending context to everything makes every question
in a session look alike, which is its own false-hit generator.
"""

import pytest

from gateway.conversation import (
    ContextWindowResolver,
    RewriteResolver,
    Turn,
    VerbatimResolver,
    looks_dependent,
    turns_from_messages,
)
from gateway.semantic_cache import SemanticCacheGateway
from gateway.vector_store import LocalVectorStore

HISTORY = [
    Turn("user", "What's included in the Basic plan?"),
    Turn("assistant", "The Basic plan includes 5 seats and email support."),
]


@pytest.mark.parametrize(
    "query",
    [
        "What about the pro plan?",
        "Is it refundable?",
        "And the retention period?",
        "the pro plan?",
        "Why?",
    ],
)
def test_elliptical_turns_are_detected_as_dependent(query):
    assert looks_dependent(query)


@pytest.mark.parametrize(
    "query",
    [
        "How do I cancel my subscription?",
        "How long must KYC records be kept?",
        "What is the retention period for customer records?",
    ],
)
def test_self_contained_questions_are_not_dependent(query):
    assert not looks_dependent(query)


def test_dependent_turn_is_embedded_with_its_context():
    resolved = ContextWindowResolver().resolve("What about the pro plan?", HISTORY)

    assert resolved.text == "What's included in the Basic plan? What about the pro plan?"
    assert resolved.query == "What about the pro plan?"
    assert resolved.context_turns == 1


def test_independent_turn_is_left_alone():
    resolved = ContextWindowResolver().resolve("How do I cancel my subscription?", HISTORY)

    assert resolved.text == "How do I cancel my subscription?"
    assert resolved.context_turns == 0


def test_cache_key_is_the_resolved_text_so_the_guard_sees_it():
    """The guard compares strings. If it saw only "what about the pro plan?"
    it would have nothing to compare, which is how a follow-up about one plan
    gets answered with another plan's pricing."""
    resolved = ContextWindowResolver().resolve("What about the pro plan?", HISTORY)

    assert resolved.cache_key == resolved.text


def test_assistant_turns_are_excluded_by_default():
    """An assistant turn is usually far longer than the question and would
    dominate the embedding it was meant to disambiguate."""
    resolved = ContextWindowResolver(window=2).resolve("What about it?", HISTORY)

    assert "email support" not in resolved.text


def test_assistant_turns_can_be_included_and_are_truncated():
    resolved = ContextWindowResolver(
        window=2, include_assistant=True, assistant_chars=10
    ).resolve("What about it?", HISTORY)

    assert "The Basic " in resolved.text
    assert "email support" not in resolved.text


def test_verbatim_resolver_changes_nothing():
    resolved = VerbatimResolver().resolve("What about the pro plan?", HISTORY)

    assert resolved.text == "What about the pro plan?"


def test_rewrite_resolver_uses_its_rewriter():
    resolver = RewriteResolver(lambda query, history: "What is included in the Pro plan?")

    resolved = resolver.resolve("What about the pro plan?", HISTORY)

    assert resolved.text == "What is included in the Pro plan?"
    assert resolved.query == "What about the pro plan?"


def test_rewrite_resolver_falls_back_rather_than_failing_the_request():
    """A broken rewriter should cost hit rate, not availability."""

    def broken(query, history):
        raise RuntimeError("model unavailable")

    resolved = RewriteResolver(broken).resolve("What about the pro plan?", HISTORY)

    assert resolved.text.endswith("What about the pro plan?")
    assert resolved.strategy == "context-window"


def test_empty_rewrite_falls_back_too():
    resolved = RewriteResolver(lambda query, history: "  ").resolve("What about it?", HISTORY)

    assert resolved.strategy == "context-window"


def test_turns_from_messages_drops_system_prompts():
    """A system prompt is identical on every request. Embedding it pushes every
    query toward every other one."""
    history, query = turns_from_messages(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's included in the Basic plan?"},
            {"role": "assistant", "content": "5 seats."},
            {"role": "user", "content": "What about the pro plan?"},
        ]
    )

    assert [t.role for t in history] == ["user", "assistant"]
    assert query == "What about the pro plan?"


def test_turns_from_messages_reports_no_query_without_a_user_turn():
    _, query = turns_from_messages([{"role": "assistant", "content": "anything"}])

    assert query == ""


def make_gateway(resolver):
    store = LocalVectorStore()
    store.ingest_document("doc-1", "Retention for KYC records is 5 years.")
    store.ingest_document("doc-2", "Retention for AML records is 7 years.")
    return SemanticCacheGateway(
        vector_store=store, similarity_threshold=0.75, resolver=resolver
    )


def test_identical_follow_ups_in_different_conversations_collide_without_a_resolver():
    """The failure this module exists to prevent, demonstrated.

    Two conversations about different regulations, the same elliptical
    follow-up. Embedded literally, the second turn is the *same string* as the
    first -- similarity 1.0 -- so the cache serves the first conversation's
    answer to the second conversation's question. Nothing about the threshold or
    the guard can help: the strings really are identical.
    """
    gateway = make_gateway(resolver=None)

    gateway.process("And the limit?", history=[Turn("user", "What about KYC records?")])
    result = gateway.process(
        "And the limit?", history=[Turn("user", "What about AML records?")]
    )

    assert result.status == "HIT"
    assert result.similarity == pytest.approx(1.0, abs=1e-6)


def test_a_resolver_keeps_those_follow_ups_apart():
    gateway = make_gateway(resolver="context")

    gateway.process("And the limit?", history=[Turn("user", "What about KYC records?")])
    result = gateway.process(
        "And the limit?", history=[Turn("user", "What about AML records?")]
    )

    assert result.status == "MISS", (
        "the second conversation's follow-up must not be answered from the first "
        "conversation's cache entry"
    )
