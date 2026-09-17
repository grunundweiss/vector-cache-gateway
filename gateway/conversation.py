"""Turning a turn in a conversation into something worth embedding.

A cache that embeds the literal user message assumes every message is a whole
question. In a chat assistant most are not:

    user: What's included in the Basic plan?
    user: What about the pro plan?

The second message, embedded alone, is four words with no subject. It will land
near every other "what about X" ever asked, in every conversation, about every
topic -- and the cache will serve one of them. The failure is not that the
similarity is wrong; it is that the string being compared is not the question.

Two ways to fix it, both here:

``ContextWindowResolver``  prepend recent turns to the text that gets embedded.
                           No model, no latency, and it works because the
                           missing subject is usually one or two turns back.
``RewriteResolver``        hand the history to something that rewrites the turn
                           into a standalone question. Better, and it costs a
                           model call -- see the warning on that class.

Whatever is resolved becomes both the embedded text *and* the cache key, so the
guard in gateway/guard.py compares resolved strings too. That matters: "what
about the pro plan" and "what about the basic plan" are near-identical, while
"What's included in the Basic plan? What about the pro plan?" and its Basic
counterpart differ by a word the polarity lexicon knows about.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Words that point at something said earlier instead of naming it.
_ANAPHORA = re.compile(
    r"\b(it|its|it's|that|this|these|those|they|them|their|there|he|she|his|her|"
    r"one|ones|same|above|previous|instead)\b",
    re.IGNORECASE,
)

# Openers that continue a thread rather than start one.
_CONTINUATIONS = (
    "what about", "how about", "and what", "and how", "and the", "what if",
    "why not", "why", "and", "but", "so", "then", "ok", "okay", "also",
)

_WORD = re.compile(r"[a-zA-Z0-9]+")

_STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "do", "does", "did", "what",
        "which", "who", "how", "when", "where", "why", "can", "could", "should",
        "would", "must", "may", "i", "we", "you", "of", "for", "to", "in", "on",
        "at", "by", "with", "about", "and", "or", "but", "if", "so", "then",
        "my", "me", "us", "your", "our",
    }
)


@dataclass(frozen=True)
class Turn:
    """One message. ``role`` follows the chat convention: user or assistant."""

    role: str
    content: str


@dataclass(frozen=True)
class ResolvedQuery:
    """What the cache should actually work with.

    ``text`` is embedded and stored as the cache key -- deliberately the same
    string, so that the vector and the guard agree on what the question was.
    ``query`` keeps the user's literal message, which is what a generator on the
    miss path should still see.
    """

    text: str
    query: str
    strategy: str = "verbatim"
    dependent: bool = False
    context_turns: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def cache_key(self) -> str:
        return self.text


@runtime_checkable
class QueryResolver(Protocol):
    """Decides what text represents this turn for cache purposes."""

    name: str

    def resolve(self, query: str, history=()) -> ResolvedQuery: ...


def content_words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text) if w.lower() not in _STOPWORDS]


def looks_dependent(query: str, fragment_words: int = 4) -> bool:
    """Whether this turn appears to lean on what came before.

    Three signals, any of which is enough: a pronoun pointing at something not
    in this sentence, an opener that continues a thread, or a fragment too short
    to be a question by itself. Length is the weakest of the three and is
    therefore the narrowest: "How do I cancel my subscription?" is six words and
    stands alone, while "the pro plan?" is three and cannot.

    All three are heuristics. Calling a standalone question dependent costs a
    cache miss; missing a dependent one means looking up an under-specified
    string, which is the error that can serve a wrong answer. The defaults lean
    toward the first.
    """
    lowered = query.strip().lower()
    if _ANAPHORA.search(lowered):
        return True
    if any(lowered.startswith(opener + " ") or lowered == opener for opener in _CONTINUATIONS):
        return True
    words = _WORD.findall(lowered)
    return len(words) <= fragment_words or len(content_words(lowered)) <= 1


class VerbatimResolver:
    """Embeds the message exactly as sent.

    Correct for a single-shot question-answering API where every request stands
    alone, and the thing that quietly breaks a chat assistant. It is the default
    only because it is what the gateway did before resolvers existed.
    """

    name = "verbatim"

    def resolve(self, query: str, history=()) -> ResolvedQuery:
        return ResolvedQuery(text=query, query=query, strategy=self.name)


class ContextWindowResolver:
    """Prepends recent turns when the message looks like it needs them.

    Conditional rather than unconditional, because always prepending is its own
    failure: two unrelated questions asked in the same conversation share the
    same prefix, so their embeddings converge and the cache starts serving the
    first answer for the second question. Concatenating context makes
    *everything in one session* look similar, which is the opposite of what a
    cache shared across sessions needs.

    So context is added only when `looks_dependent` says the turn cannot stand
    alone, and only from user turns by default -- an assistant turn is usually
    an order of magnitude longer than the question and would dominate the
    embedding.
    """

    name = "context-window"

    def __init__(
        self,
        window: int = 2,
        *,
        include_assistant: bool = False,
        assistant_chars: int = 200,
        always: bool = False,
        fragment_words: int = 4,
    ):
        self.window = window
        self.include_assistant = include_assistant
        self.assistant_chars = assistant_chars
        self.always = always
        self.fragment_words = fragment_words

    def resolve(self, query: str, history=()) -> ResolvedQuery:
        dependent = self.always or looks_dependent(query, self.fragment_words)
        if not dependent or not history:
            return ResolvedQuery(
                text=query, query=query, strategy=self.name, dependent=dependent
            )

        selected: list[str] = []
        for turn in reversed(list(history)):
            if len(selected) >= self.window:
                break
            if turn.role == "user":
                selected.append(turn.content)
            elif self.include_assistant:
                selected.append(turn.content[: self.assistant_chars])
        selected.reverse()

        text = " ".join([*selected, query]).strip()
        logger.debug("resolved dependent turn %r using %d prior turns", query, len(selected))
        return ResolvedQuery(
            text=text,
            query=query,
            strategy=self.name,
            dependent=True,
            context_turns=len(selected),
        )


class RewriteResolver:
    """Delegates to something that rewrites the turn into a standalone question.

    This is the better answer to the follow-up problem and the one with a bill
    attached. If the rewrite is an LLM call of the same size as the generation
    the cache is meant to skip, the cache saves nothing -- it has simply moved
    the model call earlier. It pays for itself only when the rewriter is much
    cheaper than the thing behind the cache: a small local model, or a
    rule-based rewriter, against an expensive generation.

    ``rewrite_fn(query, history) -> str``. A rewriter that raises or returns
    nothing falls back rather than failing the request: a degraded cache key is
    a worse hit rate, while an exception is a failed user request.

        gateway = SemanticCacheGateway(
            vector_store=store,
            resolver=RewriteResolver(my_small_model.rewrite),
        )
    """

    name = "rewrite"

    def __init__(self, rewrite_fn, fallback: QueryResolver | None = None):
        self.rewrite_fn = rewrite_fn
        self.fallback = fallback or ContextWindowResolver()

    def resolve(self, query: str, history=()) -> ResolvedQuery:
        history = list(history)
        if not history:
            return ResolvedQuery(text=query, query=query, strategy=self.name)
        try:
            rewritten = self.rewrite_fn(query, history)
        except Exception:  # noqa: BLE001 - a broken rewriter must not fail the request
            logger.exception("query rewriter failed, falling back to %s", self.fallback.name)
            return self.fallback.resolve(query, history)
        if not rewritten or not rewritten.strip():
            logger.warning("query rewriter returned nothing, falling back")
            return self.fallback.resolve(query, history)
        return ResolvedQuery(
            text=rewritten.strip(),
            query=query,
            strategy=self.name,
            dependent=True,
            context_turns=len(history),
        )


def turns_from_messages(messages) -> tuple[list[Turn], str]:
    """Splits chat-completions style messages into (history, current query).

    Accepts the OpenAI shape -- ``{"role": ..., "content": ...}`` -- because that
    is what a proxy in front of a chat endpoint receives. System messages are
    dropped: they are prompt configuration, identical across every request, and
    including them pushes every query's embedding toward every other one.
    """
    turns = [
        Turn(str(m.get("role", "")), m["content"])
        for m in messages
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
    ]
    if not turns or turns[-1].role != "user":
        return turns, ""
    return turns[:-1], turns[-1].content


def resolve_resolver(resolver) -> QueryResolver:
    """Normalizes the constructor argument callers pass.

    ``None`` keeps verbatim embedding, ``"context"`` installs the windowed
    resolver with its defaults, and anything else is used as given.
    """
    if resolver is None:
        return VerbatimResolver()
    if resolver == "context":
        return ContextWindowResolver()
    if resolver == "verbatim":
        return VerbatimResolver()
    return resolver
