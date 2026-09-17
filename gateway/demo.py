"""Runnable demo of every gate the cache puts in front of an answer.

    python -m gateway.demo

Downloads all-mpnet-base-v2 on first run. Five sections, each one a failure mode
this repo measured before it fixed it:

    1. the similarity gate      a paraphrase hits, an unrelated question misses
    2. the guard                a near-miss clears 0.85 and is vetoed anyway
    3. conversation             an elliptical follow-up resolved against context
    4. tenancy                  two users, the same question, different answers
    5. freshness                an answer that expires on a clock
"""

import logging
import time

from gateway.cache import DEFAULT_THRESHOLD
from gateway.conversation import Turn
from gateway.guard import default_guard
from gateway.semantic_cache import SemanticCacheGateway
from gateway.vector_store import LocalVectorStore

RULE = "=" * 74


def banner(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def show(gateway, query: str, note: str, **kwargs) -> None:
    result = gateway.process(query, **kwargs)
    similarity = f" ({result.similarity:.4f})" if result.similarity is not None else ""
    print(f"\n{note}\n  query:  {query}\n  status: {result.status}{similarity}")
    print(f"  answer: {result.answer}")


def build_store() -> LocalVectorStore:
    store = LocalVectorStore()
    store.ingest_document(
        "doc-1",
        "Handelsbanken protocol dictates strict multi-factor authentication "
        "for transfers over 500,000 NOK.",
    )
    store.ingest_document(
        "doc-2",
        "KYC compliance data records must be preserved locally for a minimum "
        "retention window of 5 years.",
    )
    return store


def demo_similarity_gate() -> None:
    banner(f"1. the similarity gate (threshold {DEFAULT_THRESHOLD})")
    gateway = SemanticCacheGateway(vector_store=build_store())

    show(gateway, "What is the retention limit for KYC records?", "nothing cached yet")
    show(gateway, "What is the retention period for KYC records?", "paraphrase -- should hit")
    show(gateway, "What is the capital of France?", "unrelated -- must not hit")


def demo_guard() -> None:
    banner("2. the guard: what the threshold alone cannot stop")
    print(
        "\nThe sweep measured these two at 0.9792 similarity -- above every genuine\n"
        "paraphrase in the labeled set. They differ by a factor of ten in NOK."
    )
    a = "Do transfers over 500,000 NOK need multi-factor authentication?"
    b = "Do transfers over 50,000 NOK need multi-factor authentication?"

    decision = default_guard().approve(a, b, 0.9792)
    print(f"\n  cached:  {a}\n  query:   {b}\n  guard:   {'approved' if decision.ok else 'VETOED'}")
    print(f"  reason:  {decision.reason}")

    gateway = SemanticCacheGateway(vector_store=build_store())
    show(gateway, a, "first question, cached")
    show(gateway, b, "different amount -- must not be served the cached answer")


def demo_conversation() -> None:
    banner("3. conversation: a follow-up is not a question")
    gateway = SemanticCacheGateway(vector_store=build_store(), resolver="context")

    kyc = [Turn("user", "What is the retention period for KYC records?")]
    mfa = [Turn("user", "What is the MFA threshold for large transfers?")]

    print("\nThe same four words, in two conversations about different things.")
    show(gateway, "And the limit?", "after a question about KYC", history=kyc)
    show(gateway, "And the limit?", "after a question about MFA", history=mfa)
    print(
        "\nEmbedded literally these are the same string, similarity 1.0, and the\n"
        "second would be served the first's answer. The resolver embeds each one\n"
        "with the turn it depends on instead."
    )


def demo_tenancy() -> None:
    banner("4. tenancy: the same question, two users")
    gateway = SemanticCacheGateway(vector_store=build_store())
    question = "What is the retention limit for KYC records?"

    show(gateway, question, "alice asks", namespace="alice")
    show(gateway, question, "bob asks the identical question", namespace="bob")
    print(
        "\nA miss, not a hit. Partitions are enforced by construction: alice's\n"
        "vectors and bob's are never in the same matrix, so no code path can\n"
        "compare them. That costs a duplicate entry and removes a class of leak."
    )


def demo_freshness() -> None:
    banner("5. freshness: an answer with a deadline")
    gateway = SemanticCacheGateway(vector_store=build_store())
    question = "What is the retention limit for KYC records?"

    show(gateway, question, "cached with a 2 second ttl", ttl=2)
    show(gateway, question, "immediately after -- still fresh", ttl=2)
    print("\n  ... waiting 2.1s ...")
    time.sleep(2.1)
    show(gateway, question, "after the ttl -- regenerated", ttl=2)
    print(
        "\nCorpus-change invalidation would not have caught this: no document\n"
        "changed. Some answers are only true for a while."
    )

    print("\nmetrics for this section's gateway (each section builds its own):")
    print(gateway.metrics.render())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("gateway").setLevel(logging.WARNING)

    demo_similarity_gate()
    demo_guard()
    demo_conversation()
    demo_tenancy()
    demo_freshness()

    print(f"\n{RULE}")
    print(
        "Every number quoted above comes from benchmarks/, not from this file.\n"
        "See 'Choosing the threshold' in the README for the sweep behind 0.85\n"
        "and results/guard_eval.json for what the guard catches."
    )


if __name__ == "__main__":
    main()
