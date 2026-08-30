"""Runnable demo: a paraphrase hit, and a near-miss that should not hit.

    python -m gateway.demo

Downloads all-mpnet-base-v2 on first run.
"""

import logging

from gateway.cache import DEFAULT_THRESHOLD
from gateway.semantic_cache import SemanticCacheGateway
from gateway.vector_store import LocalVectorStore

QUERIES = [
    ("first query, nothing cached yet", "What is the retention limit for KYC records?"),
    ("paraphrase of the above -- should hit", "What is the retention period for KYC records?"),
    ("different regulation -- must not hit", "What is the retention limit for AML records?"),
    ("unrelated question", "What is the capital of France?"),
]


def main() -> None:
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

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

    gateway = SemanticCacheGateway(vector_store=store, similarity_threshold=DEFAULT_THRESHOLD)
    print(f"threshold: {DEFAULT_THRESHOLD}\n" + "=" * 70)

    for note, query in QUERIES:
        answer, status = gateway.process_query(query)
        print(f"\n{note}\n  query:  {query}\n  status: {status}\n  answer: {answer}")

    print("\n" + "=" * 70)
    print(
        "The third query asks about a different regulation. Whether it was served\n"
        "from cache is the point of benchmarks/threshold_sweep.py -- see the\n"
        "'Choosing the threshold' section of the README."
    )


if __name__ == "__main__":
    main()
