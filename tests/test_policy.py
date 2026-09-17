"""Not every question is equally dangerous to get wrong.

The sweep's highest-scoring wrong pairs hinge on a number or a polarity word.
Those are also the questions where a confident wrong answer does the most
damage, so they get a higher bar. See gateway/policy.py for what is measured
here and what is policy.
"""

import json

import pytest

from gateway.policy import (
    DEFAULT_INTENT_THRESHOLDS,
    GlobalThreshold,
    IntentThresholds,
    classify,
    resolve_policy,
)
from gateway.semantic_cache import SemanticCacheGateway
from gateway.vector_store import LocalVectorStore


@pytest.mark.parametrize(
    "query,intent",
    [
        ("Do transfers over 500,000 NOK need MFA?", "numeric"),
        ("How long must records be kept for five years?", "numeric"),
        ("What is the maximum retention period for marketing data?", "polarity"),
        ("Is MFA required for internal transfers?", "polarity"),
        ("How long must KYC records be kept?", "entity"),
        ("Who owns the incident response process?", "general"),
    ],
)
def test_queries_are_classified_by_what_makes_them_risky(query, intent):
    assert classify(query) == intent


def test_numeric_queries_get_the_strictest_bar():
    policy = IntentThresholds()

    numeric = policy.threshold_for("Do transfers over 500,000 NOK need MFA?")
    general = policy.threshold_for("Who owns the incident response process?")

    assert numeric > general
    assert general == DEFAULT_INTENT_THRESHOLDS["general"]


def test_explain_reports_the_decision_it_made():
    assert IntentThresholds().explain("What is the maximum retention period?") == (
        "polarity",
        DEFAULT_INTENT_THRESHOLDS["polarity"],
    )


def test_global_threshold_is_the_default_policy():
    policy = resolve_policy(None, 0.85)

    assert isinstance(policy, GlobalThreshold)
    assert policy.threshold_for("anything at all") == 0.85


def test_intent_policy_is_opt_in_by_name():
    assert isinstance(resolve_policy("intent", 0.85), IntentThresholds)


def test_thresholds_can_be_loaded_from_a_measured_sweep(tmp_path):
    """The point of `benchmarks/intent_sweep.py`: replace the shipped policy
    with numbers measured on your own labeled pairs."""
    path = tmp_path / "intent_thresholds.json"
    path.write_text(json.dumps({"thresholds": {"numeric": 0.99, "general": 0.80}}))

    policy = IntentThresholds.from_json(path)

    assert policy.threshold_for("over 500,000 NOK?") == 0.99
    assert policy.threshold_for("Who owns incident response?") == 0.80


def test_unlisted_intent_falls_back_to_the_default():
    policy = IntentThresholds({"numeric": 0.99}, default=0.85)

    assert policy.threshold_for("Who owns the incident response process?") == 0.85


def test_gateway_applies_the_per_intent_bar():
    """A pair that clears the global threshold but not the numeric one."""
    store = LocalVectorStore()
    store.ingest_document("doc-1", "Transfers over 500,000 NOK require approval.")

    permissive = SemanticCacheGateway(
        vector_store=store, similarity_threshold=0.75, guard=None
    )
    permissive.process_query("Do transfers over 500,000 NOK need sign off?")
    assert permissive.process_query("Do transfers over 500,000 NOK need approval?")[1].startswith(
        "CACHE_HIT"
    )

    strict = SemanticCacheGateway(
        vector_store=store,
        similarity_threshold=0.75,
        guard=None,
        threshold_policy="intent",
    )
    strict.process_query("Do transfers over 500,000 NOK need sign off?")
    assert strict.process_query("Do transfers over 500,000 NOK need approval?")[1] == "CACHE_MISS"
