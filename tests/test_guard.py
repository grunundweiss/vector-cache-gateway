"""The verification pass that runs after the threshold and before the answer.

These are the cases `benchmarks/threshold_sweep.py` proved a threshold cannot
handle: pairs that score above 0.85 while asking different questions. Each one
here is taken from that measured set, so a regression in the guard is a
regression against evidence rather than against an opinion.
"""

import pytest

from gateway.guard import (
    AcronymGuard,
    CompositeGuard,
    NullGuard,
    NumericGuard,
    PolarityGuard,
    RerankGuard,
    TokenOverlapGuard,
    default_guard,
)

# (query, cached key, why it must be vetoed). Scores are from results/threshold_sweep.json.
DANGEROUS_PAIRS = [
    (
        "Do transfers over 500,000 NOK need multi-factor authentication?",
        "Do transfers over 50,000 NOK need multi-factor authentication?",
        "a 10x difference in a reporting threshold, scored 0.9792",
    ),
    (
        "What is the maximum retention period for marketing data?",
        "What is the minimum retention period for marketing data?",
        "exact opposites, scored 0.9520",
    ),
    (
        "Is MFA required for internal transfers?",
        "Is MFA required for external transfers?",
        "opposite scope, scored 0.9343",
    ),
    (
        "What are the requirements for onboarding a retail customer?",
        "What are the requirements for onboarding a corporate customer?",
        "different customer class, scored 0.8908",
    ),
    (
        "What is the retention limit for KYC records?",
        "What is the retention limit for AML records?",
        "different regulation, one token apart",
    ),
]

# Genuine paraphrases from the same labeled set. A guard that vetoes these is
# not safe, it is just off.
SAFE_PAIRS = [
    (
        "How long must KYC records be kept?",
        "What is the retention period for KYC records?",
    ),
    (
        "Do transfers over 500,000 NOK need multi-factor authentication?",
        "Is MFA required for transfers above 500,000 NOK?",
    ),
    (
        "Must staff complete compliance training every year?",
        "Is annual compliance training mandatory for employees?",
    ),
    (
        "What records must be kept for cross-border payments?",
        "Which documentation is required for international payment transactions?",
    ),
]


@pytest.mark.parametrize("query,key,why", DANGEROUS_PAIRS)
def test_guard_vetoes_pairs_the_threshold_cannot_separate(query, key, why):
    decision = default_guard().approve(query, key, 0.95)

    assert not decision.ok, f"must veto: {why}"
    assert decision.reason, "a veto has to say why, or nobody can debug a low hit rate"


@pytest.mark.parametrize("query,key", SAFE_PAIRS)
def test_guard_approves_genuine_paraphrases(query, key):
    assert default_guard().approve(query, key, 0.9).ok


def test_numeric_guard_ignores_formatting_and_spelling():
    guard = NumericGuard()

    assert guard.approve("kept for 5 years", "kept for five years", 0.9).ok
    assert guard.approve("over 500,000 NOK", "over 500000 NOK", 0.9).ok
    assert not guard.approve("kept for 5 years", "kept for 7 years", 0.9).ok


def test_polarity_guard_catches_negation():
    guard = PolarityGuard()

    assert not guard.approve(
        "Can customer data be transferred outside the EEA?",
        "Can customer data not be transferred outside the EEA?",
        0.95,
    ).ok


def test_polarity_guard_allows_a_side_that_makes_no_choice():
    """Absence is not disagreement.

    "cross-border payments" and "international payment transactions" are
    synonyms; "domestic payments" is not. Only the second pair picks a different
    member of the same group.
    """
    guard = PolarityGuard()

    assert guard.approve(
        "What is the threshold for reporting a cash transaction?",
        "Above what amount must a cash transaction be reported?",
        0.9,
    ).ok
    assert not guard.approve(
        "What records must be kept for cross-border payments?",
        "What records must be kept for domestic payments?",
        0.9,
    ).ok


def test_acronym_guard_allows_expansion_but_not_substitution():
    guard = AcronymGuard()

    assert guard.approve("Is MFA required for NOK transfers?", "NOK transfers", 0.9).ok
    assert not guard.approve("KYC retention", "AML retention", 0.9).ok


def test_token_overlap_guard_does_nothing_by_default():
    """Default-off is the measured position, not an oversight.

    Any floor high enough to catch near-misses also rejects paraphrases that
    reword heavily, so the default has to be disabled.
    """
    unrelated = ("What is the capital of France?", "How long must KYC records be kept?")

    assert TokenOverlapGuard().approve(*unrelated, 0.9).ok
    assert not TokenOverlapGuard(min_overlap=0.3).approve(*unrelated, 0.9).ok


def test_rerank_guard_skips_confident_hits():
    calls = []

    def scorer(query, key):
        calls.append((query, key))
        return 0.0

    guard = RerankGuard(scorer, min_score=0.5, confident_above=0.97)

    assert guard.approve("a", "b", 0.99).ok, "a near-identical match needs no second opinion"
    assert not calls, "the expensive path must not run on confident hits"

    assert not guard.approve("a", "b", 0.90).ok
    assert calls == [("a", "b")]


def test_composite_reports_which_guard_vetoed():
    guard = CompositeGuard([NumericGuard(), PolarityGuard()])

    decision = guard.approve("over 500,000 NOK", "over 50,000 NOK", 0.98)

    assert not decision.ok
    assert decision.reason.startswith("numeric:")


def test_null_guard_approves_everything():
    assert NullGuard().approve("cancel my order", "cancel my subscription", 0.9).ok
