"""Cheap verification between a query and the cache key that nearly matched it.

The threshold sweep in `benchmarks/threshold_sweep.py` establishes that cosine
similarity alone cannot gate this cache: the highest-scoring *wrong* pair in the
labeled set is 0.9792 ("transfers over 500,000 NOK" against "over 50,000 NOK"),
above every genuine paraphrase in the same set. No threshold separates them,
because the distinction is a numeric literal and the encoder is modelling topic.

A guard runs after the threshold and before the answer is served. It sees both
strings -- the incoming query and the cached key that scored highest -- and gets
to veto. It is deliberately lexical and deterministic: the expensive semantic
judgement already happened, and what is left is the class of difference that
embeddings are known to flatten.

    numeric   different numbers mean different answers. 500,000 != 50,000.
    polarity  maximum vs minimum, internal vs external, may vs must not.
    acronym   KYC vs AML: one token, entirely different regulation.

Each is a *veto*, never an endorsement: passing the guard does not make a hit
correct, it only means this particular cheap check found nothing wrong. A veto
costs a cache miss, which costs money and latency; a false hit costs a
confidently wrong answer. The asymmetry is the whole reason to run the guard.

The lexicons below were written against the 50 labeled pairs in
`benchmarks/pairs.py` and are domain-specific by design -- `benchmarks/guard_eval.py`
measures them, and the README reports the numbers as in-sample. Deployed
somewhere else, this file is the thing to edit.
"""

import re
from functools import lru_cache
from typing import NamedTuple, Protocol, runtime_checkable

_WORD = re.compile(r"[a-z0-9]+")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d{3})*(?:\.\d+)?")
_ACRONYM = re.compile(r"\b[A-Z]{2,6}\b")

# Spelled-out numbers that appear in questions about periods and counts.
WORD_NUMBERS = {
    "zero": 0.0, "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0, "five": 5.0,
    "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0, "ten": 10.0,
    "eleven": 11.0, "twelve": 12.0, "fifteen": 15.0, "twenty": 20.0,
    "thirty": 30.0, "forty": 40.0, "fifty": 50.0, "sixty": 60.0, "ninety": 90.0,
    "hundred": 100.0, "thousand": 1000.0, "million": 1_000_000.0,
}

# Words that flip an answer rather than rephrase the question.
_NEGATIONS = frozenset(
    {"not", "no", "never", "without", "except", "unless", "cannot", "exempt", "excluded"}
)

# Mutually exclusive choices. Each group is a tuple of *members*; each member is
# a tuple of surface forms that mean the same thing. Two texts contradict each
# other when they pick different members of the same group -- picking nothing is
# not a contradiction, which is what keeps "cross-border payments" compatible
# with "international payment transactions" while still separating either from
# "domestic payments".
POLARITY_GROUPS: tuple[tuple[tuple[str, ...], ...], ...] = (
    (("minimum", "min", "at least", "shortest"), ("maximum", "max", "at most", "longest")),
    (("internal",), ("external",)),
    (("domestic", "national", "local"), ("international", "cross border", "overseas", "foreign")),
    (("inbound", "incoming"), ("outbound", "outgoing")),
    (("above", "over", "more than", "greater than", "exceeding"),
     ("below", "under", "less than", "fewer than")),
    (("before", "prior to"), ("after", "following")),
    (("enhanced", "extended"), ("simplified", "standard", "basic")),
    (("required", "mandatory", "obligatory"), ("optional", "voluntary", "discretionary"),
     ("prohibited", "forbidden", "banned")),
    (("at rest", "stored data", "data storage"), ("in transit", "in transfer")),
    (("retention", "retained", "retain", "kept", "keep", "preserved", "archive", "archived"),
     ("deletion", "deleted", "delete", "erasure", "disposal")),
    (("approve", "approval", "sign off", "authorise", "authorize"),
     ("reject", "refuse", "deny", "decline"), ("reverse", "revoke", "cancel"),
     ("review", "re review", "reviewed")),
    (("open", "opened", "opening", "onboard", "onboarding"),
     ("close", "closed", "closing", "offboard", "offboarding")),
    (("retail", "consumer", "private"), ("corporate", "business", "institutional")),
    (("customer", "client"), ("employee", "staff", "personnel"), ("contractor", "freelancer"),
     ("subsidiary", "affiliate"), ("vendor", "supplier", "third party")),
    (("kyc",), ("aml",), ("tax",), ("gdpr",), ("sanction", "sanctioned"),
     ("pep", "politically exposed person"), ("fraud",),
     ("sar", "suspicious activity report")),
    (("cash",), ("wire",), ("card",), ("crypto", "cryptocurrency", "virtual asset")),
    (("breach",), ("suspicious transaction",), ("complaint",), ("outage", "incident")),
    (("daily",), ("weekly",), ("monthly",), ("quarterly",),
     ("annual", "annually", "yearly", "every year")),
    (("risk",), ("pricing", "price"), ("credit",), ("liquidity",)),
    (("audit",), ("training",), ("assessment",), ("filing", "filed", "file", "report")),
    (("compliance",), ("security",), ("privacy",)),
    (("eea", "europe", "european economic area", "eu"),
     ("united states", "usa", "us", "america"), ("uk", "united kingdom"), ("norway", "nok")),
    (("late", "overdue"), ("inaccurate", "incorrect", "erroneous")),
)


class GuardDecision(NamedTuple):
    """``ok`` is whether the hit may be served; ``reason`` says why it may not."""

    ok: bool
    reason: str = ""


APPROVED = GuardDecision(True, "")


@runtime_checkable
class CacheGuard(Protocol):
    """Vetoes a cache hit that cleared the similarity threshold."""

    name: str

    def approve(self, query: str, cached_key: str, score: float) -> GuardDecision: ...


def _singular(token: str) -> str:
    """Strips a trailing plural/third-person ``s`` so surface forms line up."""
    if len(token) > 3:
        if token.endswith("ies"):
            return token[:-3] + "y"
        if token.endswith("sses") or token.endswith("shes") or token.endswith("ches"):
            return token[:-2]
        if token.endswith("s") and not token.endswith("ss"):
            return token[:-1]
    return token


@lru_cache(maxsize=4096)
def tokens(text: str) -> tuple[str, ...]:
    """Normalized, naively singularized tokens.

    Memoized because the guard runs on the hit path and both of its arguments
    repeat constantly: the cached key is by definition a string the cache has
    seen before, and a busy cache compares the same few thousand keys over and
    over.
    """
    return tuple(_singular(token) for token in _WORD.findall(text.lower()))


@lru_cache(maxsize=4096)
def padded_tokens(text: str) -> str:
    """Normalized token stream padded with spaces, for phrase membership tests."""
    return " " + " ".join(tokens(text)) + " "


def _normalize_member(member: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(" ".join(tokens(alias)) for alias in member)


_NORMALIZED_GROUPS = tuple(
    tuple(_normalize_member(member) for member in group) for group in POLARITY_GROUPS
)

# term -> (group index, member index), so one regex pass over a query finds every
# polarity term in it. The alternative -- a substring search per alias per group --
# is ~100 scans of the same short string on every cache hit.
_TERM_INDEX: dict[str, tuple[int, int]] = {
    alias: (group_index, member_index)
    for group_index, group in enumerate(_NORMALIZED_GROUPS)
    for member_index, member in enumerate(group)
    for alias in member
}
_TERM_RE = re.compile(
    r"(?<![a-z0-9])(?:"
    + "|".join(re.escape(term) for term in sorted(_TERM_INDEX, key=len, reverse=True))
    + r")(?![a-z0-9])"
)


@lru_cache(maxsize=4096)
def _member_choices(text: str) -> dict[int, frozenset[int]]:
    """Which member of each polarity group this text picks, if any."""
    found: dict[int, set[int]] = {}
    for match in _TERM_RE.finditer(padded_tokens(text)):
        group_index, member_index = _TERM_INDEX[match.group(0)]
        found.setdefault(group_index, set()).add(member_index)
    return {group: frozenset(members) for group, members in found.items()}


def _numbers(text: str) -> frozenset[float]:
    """Numeric literals, thousands separators removed, spelled-out words included."""
    values = set()
    for raw in NUMBER_RE.findall(text):
        cleaned = raw.replace(",", "")
        try:
            values.add(float(cleaned))
        except ValueError:  # pragma: no cover - the regex cannot produce this
            continue
    for token in tokens(text):
        if token in WORD_NUMBERS:
            values.add(WORD_NUMBERS[token])
    return frozenset(values)


def _incompatible(left: frozenset[int], right: frozenset[int]) -> bool:
    """True when both sides chose and neither choice contains the other."""
    return bool(left) and bool(right) and not (left <= right or right <= left)


class NumericGuard:
    """Rejects a hit whose numeric literals differ from the cached query's.

    This is the guard that catches the worst pair the sweep found: 500,000 NOK
    and 50,000 NOK score 0.9792 and differ by a factor of ten. A number in a
    compliance question is almost always the answer's hinge -- a threshold, a
    retention period, a deadline -- so a difference in the set of numbers is
    treated as a difference in the question.
    """

    name = "numeric"

    def approve(self, query: str, cached_key: str, score: float) -> GuardDecision:
        left, right = _numbers(query), _numbers(cached_key)
        if left != right:
            return GuardDecision(
                False, f"numeric literals differ: {sorted(left)} vs {sorted(right)}"
            )
        return APPROVED


class PolarityGuard:
    """Rejects a hit that picks a different member of a mutually exclusive group.

    Covers the second and third worst pairs in the sweep -- maximum vs minimum
    retention (0.9520) and internal vs external transfers (0.9343) -- plus
    negation, where a single "not" inverts the answer while barely moving the
    embedding.
    """

    name = "polarity"

    def approve(self, query: str, cached_key: str, score: float) -> GuardDecision:
        left_neg = {t for t in tokens(query) if t in _NEGATIONS}
        right_neg = {t for t in tokens(cached_key) if t in _NEGATIONS}
        if left_neg != right_neg:
            return GuardDecision(
                False, f"negation differs: {sorted(left_neg)} vs {sorted(right_neg)}"
            )

        left_choices = _member_choices(query)
        right_choices = _member_choices(cached_key)
        for group_index in left_choices.keys() & right_choices.keys():
            left = left_choices[group_index]
            right = right_choices[group_index]
            if _incompatible(left, right):
                group = _NORMALIZED_GROUPS[group_index]
                return GuardDecision(
                    False,
                    "mutually exclusive terms: "
                    f"{sorted(group[i][0] for i in left)} vs "
                    f"{sorted(group[i][0] for i in right)}",
                )
        return APPROVED


class AcronymGuard:
    """Rejects a hit whose acronyms are not a subset of the query's, or vice versa.

    KYC and AML differ by one token and score 0.8608 in the sweep. Subset rather
    than equality, because expanding an acronym is a normal paraphrase: "Do
    transfers over 500,000 NOK need multi-factor authentication?" and "Is MFA
    required for transfers above 500,000 NOK?" are the same question, and {NOK}
    is a subset of {MFA, NOK}.

    Subset has a cost: "retention limit for KYC records" against "retention
    limit for KYC and AML records" passes this guard, because {KYC} is a subset
    of {KYC, AML}. Allowing expansion means allowing that. The alternative --
    requiring equality -- rejects the paraphrase above, which is a more common
    case than the compound question.
    """

    name = "acronym"

    def approve(self, query: str, cached_key: str, score: float) -> GuardDecision:
        left = frozenset(_ACRONYM.findall(query))
        right = frozenset(_ACRONYM.findall(cached_key))
        if bool(left) and bool(right) and not (left <= right or right <= left):
            return GuardDecision(
                False, f"acronyms differ: {sorted(left)} vs {sorted(right)}"
            )
        return APPROVED


class TokenOverlapGuard:
    """Rejects a hit whose content-word overlap with the cached key is too low.

    Off by default (``min_overlap=0.0``). Measured on the labeled pairs, any
    floor high enough to catch near-misses also rejects genuine paraphrases:
    "How long must KYC records be kept?" and "What is the retention period for
    KYC records?" share a Jaccard overlap of 0.33, which is *below* several
    near-miss pairs that differ by a single decisive token. Overlap measures
    rewriting effort, not agreement. Kept because it is occasionally the right
    tool for a narrow domain, with a default that does nothing.
    """

    name = "overlap"

    _STOPWORDS = frozenset(
        {
            "a", "an", "and", "are", "at", "be", "by", "can", "do", "doe", "for", "from",
            "how", "i", "in", "is", "it", "long", "may", "must", "of", "on", "or", "our",
            "required", "the", "there", "to", "we", "what", "when", "where", "which",
            "who", "why", "will", "with",
        }
    )

    def __init__(self, min_overlap: float = 0.0):
        self.min_overlap = min_overlap

    def _content(self, text: str) -> frozenset[str]:
        return frozenset(t for t in tokens(text) if t not in self._STOPWORDS)

    def approve(self, query: str, cached_key: str, score: float) -> GuardDecision:
        if self.min_overlap <= 0.0:
            return APPROVED
        left, right = self._content(query), self._content(cached_key)
        union = left | right
        if not union:
            return APPROVED
        overlap = len(left & right) / len(union)
        if overlap < self.min_overlap:
            return GuardDecision(
                False, f"token overlap {overlap:.2f} below {self.min_overlap:.2f}"
            )
        return APPROVED


class RerankGuard:
    """Second opinion from a cross-encoder, on borderline hits only.

    A bi-encoder embeds the two strings independently; a cross-encoder reads them
    together and can therefore notice what the first one flattened. It is also
    perhaps 50x more expensive, which is why this runs only in the band between
    the threshold and ``confident_above`` -- a 0.99 match does not need a second
    opinion, and paying for one on every hit erodes the saving the cache exists
    to produce.

    Duck-typed on a callable ``scorer(query, cached_key) -> float`` so nothing is
    imported unless a caller wants it:

        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        guard = RerankGuard(lambda q, k: float(model.predict([(q, k)])[0]), min_score=0.5)
    """

    name = "rerank"

    def __init__(self, scorer, min_score: float = 0.5, confident_above: float = 0.97):
        self.scorer = scorer
        self.min_score = min_score
        self.confident_above = confident_above

    def approve(self, query: str, cached_key: str, score: float) -> GuardDecision:
        if score >= self.confident_above:
            return APPROVED
        reranked = float(self.scorer(query, cached_key))
        if reranked < self.min_score:
            return GuardDecision(
                False, f"reranker scored {reranked:.4f} below {self.min_score:.2f}"
            )
        return APPROVED


class CompositeGuard:
    """Runs guards in order and returns the first veto.

    Order matters for cost, not for outcome: the cheap lexical checks run before
    anything that might call a model.
    """

    name = "composite"

    def __init__(self, guards):
        self.guards = list(guards)

    def approve(self, query: str, cached_key: str, score: float) -> GuardDecision:
        for guard in self.guards:
            decision = guard.approve(query, cached_key, score)
            if not decision.ok:
                return GuardDecision(False, f"{guard.name}: {decision.reason}")
        return APPROVED


class NullGuard:
    """Approves everything. The behaviour this repo shipped before the guard existed."""

    name = "none"

    def approve(self, query: str, cached_key: str, score: float) -> GuardDecision:
        return APPROVED


def default_guard() -> CompositeGuard:
    """The guard the gateway installs unless told otherwise.

    Lexical only: no model, no network, microseconds per check. Deliberately not
    including ``TokenOverlapGuard`` or ``RerankGuard`` -- the first rejects
    genuine paraphrases at any useful setting, the second needs a model the repo
    does not ship.
    """
    return CompositeGuard([NumericGuard(), PolarityGuard(), AcronymGuard()])
