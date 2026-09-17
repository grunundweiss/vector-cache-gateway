"""Per-intent similarity thresholds.

One global threshold treats every question as equally dangerous to get wrong.
The sweep says otherwise: the near-miss pairs that score highest are the ones
that hinge on a number or a polarity word, and those are exactly the questions
where a confidently wrong answer does the most damage. "What is the reporting
threshold?" and "Who owns the incident response process?" do not deserve the
same gate.

So the threshold becomes a function of the query rather than a constant. The
classifier is rule-based and reuses the guard's lexicons, because a learned
intent classifier would be a second model to serve, a second thing to keep in
sync with the data, and a second source of silent failure in front of a cache
whose entire job is to be cheaper than the thing behind it.

    Honest accounting: only the global 0.85 in gateway/cache.py is measured --
    `benchmarks/threshold_sweep.py` produced it from 50 labeled pairs. The
    per-intent values below are a *policy*, chosen conservatively in the
    direction the sweep points, not a measurement. They raise the bar on the
    classes where the guard's lexicon is most likely to be incomplete, which is
    the same place the encoder is least reliable.

    `benchmarks/intent_sweep.py` turns them into measurements: it re-runs the
    sweep per intent cluster and writes `results/intent_thresholds.json`, which
    `IntentThresholds.from_json` loads. Until that is run on your own labeled
    pairs, these are defaults with a rationale, not evidence.
"""

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from gateway.cache import DEFAULT_THRESHOLD
from gateway.guard import NUMBER_RE, WORD_NUMBERS, padded_tokens, tokens

# Intent labels, most dangerous first. ``classify`` returns the first that matches.
NUMERIC = "numeric"
POLARITY = "polarity"
ENTITY = "entity"
GENERAL = "general"

INTENTS = (NUMERIC, POLARITY, ENTITY, GENERAL)

# Terms whose presence means the answer inverts if the term is swapped. A subset
# of the guard's groups: the ones where both members are common enough that an
# encoder will happily score them near-identical.
_POLARITY_MARKERS = (
    "minimum", "maximum", "min", "max", "at least", "at most",
    "internal", "external", "inbound", "outbound",
    "domestic", "international", "cross border",
    "above", "over", "below", "under", "before", "after",
    "enhanced", "simplified", "required", "mandatory", "optional", "prohibited",
    "at rest", "in transit", "retention", "deletion",
    "not", "never", "without", "unless", "except",
)

_ENTITY_MARKERS = (
    "kyc", "aml", "gdpr", "pep", "sar", "mfa", "eea", "eu", "psd2",
    "retail", "corporate", "customer", "employee", "contractor", "subsidiary",
    "vendor", "cash", "wire", "crypto",
)

# Unmeasured, conservative. See the module docstring before quoting these.
DEFAULT_INTENT_THRESHOLDS: dict[str, float] = {
    NUMERIC: 0.95,
    POLARITY: 0.93,
    ENTITY: 0.90,
    GENERAL: DEFAULT_THRESHOLD,
}


def classify(query: str) -> str:
    """Labels a query with the intent cluster that decides its threshold.

    Deliberately a few `in` tests over a normalized token stream: it runs once
    per query, before the embedding pass that costs four orders of magnitude
    more, and a rule that can be read is a rule that can be corrected.
    """
    padded = padded_tokens(query)

    if NUMBER_RE.search(query) or any(token in WORD_NUMBERS for token in tokens(query)):
        return NUMERIC
    if any(f" {marker} " in padded for marker in _POLARITY_MARKERS):
        return POLARITY
    if any(f" {marker} " in padded for marker in _ENTITY_MARKERS):
        return ENTITY
    return GENERAL


@runtime_checkable
class ThresholdPolicy(Protocol):
    """Decides the similarity a query must clear to be served from cache."""

    def threshold_for(self, query: str) -> float: ...


class GlobalThreshold:
    """One threshold for every query. What the gateway did before intents existed."""

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold

    def threshold_for(self, query: str) -> float:
        return self.threshold


class IntentThresholds:
    """A threshold per intent cluster, with a fallback for unlabeled intents.

    The incoming query decides the threshold, not the cached key it matches --
    the key is not known until after the search, and a gate that depends on what
    it finds is not a gate. A numeric query therefore has to clear the numeric
    bar even when it lands on a cached entry that contains no numbers at all,
    which is the conservative direction.
    """

    def __init__(
        self,
        thresholds: dict[str, float] | None = None,
        default: float = DEFAULT_THRESHOLD,
        classifier=classify,
    ):
        self.thresholds = dict(DEFAULT_INTENT_THRESHOLDS if thresholds is None else thresholds)
        self.default = default
        self.classifier = classifier

    def threshold_for(self, query: str) -> float:
        return self.thresholds.get(self.classifier(query), self.default)

    def explain(self, query: str) -> tuple[str, float]:
        """Returns (intent, threshold) -- the same decision, for logs and demos."""
        intent = self.classifier(query)
        return intent, self.thresholds.get(intent, self.default)

    @classmethod
    def from_json(cls, path: str | Path, default: float = DEFAULT_THRESHOLD) -> "IntentThresholds":
        """Loads thresholds measured by `benchmarks/intent_sweep.py`.

        Accepts either a bare ``{intent: threshold}`` mapping or the full sweep
        payload, which nests one under ``"thresholds"`` alongside the evidence
        that produced it.
        """
        payload = json.loads(Path(path).read_text())
        thresholds = payload.get("thresholds", payload)
        return cls({str(k): float(v) for k, v in thresholds.items()}, default=default)


def resolve_policy(policy, threshold: float) -> ThresholdPolicy:
    """Normalizes the constructor argument callers actually pass.

    ``None`` keeps the single measured threshold, ``"intent"`` opts into the
    unmeasured per-intent policy, and anything else is used as given.
    """
    if policy is None:
        return GlobalThreshold(threshold)
    if policy == "intent":
        return IntentThresholds(default=threshold)
    return policy
