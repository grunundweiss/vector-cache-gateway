"""Sweep the threshold separately for each intent cluster.

`benchmarks/threshold_sweep.py` produces one number for all traffic. That number
is a compromise between questions where a wrong answer is embarrassing and
questions where it is a compliance incident. This splits the same labeled pairs
by what makes them risky -- a number, a polarity word, a named entity -- and
sweeps each cluster on its own, so each gets the threshold its own data
supports.

The output is `results/intent_thresholds.json`, which `IntentThresholds.from_json`
loads. Until it is run, gateway/policy.py ships conservative defaults that are
policy rather than measurement, and says so.

Small clusters are the honest caveat: 50 pairs split four ways leaves some
clusters with a handful of examples, and a threshold chosen from six pairs is a
guess with a decimal point. The per-cluster counts are printed and written so
the guess is visible as one.

    python -m benchmarks.intent_sweep
    python -m benchmarks.intent_sweep --min-pairs 8
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmarks.pairs import NEAR_MISSES, PARAPHRASES
from benchmarks.threshold_sweep import DEFAULT_MODEL, THRESHOLDS, pair_similarities
from gateway.cache import DEFAULT_THRESHOLD
from gateway.guard import default_guard
from gateway.policy import INTENTS, classify

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def group_by_intent(pairs, scores) -> dict[str, list[float]]:
    """Buckets each pair by the intent of its first query.

    The first query stands in for the pair: it is the one that would arrive at
    the cache and choose the threshold, since the cached key it lands on is not
    known until after the search.
    """
    grouped: dict[str, list[float]] = defaultdict(list)
    for (first, _), score in zip(pairs, scores, strict=True):
        grouped[classify(first)].append(float(score))
    return grouped


def best_threshold(hits: list[float], false_hits: list[float]) -> tuple[float, float]:
    """The threshold with the widest margin between hit rate and false-hit rate.

    Ties go to the stricter threshold: two operating points that buy the same
    margin are not equivalent when one of them serves fewer wrong answers.
    """
    best, best_margin = DEFAULT_THRESHOLD, -1.0
    for threshold in THRESHOLDS:
        hit_rate = float(np.mean([s >= threshold for s in hits])) if hits else 0.0
        false_rate = float(np.mean([s >= threshold for s in false_hits])) if false_hits else 0.0
        margin = hit_rate - false_rate
        if margin >= best_margin:
            best, best_margin = float(threshold), margin
    return best, best_margin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=6,
        help="clusters with fewer pairs keep the global threshold instead",
    )
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    print(f"loading {args.model} ...")
    model = SentenceTransformer(args.model)

    paraphrase_scores = pair_similarities(model, PARAPHRASES)
    near_miss_scores = pair_similarities(model, NEAR_MISSES)
    guard = default_guard()

    hits_by_intent = group_by_intent(PARAPHRASES, paraphrase_scores)
    # A near-miss the guard vetoes is not a false hit at any threshold, so
    # sweeping without accounting for the guard measures a system that is not
    # the one being shipped.
    guarded = [
        (pair, score)
        for pair, score in zip(NEAR_MISSES, near_miss_scores, strict=True)
        if guard.approve(pair[0], pair[1], float(score)).ok
    ]
    false_by_intent = group_by_intent(
        [pair for pair, _ in guarded], [score for _, score in guarded]
    )

    thresholds, evidence = {}, {}
    print(f"\n{'intent':>10} {'pairs':>6} {'vetoed':>7} {'threshold':>10} {'margin':>8}")
    for intent in INTENTS:
        hits = hits_by_intent.get(intent, [])
        false_hits = false_by_intent.get(intent, [])
        vetoed = len(
            [p for p in NEAR_MISSES if classify(p[0]) == intent]
        ) - len(false_hits)

        if len(hits) + len(false_hits) < args.min_pairs:
            threshold, margin = DEFAULT_THRESHOLD, float("nan")
            note = "too few pairs; kept the global threshold"
        else:
            threshold, margin = best_threshold(hits, false_hits)
            note = ""

        thresholds[intent] = threshold
        evidence[intent] = {
            "paraphrase_pairs": len(hits),
            "near_miss_pairs_after_guard": len(false_hits),
            "near_miss_pairs_vetoed_by_guard": vetoed,
            "margin": None if margin != margin else margin,
            "note": note,
        }
        print(
            f"{intent:>10} {len(hits) + len(false_hits):>6} {vetoed:>7} "
            f"{threshold:>10.2f} {margin:>8.1%}"
            + (f"   {note}" if note else "")
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "note": (
            "Measured per intent cluster, with the lexical guard applied to the "
            "near-miss set first. Cluster sizes are small; see evidence."
        ),
        "thresholds": thresholds,
        "evidence": evidence,
    }
    path = RESULTS_DIR / "intent_thresholds.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {path}")
    print("load it with IntentThresholds.from_json(...)")


if __name__ == "__main__":
    main()
