"""Measure the verification guard against the same labeled pairs as the sweep.

The threshold sweep needs the encoder and therefore a model download. This does
not: the guard is lexical and deterministic, so its behaviour on the 50 labeled
pairs can be measured exactly, offline, in milliseconds.

Two rates, and they are not symmetric:

    catch rate  near-miss pairs the guard vetoes. These are the wrong answers it
                prevents -- but only the ones that also clear the threshold were
                ever going to be served, so this is an upper bound on what the
                guard saves, not a claim about production.
    veto rate   paraphrase pairs the guard vetoes. Each one is a cache miss that
                should have been a hit: money and latency, not correctness.

The lexicons in gateway/guard.py were written against these pairs, so every
number here is in-sample. It measures whether the guard does what it says on the
cases the sweep proved were dangerous; it does not measure generalization.

    python -m benchmarks.guard_eval
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from benchmarks.pairs import NEAR_MISSES, PARAPHRASES
from gateway.guard import default_guard

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SWEEP_PATH = RESULTS_DIR / "threshold_sweep.json"


def evaluate(guard, pairs: list[tuple[str, str]]) -> list[dict]:
    rows = []
    for left, right in pairs:
        decision = guard.approve(left, right, 1.0)
        rows.append(
            {"a": left, "b": right, "vetoed": not decision.ok, "reason": decision.reason}
        )
    return rows


def worst_offenders_check(guard) -> list[dict]:
    """The guard against the near-miss pairs the encoder scored highest.

    These are the pairs the threshold demonstrably cannot stop: every one of them
    scores above the shipped 0.85, so each is a wrong answer served with
    confidence unless something else vetoes it.
    """
    if not SWEEP_PATH.exists():
        return []
    payload = json.loads(SWEEP_PATH.read_text())
    rows = []
    for item in payload.get("worst_offenders", []):
        decision = guard.approve(item["a"], item["b"], item["score"])
        rows.append(
            {
                "score": item["score"],
                "a": item["a"],
                "b": item["b"],
                "vetoed": not decision.ok,
                "reason": decision.reason,
            }
        )
    return rows


def time_guard(guard, pairs: list[tuple[str, str]], repeats: int = 200) -> dict:
    """Per-check cost. The guard runs on the hit path, so this is latency added
    to every cache hit -- worth knowing against the ~40 ms embedding pass that
    precedes it."""
    samples = []
    for _ in range(repeats):
        for left, right in pairs:
            start = time.perf_counter()
            guard.approve(left, right, 0.9)
            samples.append((time.perf_counter() - start) * 1_000_000)
    ordered = sorted(samples)
    return {
        "p50_us": statistics.median(ordered),
        "p95_us": ordered[int(len(ordered) * 0.95) - 1],
        "mean_us": statistics.mean(ordered),
        "checks": len(ordered),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="print every pair, not just vetoes")
    parser.add_argument(
        "--min-catch",
        type=int,
        default=0,
        help="exit non-zero if fewer near-miss pairs than this are vetoed (CI uses this)",
    )
    parser.add_argument(
        "--max-false-veto",
        type=int,
        default=None,
        help="exit non-zero if more paraphrase pairs than this are vetoed",
    )
    args = parser.parse_args()

    guard = default_guard()
    near_miss_rows = evaluate(guard, NEAR_MISSES)
    paraphrase_rows = evaluate(guard, PARAPHRASES)

    caught = sum(row["vetoed"] for row in near_miss_rows)
    vetoed = sum(row["vetoed"] for row in paraphrase_rows)

    print(f"near-misses vetoed (caught):     {caught}/{len(near_miss_rows)}")
    print(f"paraphrases vetoed (false veto): {vetoed}/{len(paraphrase_rows)}")

    print("\nnear-misses the guard does NOT catch:")
    for row in near_miss_rows:
        if not row["vetoed"]:
            print(f"  {row['a']}\n    vs {row['b']}")

    if vetoed:
        print("\nparaphrases the guard wrongly vetoes:")
        for row in paraphrase_rows:
            if row["vetoed"]:
                print(f"  {row['a']}\n    vs {row['b']}\n    {row['reason']}")

    timing = time_guard(guard, NEAR_MISSES + PARAPHRASES)
    print(
        f"\nper-check cost: p50={timing['p50_us']:.1f} us  p95={timing['p95_us']:.1f} us "
        f"({timing['checks']:,} checks)"
    )

    offenders = worst_offenders_check(guard)
    if offenders:
        print("\nagainst the highest-scoring near-misses from the sweep:")
        for row in offenders:
            mark = "vetoed" if row["vetoed"] else "SERVED"
            print(f"  {row['score']:.4f}  {mark:>6}  {row['a']}")
            print(f"                    vs  {row['b']}")
            if row["vetoed"]:
                print(f"                        {row['reason']}")

    if args.verbose:
        print("\nall near-miss decisions:")
        for row in near_miss_rows:
            print(f"  {'veto' if row['vetoed'] else 'pass'}  {row['a']} || {row['b']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_paraphrases": len(paraphrase_rows),
        "n_near_misses": len(near_miss_rows),
        "near_miss_catch_rate": caught / len(near_miss_rows),
        "paraphrase_veto_rate": vetoed / len(paraphrase_rows),
        "worst_offenders": offenders,
        "timing": timing,
        "near_misses": near_miss_rows,
        "paraphrases": paraphrase_rows,
    }
    (RESULTS_DIR / "guard_eval.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {RESULTS_DIR / 'guard_eval.json'}")

    # A guard that quietly stops catching things is worse than no guard, because
    # the README still says it catches them. These make that a build failure.
    failures = []
    if caught < args.min_catch:
        failures.append(
            f"caught {caught}/{len(near_miss_rows)}, required at least {args.min_catch}"
        )
    if args.max_false_veto is not None and vetoed > args.max_false_veto:
        failures.append(
            f"vetoed {vetoed} paraphrases, allowed at most {args.max_false_veto}"
        )
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
