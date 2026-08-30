"""Sweep the similarity threshold over labeled query pairs.

Answers the question the README used to assert without evidence: what does a
given threshold actually buy, and what does it cost?

    hit rate        fraction of PARAPHRASES served from cache (higher better)
    false-hit rate  fraction of NEAR_MISSES served from cache (lower better)

A false hit is not a cache miss with extra steps -- it returns a confidently
wrong answer about a different regulation. That asymmetry is why the operating
point is not simply "whatever maximizes hit rate".

Usage:
    python benchmarks/threshold_sweep.py
    python benchmarks/threshold_sweep.py --model all-MiniLM-L6-v2
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from benchmarks.pairs import NEAR_MISSES, PARAPHRASES

DEFAULT_MODEL = "all-mpnet-base-v2"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
THRESHOLDS = np.round(np.arange(0.50, 0.96, 0.05), 2)


def pair_similarities(model: SentenceTransformer, pairs: list[tuple[str, str]]) -> np.ndarray:
    """Cosine similarity for each (a, b) pair, encoding every text once."""
    texts = [text for pair in pairs for text in pair]
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    left, right = vectors[0::2], vectors[1::2]
    return np.sum(left * right, axis=1)


def sweep(paraphrase_scores: np.ndarray, near_miss_scores: np.ndarray) -> list[dict]:
    rows = []
    for threshold in THRESHOLDS:
        hit_rate = float(np.mean(paraphrase_scores >= threshold))
        false_hit_rate = float(np.mean(near_miss_scores >= threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "hit_rate": hit_rate,
                "false_hit_rate": false_hit_rate,
                "margin": hit_rate - false_hit_rate,
            }
        )
    return rows


def worst_offenders(
    pairs: list[tuple[str, str]], scores: np.ndarray, count: int = 5
) -> list[dict]:
    """The near-miss pairs the encoder considers most similar."""
    return [
        {"score": float(scores[i]), "a": pairs[i][0], "b": pairs[i][1]}
        for i in np.argsort(scores)[::-1][:count]
    ]


def write_chart(rows: list[dict], model_name: str, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    thresholds = [r["threshold"] for r in rows]
    hits = [r["hit_rate"] * 100 for r in rows]
    false_hits = [r["false_hit_rate"] * 100 for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, hits, marker="o", label="hit rate (paraphrases served)")
    ax.plot(thresholds, false_hits, marker="s", label="false-hit rate (near-misses served)")
    ax.set_xlabel("similarity threshold")
    ax.set_ylabel("percent of pairs")
    ax.set_title(
        f"Threshold sweep - {model_name}\n"
        f"{len(PARAPHRASES)} paraphrase / {len(NEAR_MISSES)} near-miss pairs"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    print(f"loading {args.model} ...")
    model = SentenceTransformer(args.model)

    paraphrase_scores = pair_similarities(model, PARAPHRASES)
    near_miss_scores = pair_similarities(model, NEAR_MISSES)

    print(
        f"\nparaphrases  n={len(paraphrase_scores):3d}  "
        f"mean={paraphrase_scores.mean():.4f}  min={paraphrase_scores.min():.4f}  "
        f"max={paraphrase_scores.max():.4f}"
    )
    print(
        f"near-misses  n={len(near_miss_scores):3d}  "
        f"mean={near_miss_scores.mean():.4f}  min={near_miss_scores.min():.4f}  "
        f"max={near_miss_scores.max():.4f}"
    )

    rows = sweep(paraphrase_scores, near_miss_scores)

    print(f"\n{'threshold':>10} {'hit rate':>10} {'false hits':>12} {'margin':>9}")
    for row in rows:
        print(
            f"{row['threshold']:>10.2f} {row['hit_rate']:>9.1%} "
            f"{row['false_hit_rate']:>12.1%} {row['margin']:>9.1%}"
        )

    offenders = worst_offenders(NEAR_MISSES, near_miss_scores)
    print("\nnear-misses the encoder scores most similar:")
    for item in offenders:
        print(f"  {item['score']:.4f}  {item['a']}")
        print(f"          vs  {item['b']}")

    above_weakest = int(np.sum(near_miss_scores >= paraphrase_scores.min()))
    print(
        f"\ndistribution overlap: {above_weakest}/{len(near_miss_scores)} near-misses "
        f"score above the weakest paraphrase ({paraphrase_scores.min():.4f})"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "n_paraphrases": len(PARAPHRASES),
        "n_near_misses": len(NEAR_MISSES),
        "paraphrase_similarity": {
            "mean": float(paraphrase_scores.mean()),
            "min": float(paraphrase_scores.min()),
            "max": float(paraphrase_scores.max()),
        },
        "near_miss_similarity": {
            "mean": float(near_miss_scores.mean()),
            "min": float(near_miss_scores.min()),
            "max": float(near_miss_scores.max()),
        },
        "near_misses_above_weakest_paraphrase": above_weakest,
        "worst_offenders": offenders,
        "rows": rows,
    }
    (RESULTS_DIR / "threshold_sweep.json").write_text(json.dumps(payload, indent=2) + "\n")

    chart_path = RESULTS_DIR / "threshold_sweep.png"
    if write_chart(rows, args.model, chart_path):
        print(f"\nwrote {chart_path}")
    else:
        print("\nmatplotlib not installed, skipped chart (pip install -e '.[bench]')")
    print(f"wrote {RESULTS_DIR / 'threshold_sweep.json'}")


if __name__ == "__main__":
    main()
