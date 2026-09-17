"""What approximate nearest neighbour actually costs and actually buys.

`benchmarks/bench_cache.py` measures the full scan and shows it going linear:
58.9 ms at 100k entries, which is more than the embedding pass it was meant to
be cheaper than. The obvious fix is an ANN index. The non-obvious part is what
that trades away, and this measures both halves:

    recall@1   how often the ANN index returns the entry an exact scan would
               have. Anything it misses is a cache hit the cache already held
               and silently failed to serve -- indistinguishable, from the
               outside, from a threshold that is too strict.
    latency    lookup cost at the same sizes, against the same vectors.
    memory     the index structure's own copy, on top of the mirror every
               backend keeps so candidates can be rescored exactly.

Two vector distributions, because the answer depends entirely on which one you
have. Uniform random unit vectors in 768 dimensions are the adversarial case:
every pair is near-orthogonal, distances concentrate, and there is no
neighbourhood structure for a graph or a partition to exploit -- ANN recall
collapses and the honest conclusion is "do not use ANN on data like this".
Sentence embeddings are not like this. They lie on a much lower-dimensional
manifold and arrive in topic clusters, which is exactly what HNSW and IVF are
built for. The clustered distribution is a stand-in for that.

Reporting only the first would libel ANN; reporting only the second would sell
it. Both are below.

No encoder, no model download:

    python -m benchmarks.bench_ann
    python -m benchmarks.bench_ann --sizes 1000,10000 --distributions clustered
"""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import numpy as np

from gateway.index import available_backends, build_index

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DIM = 768


def rss_mb() -> float:
    with open("/proc/self/statm") as handle:
        pages = int(handle.read().split()[1])
    return pages * 4096 / 1024**2


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def unit_vectors(count: int, dim: int, seed: int) -> np.ndarray:
    """Uniform on the sphere: the case with no structure to exploit."""
    rng = np.random.default_rng(seed)
    return _normalize_rows(rng.normal(size=(count, dim)).astype(np.float32))


def clustered_vectors(
    count: int, dim: int, seed: int, *, clusters: int = 0, spread: float = 1.0
) -> np.ndarray:
    """Topic clusters with noise: a stand-in for real query embeddings.

    Not a claim about any particular corpus -- a claim that real embeddings have
    neighbourhood structure and uniform noise does not, which is the whole
    premise an ANN index rests on.

    ``spread`` is the norm of the noise relative to the centroid, so it has to
    be divided by sqrt(dim) to get the per-component scale. Getting that wrong
    is the easy mistake here: at 768 dimensions, per-component noise of 0.35
    produces a perturbation ten times longer than the centroid it was meant to
    perturb, and the "clustered" data comes out indistinguishable from uniform
    noise. ``mean_top1_similarity`` below is the check -- if it is near zero,
    there are no clusters.
    """
    rng = np.random.default_rng(seed)
    clusters = clusters or max(8, count // 200)
    centroids = _normalize_rows(rng.normal(size=(clusters, dim)).astype(np.float32))
    assignments = rng.integers(0, clusters, size=count)
    noise = rng.normal(scale=spread / np.sqrt(dim), size=(count, dim)).astype(np.float32)
    return _normalize_rows(centroids[assignments] + noise)


DISTRIBUTIONS = {"random": unit_vectors, "clustered": clustered_vectors}


def measure(
    backend: str, vectors: np.ndarray, probes: np.ndarray, k: int, distribution: str
) -> dict:
    size, dim = vectors.shape

    gc.collect()
    before = rss_mb()
    build_start = time.perf_counter()
    index = build_index(dim, size, backend)
    for slot, vector in enumerate(vectors):
        index.upsert(slot, vector)
    build_s = time.perf_counter() - build_start
    gc.collect()
    after = rss_mb()

    similarities = probes @ vectors.T
    truth = [int(i) for i in np.argmax(similarities, axis=1)]
    mean_top1 = float(np.mean(np.max(similarities, axis=1)))

    samples, found = [], 0
    for probe, expected in zip(probes, truth, strict=True):
        start = time.perf_counter()
        candidates = index.search(probe, k)
        samples.append(time.perf_counter() - start)
        found += expected in candidates

    del index
    gc.collect()

    ordered = sorted(samples)
    return {
        "backend": backend,
        "entries": size,
        "distribution": distribution,
        "recall_at_k": found / len(probes),
        "mean_top1_similarity": mean_top1,
        "k": k,
        "lookup_p50_ms": statistics.median(ordered) * 1000,
        "lookup_p95_ms": ordered[int(len(ordered) * 0.95) - 1] * 1000,
        "build_s": build_s,
        "rss_delta_mb": after - before,
        "mirror_mb": size * dim * 4 / 1024**2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="1000,10000,100000")
    parser.add_argument("--probes", type=int, default=200)
    parser.add_argument("--k", type=int, default=5, help="candidates the cache asks for")
    parser.add_argument("--dim", type=int, default=DIM)
    parser.add_argument("--distributions", default="clustered,random")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    distributions = [d.strip() for d in args.distributions.split(",")]
    backends = available_backends()
    print(f"backends: {', '.join(backends)}")
    if backends == ["exact"]:
        print("(no ANN backend installed -- pip install -e '.[ann]')")

    rows = []
    for distribution in distributions:
        generate = DISTRIBUTIONS[distribution]
        for size in sizes:
            # Data and probes are drawn together, so the queries live in the
            # same cluster structure as the cache. Drawing them separately gives
            # every probe a nearest neighbour at ~0.1 similarity -- a query
            # unlike anything cached, which measures nothing a cache would ever
            # be asked.
            population = generate(size + args.probes, args.dim, seed=size)
            vectors, probes = population[:size], population[size:]
            print(f"\n{distribution} vectors, n={size:,}")
            for backend in backends:
                row = measure(backend, vectors, probes, args.k, distribution)
                rows.append(row)
                print(
                    f"  {row['backend']:>10}  recall@{args.k}={row['recall_at_k']:.3f}  "
                    f"p50={row['lookup_p50_ms']:8.4f} ms  p95={row['lookup_p95_ms']:8.4f} ms  "
                    f"build={row['build_s']:6.2f} s  rss+={row['rss_delta_mb']:7.1f} MB"
                )
            del population, vectors, probes
            gc.collect()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "dim": args.dim,
        "probes": args.probes,
        "k": args.k,
        "distributions": distributions,
        "rows": rows,
    }
    (RESULTS_DIR / "ann.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {RESULTS_DIR / 'ann.json'}")


if __name__ == "__main__":
    main()
