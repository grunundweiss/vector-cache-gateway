"""Latency, memory and scaling benchmarks for the semantic cache.

Measures the claim the README makes rather than asserting it. Run:

    python -m benchmarks.bench_cache
    python -m benchmarks.bench_cache --max-scale 10000   # skip the 100k case
"""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import numpy as np

from benchmarks.pairs import NEAR_MISSES, PARAPHRASES
from gateway.cache import DEFAULT_THRESHOLD, SemanticCache
from gateway.semantic_cache import SemanticCacheGateway
from gateway.vector_store import LocalVectorStore

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DIM = 768
DOC = "KYC records must be retained for a minimum of 5 years."


def rss_mb() -> float:
    """Current resident set size in MB, read from /proc."""
    with open("/proc/self/statm") as handle:
        pages = int(handle.read().split()[1])
    return pages * 4096 / 1024**2


def percentiles(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "p50_ms": statistics.median(ordered) * 1000,
        "p95_ms": ordered[int(len(ordered) * 0.95) - 1] * 1000,
        "mean_ms": statistics.mean(ordered) * 1000,
        "n": len(ordered),
    }


# Deliberately spread across unrelated domains. Sequentially-numbered variants
# of one sentence do not work here: they differ by a single token and collide
# with each other above the threshold, which is the same failure the threshold
# sweep documents.
DISTINCT_QUERIES = [
    "What is the boiling point of water at sea level?",
    "Who composed the Brandenburg Concertos?",
    "How do I repot a fiddle leaf fig?",
    "What causes the aurora borealis?",
    "When was the Suez Canal opened?",
    "How does a heat pump work in winter?",
    "What is the offside rule in football?",
    "Which vitamins are fat soluble?",
    "How long does bread dough need to prove?",
    "What is the tallest mountain in Japan?",
    "How do noise cancelling headphones work?",
    "What language is spoken in Suriname?",
    "Why do cats purr?",
    "What is the capital of Uruguay?",
    "How is maple syrup harvested?",
    "What does a sommelier do?",
    "When did the Berlin Wall fall?",
    "How do submarines control buoyancy?",
    "What is the difference between fog and mist?",
    "Who painted the ceiling of the Sistine Chapel?",
]


def bench_query_latency(model_name: str, repeats: int = 40) -> dict:
    """End-to-end process_query latency, hit vs miss, with the real encoder.

    Samples are classified by the status actually returned rather than asserted,
    so an unexpected cache hit skews nothing -- it is simply counted as a hit.
    """
    store = LocalVectorStore(model_name)
    store.ingest_document("doc-1", DOC)
    gateway = SemanticCacheGateway(vector_store=store, similarity_threshold=DEFAULT_THRESHOLD)

    query = "How long must KYC records be kept?"
    gateway.process_query(query)  # warm the cache and the model

    hit_samples, miss_samples = [], []

    for _ in range(repeats):
        start = time.perf_counter()
        _, status = gateway.process_query(query)
        elapsed = time.perf_counter() - start
        (hit_samples if status.startswith("CACHE_HIT") else miss_samples).append(elapsed)

    for text in DISTINCT_QUERIES:
        start = time.perf_counter()
        _, status = gateway.process_query(text)
        elapsed = time.perf_counter() - start
        (hit_samples if status.startswith("CACHE_HIT") else miss_samples).append(elapsed)

    if not miss_samples:
        raise RuntimeError("no cache misses observed; latency comparison is meaningless")

    return {"hit": percentiles(hit_samples), "miss": percentiles(miss_samples)}


def bench_lookup_scaling(sizes: list[int], probes: int = 200) -> list[dict]:
    """Isolated cache-lookup cost as the cache grows. No encoding involved."""
    rng = np.random.default_rng(0)
    rows = []

    for size in sizes:
        # Pinned to the exact backend: this table is what a full scan costs, and
        # is the evidence for the bound that `build_index(..., "auto")` uses to
        # decide when to stop doing one. benchmarks/bench_ann.py measures the
        # alternative.
        cache = SemanticCache(
            threshold=2.0, max_entries=size, index_backend="exact"
        )  # never hits; measures scan
        vectors = rng.normal(size=(size, DIM)).astype(np.float32)

        gc.collect()
        before = rss_mb()
        for i in range(size):
            cache.insert(f"q-{i}", vectors[i], f"a-{i}")
        gc.collect()
        after = rss_mb()

        probe_vectors = rng.normal(size=(probes, DIM)).astype(np.float32)
        samples = []
        for probe in probe_vectors:
            start = time.perf_counter()
            cache.lookup(probe)
            samples.append(time.perf_counter() - start)

        buffer_mb = cache.nbytes / 1024**2
        rows.append(
            {
                "entries": size,
                "lookup": percentiles(samples),
                "vector_buffer_mb": buffer_mb,
                "rss_delta_mb": after - before,
                "bytes_per_entry": cache.nbytes / size,
            }
        )
        print(
            f"  n={size:>7,}  lookup p50={rows[-1]['lookup']['p50_ms']:7.4f} ms  "
            f"p95={rows[-1]['lookup']['p95_ms']:7.4f} ms  "
            f"buffer={buffer_mb:7.1f} MB  rss delta={after - before:7.1f} MB"
        )
        del cache, vectors, probe_vectors
        gc.collect()

    return rows


class SleepGenerator:
    """Stands in for a generator whose call costs real time.

    Not a model: a fixed sleep, so the number it produces is a clean statement
    of 'what a hit saves when the miss path costs X', with no hardware or
    sampling noise mixed in. TinyLlama on CPU is comfortably in the
    hundreds-of-milliseconds range for a short answer.
    """

    def __init__(self, delay_ms: float):
        self.delay_s = delay_ms / 1000.0

    def generate(self, query: str, context: str) -> str:
        time.sleep(self.delay_s)
        return context


def bench_generation_saving(model_name: str, delay_ms: float, repeats: int = 12) -> dict:
    """Hit vs miss latency when the miss path actually generates.

    This is the configuration the 'reduces inference cost' claim refers to.
    """
    store = LocalVectorStore(model_name)
    store.ingest_document("doc-1", DOC)
    gateway = SemanticCacheGateway(
        vector_store=store,
        similarity_threshold=DEFAULT_THRESHOLD,
        generator=SleepGenerator(delay_ms),
    )

    query = "How long must KYC records be kept?"
    gateway.process_query(query)  # warm

    hit_samples, miss_samples = [], []
    for _ in range(repeats):
        start = time.perf_counter()
        gateway.process_query(query)
        hit_samples.append(time.perf_counter() - start)

    for text in DISTINCT_QUERIES[:repeats]:
        start = time.perf_counter()
        _, status = gateway.process_query(text)
        elapsed = time.perf_counter() - start
        if status == "CACHE_MISS":
            miss_samples.append(elapsed)

    hit = percentiles(hit_samples)
    miss = percentiles(miss_samples)
    return {
        "simulated_generation_ms": delay_ms,
        "hit": hit,
        "miss": miss,
        "p50_speedup": miss["p50_ms"] / hit["p50_ms"],
    }


def bench_hit_rate(model_name: str) -> dict:
    """Hit rate over a mixed stream, not a hand-picked pair.

    The stream interleaves paraphrases (should hit) and near-misses (should
    not), so the measured rate reflects both.
    """
    store = LocalVectorStore(model_name)
    store.ingest_document("doc-1", DOC)
    gateway = SemanticCacheGateway(vector_store=store, similarity_threshold=DEFAULT_THRESHOLD)

    stream = []
    for first, second in PARAPHRASES:
        stream.extend([(first, "paraphrase"), (second, "paraphrase")])
    for first, second in NEAR_MISSES:
        stream.extend([(first, "near_miss"), (second, "near_miss")])

    hits = 0
    for text, _ in stream:
        _, status = gateway.process_query(text)
        if status.startswith("CACHE_HIT"):
            hits += 1

    return {
        "queries": len(stream),
        "hits": hits,
        "hit_rate": hits / len(stream),
        "threshold": DEFAULT_THRESHOLD,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="all-mpnet-base-v2")
    parser.add_argument("--max-scale", type=int, default=100_000)
    parser.add_argument(
        "--generation-ms",
        type=float,
        default=500.0,
        help="simulated cost of a generation call on the miss path",
    )
    args = parser.parse_args()

    sizes = [n for n in (100, 10_000, 100_000) if n <= args.max_scale]

    print("query latency (real encoder, end to end) ...")
    latency = bench_query_latency(args.model)
    print(
        f"  hit  p50={latency['hit']['p50_ms']:.3f} ms  p95={latency['hit']['p95_ms']:.3f} ms"
    )
    print(
        f"  miss p50={latency['miss']['p50_ms']:.3f} ms  p95={latency['miss']['p95_ms']:.3f} ms"
    )
    speedup = latency["miss"]["p50_ms"] / latency["hit"]["p50_ms"]
    print(f"  p50 speedup on a hit: {speedup:.1f}x")

    print(f"\nwith a generator on the miss path (simulated {args.generation_ms:.0f} ms) ...")
    generation = bench_generation_saving(args.model, args.generation_ms)
    print(
        f"  hit  p50={generation['hit']['p50_ms']:.3f} ms  "
        f"miss p50={generation['miss']['p50_ms']:.3f} ms  "
        f"speedup {generation['p50_speedup']:.1f}x"
    )

    print("\nlookup scaling (cache only, no encoding) ...")
    scaling = bench_lookup_scaling(sizes)

    print("\nhit rate over a mixed query stream ...")
    hit_rate = bench_hit_rate(args.model)
    print(
        f"  {hit_rate['hits']}/{hit_rate['queries']} = {hit_rate['hit_rate']:.1%} "
        f"at threshold {hit_rate['threshold']}"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "dim": DIM,
        "query_latency": latency,
        "hit_p50_speedup": speedup,
        "with_generation": generation,
        "lookup_scaling": scaling,
        "hit_rate": hit_rate,
    }
    (RESULTS_DIR / "benchmarks.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {RESULTS_DIR / 'benchmarks.json'}")


if __name__ == "__main__":
    main()
