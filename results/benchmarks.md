# Benchmark results

Hardware: AMD Ryzen 7 5800X3D (8 cores), 7.9 GB RAM, CPU only — no GPU.
Encoder: `all-mpnet-base-v2`, 768 dimensions, float32.
Reproduce with `python -m benchmarks.bench_cache`.

## 1. A cache hit saves nothing without a generator

| Configuration | Hit p50 | Miss p50 | Speedup |
|---|---|---|---|
| Retrieval-only (default) | 47.9 ms | 40.0 ms | **~1×** |
| With a 500 ms generation call | 60.0 ms | 539.6 ms | **9.0×** |

This is the most important number here, and it is not flattering.

With the default `RetrievalOnlyGenerator`, **a cache hit is not measurably faster
than a miss**. Both paths must embed the query before anything else can happen,
and that embedding pass costs ~40–48 ms while the vector search it avoids costs
microseconds. The two rows are within noise of each other — the hit row is even
nominally slower, because those samples reuse a longer query string than the
miss samples, which is itself a reminder of how completely encoding dominates.

The cache only pays for itself when the work it skips is expensive. With a
generation call on the miss path it is a 9× improvement, and that gap widens as
generation gets slower. **The saving comes from skipping generation, not from
skipping retrieval.**

The 500 ms figure is a fixed sleep, not a model — a clean statement of "what a
hit saves when the miss path costs X", without hardware noise mixed in. Wire in
a real engine via `InferenceEngineGenerator` for end-to-end numbers.

## 2. Lookup cost grows linearly and becomes the bottleneck

| Entries | Lookup p50 | Lookup p95 | Vector buffer | RSS delta | Bytes/entry |
|---|---|---|---|---|---|
| 100 | 0.069 ms | 0.120 ms | 0.3 MB | ~0 MB | 3,072 |
| 10,000 | 5.67 ms | 6.46 ms | 29.3 MB | 34.8 MB | 3,072 |
| 100,000 | 58.9 ms | 61.7 ms | 293.0 MB | 341.7 MB | 3,072 |

The lookup is a single matmul, but it is still O(n): every cached vector is
scored on every query. At 100k entries a lookup costs **58.9 ms**, which is more
than the embedding pass it was supposed to be cheaper than. Past roughly 10k
entries the cache becomes the thing you need to optimize.

The fix is approximate nearest neighbour (FAISS IVF or HNSW) instead of a full
scan, trading exactness for sublinear lookup. That is not implemented here; the
`max_entries` bound (default 10k) keeps the cache inside the range where a full
scan is still reasonable, which is a deliberate limit rather than a solution.

Memory is 3,072 bytes per entry for the vectors — exactly 768 × 4, confirming
float32 storage. Measured RSS runs ~17% above the raw buffer, which is the
Python-side keys, answers and index dict.

## 3. Hit rate on a mixed stream

**42%** (42 of 100 queries) at threshold 0.85, over a stream interleaving all 25
paraphrase pairs and all 25 near-miss pairs.

Do not read this as a production hit rate. The stream is half adversarial
near-misses by construction, so some of those 42 hits are the *false* hits
quantified in the threshold sweep, not wins. It is a floor for a
deliberately hostile workload, not an estimate for real traffic.
