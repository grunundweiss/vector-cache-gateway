# Approximate nearest neighbour: what it costs and what it buys

Hardware: Intel Xeon @ 2.10 GHz (4 cores), 15 GB RAM, container.
768 dimensions, float32, 200 probes, k = 5 candidates — the number the cache
asks for on every lookup.
Reproduce with `python -m benchmarks.bench_ann`.

**This is a different machine from [`benchmarks.md`](benchmarks.md)**, and the
two disagree in both directions. Exact scan at 10k: 0.342 ms here, 0.156 ms
there. At 100k: 4.10 ms here, 13.95 ms there. The same operation, the same code,
one machine twice as slow at the small size and three times faster at the large
one — because 293 MB of vectors fits differently in two cache hierarchies.

Treat that as the headline finding of both files. BLAS, core count and memory
bandwidth move these numbers more than any choice in `gateway/index.py` does,
and a scaling table from someone else's machine is not evidence about yours.

## Clustered vectors — the case a cache actually sees

Query embeddings arrive in topic clusters. This is the distribution that matters.

| Entries | Backend | Recall@5 | Lookup p50 | Speedup | Build | RSS delta |
|---|---|---|---|---|---|---|
| 1,000 | exact | 1.000 | 0.026 ms | 1× | 0.00 s | 2.9 MB |
| 1,000 | hnsw | 1.000 | 0.067 ms | **0.4×** | 0.17 s | 7.3 MB |
| 1,000 | faiss-ivf | 1.000 | 0.023 ms | 1.1× | 0.00 s | 0.0 MB |
| 10,000 | exact | 1.000 | 0.342 ms | 1× | 0.03 s | 74.3 MB |
| 10,000 | hnsw | 1.000 | 0.141 ms | 2.4× | 4.01 s | 102.4 MB |
| 10,000 | faiss-ivf | 1.000 | 0.126 ms | 2.7× | 0.31 s | 110.3 MB |
| 100,000 | exact | 1.000 | 4.097 ms | 1× | 0.26 s | 308.0 MB |
| 100,000 | hnsw | 1.000 | **0.247 ms** | **16.6×** | 72.16 s | 572.3 MB |
| 100,000 | faiss-ivf | 1.000 | 0.639 ms | 6.4× | 8.02 s | 633.7 MB |

Three things follow, and only the first is the one people expect:

**At 1,000 entries HNSW is slower than scanning.** 0.067 ms against 0.026 ms.
The graph traversal is Python-call-bound at that size while the scan is a single
BLAS call. An ANN index is not a free upgrade; below a few thousand entries it
is a downgrade, which is why `build_index(..., "auto")` keeps the exact scan at
the default cache size.

**At 100,000 the scan costs 4.1 ms and HNSW costs 0.25 ms, at full recall.**
No hit was lost to approximation at any size on this distribution. That is the
case for ANN, and it is a strong one — once the cache is big enough for the
question to arise.

**The index is not free in memory or in build time.** Every backend keeps a
float32 mirror so candidates can be rescored exactly; the ANN structure is a
second copy on top of it, roughly doubling vector memory. HNSW took 72 seconds
to build 100k entries one at a time, which is irrelevant for a cache that fills
incrementally and very relevant for restoring a snapshot.

## Uniform random vectors — the adversarial case

The same measurement on vectors with no neighbourhood structure at all.

| Entries | Backend | Recall@5 | Lookup p50 |
|---|---|---|---|
| 10,000 | hnsw | 0.375 | 0.297 ms |
| 10,000 | faiss-ivf | 0.175 | 0.134 ms |
| 100,000 | hnsw | **0.070** | 0.514 ms |
| 100,000 | faiss-ivf | 0.090 | 0.593 ms |

Recall collapses. In 768 uniform dimensions every pair of vectors is
near-orthogonal, distances concentrate, and there is no structure for a graph or
a partition to exploit — so the index returns the wrong neighbours quickly
instead of the right ones slowly.

This is not a reason to avoid ANN, because embeddings are not uniform noise. It
is a reason to measure recall on **your** vectors rather than trusting a
benchmark, including this one: a 7% recall index looks exactly like a
well-behaved cache from the outside. Every hit it fails to find is reported as a
miss, the answer is regenerated, and the only visible symptom is a hit rate that
is quietly lower than it should be.

## Why the cache rescores candidates exactly

Both ANN backends return candidate *slots*; the cache then computes the cosine
similarity itself from its own float32 mirror before applying the threshold.
That costs k dot products — 5 of them — and it means the threshold has the same
meaning on every backend. Switching to ANN changes which entries are found. It
never changes the score an entry is judged by.
