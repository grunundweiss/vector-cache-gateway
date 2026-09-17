# Benchmark results

Hardware: AMD Ryzen 7 5800X3D, 8 GB RAM allocated, virtual machine.
Encoder: `all-mpnet-base-v2`, 768 dimensions, float32. Python 3.14.7.
Reproduce with `python -m benchmarks.bench_cache`.

**This is the same CPU model, same RAM allocation and same "virtual machine"
description as the previous run of this benchmark** — see the note in §2, which
is the one place that matters enough to be a problem, because it rules out the
easy explanation for what follows.

## 1. A cache hit saves nothing without a generator

| Configuration | Hit p50 | Miss p50 | Speedup |
|---|---|---|---|
| Retrieval-only (default) | 39.8 ms | 39.4 ms | **0.99×** |
| With a 500 ms generation call | 40.3 ms | 540.4 ms | **13.4×** |

With the default `RetrievalOnlyGenerator`, **a cache hit is not measurably faster
than a miss** — it is nominally *slower*. Both paths must embed the query before
anything else can happen, and that ~39 ms embedding pass costs four orders of
magnitude more than the vector search a hit avoids. The two rows are the same
number with noise on top.

The cache only pays for itself when the work it skips is expensive. With a
generation call on the miss path it is a 13.4× improvement, and that gap widens
as generation gets slower. **The saving comes from skipping generation, not from
skipping retrieval.**

The 500 ms figure is a fixed sleep, not a model — a clean statement of "what a
hit saves when the miss path costs X", with no hardware noise mixed in. Wire in
a real engine via `InferenceEngineGenerator` for end-to-end numbers.

## 2. Lookup cost, and a cautionary tale about measuring it

| Entries | Lookup p50 | Lookup p95 | Vector buffer | RSS delta | Bytes/entry |
|---|---|---|---|---|---|
| 100 | 0.034 ms | 0.066 ms | 0.3 MB | ~0 MB | 3,072 |
| 10,000 | 0.156 ms | 0.195 ms | 29.3 MB | 34.2 MB | 3,072 |
| 100,000 | 13.95 ms | 16.82 ms | 293.0 MB | 348.5 MB | 3,072 |

**A previous run of this same benchmark reported 5.67 ms at 10k and 58.9 ms at
100k — 36× and 4× slower than these numbers, for the same code path.** The
obvious explanation — different machine — does not hold: both runs report the
same CPU model (5800X3D) and the same 8 GB VM allocation. Same nominal hardware,
same benchmark, same operation, 36× apart.

That makes this more useful as a cautionary note, not less. A reported CPU model
on a VM is not a guarantee of identical execution conditions: a hypervisor can
schedule the guest on different physical cores under different contention from
other tenants, and the two runs were not confirmed to be on the same numpy
build or BLAS backend (reference BLAS vs. OpenBLAS vs. a vendor library is
routinely a 10-50× difference on matrix multiplication alone, and both binary
identity and thread count were unrecorded on both runs). Nobody pinned down
which of these it actually was — that is the point being conceded, not a gap to
paper over.

The old README built an argument on that old number: *"at 100k entries a lookup
costs 58.9 ms, which is more than the embedding pass it was supposed to be
cheaper than — past roughly 10k the cache becomes the thing you need to
optimize."* **On this run that conclusion is simply false.** At 100k the lookup
costs 13.95 ms against a ~39 ms embedding pass; the scan never becomes the
bottleneck anywhere in the measured range. An architectural argument was resting
on one unvalidated number, and revalidating it on the same reported hardware
produced a different answer.

The scaling is also not the clean O(n) the old table implied. From 100 to 10,000
entries — 100× the data — lookup grows 4.6×. From 10,000 to 100,000 — 10× the
data — it grows 89×. The vector buffer crosses from 29 MB to 293 MB somewhere in
that gap, which is the difference between living in cache and streaming from
main memory. The matmul is memory-bandwidth-bound long before it is
arithmetic-bound, so "O(n)" describes the operation count and not the wall
clock.

Memory is 3,072 bytes per entry for the vectors, exactly 768 × 4, confirming
float32 storage. Measured RSS runs ~17% above the raw buffer: the Python-side
keys, answers and index dict.

`ann.md` measures the approximate-nearest-neighbour backends against the same
sizes. They are still 16.6× faster than the scan at 100k at full recall — but
with these numbers, ANN buys headroom past 100k rather than rescuing a scan that
was already too slow at 10k.

## 3. Hit rate on a mixed stream

**37%** (37 of 100 queries) at threshold 0.85 with the default guard, over a
stream interleaving all 25 paraphrase pairs and all 25 near-miss pairs.

The same stream measured **42%** before the guard existed. The guard costs 5
hits out of 100. The threshold sweep says the false-hit rate at 0.85 goes from
20% to 0% when the guard is switched on, so the arithmetic is consistent with
those 5 lost hits having been wrong answers — though the stream also allows
hits across unrelated pairs, so this is consistency rather than proof.

Do not read 37% as a production hit rate either way. The stream is half
adversarial near-misses by construction. It is a floor for a deliberately
hostile workload, not an estimate for real traffic.
