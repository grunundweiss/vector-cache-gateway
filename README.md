# vector-cache-gateway

[![CI](https://github.com/grunundweiss/vector-cache-gateway/actions/workflows/ci.yml/badge.svg)](https://github.com/grunundweiss/vector-cache-gateway/actions/workflows/ci.yml)

A similarity-gated cache in front of a local vector store, built to find out
whether that idea actually works. The short answer is: only under conditions
this README tries to state precisely.

Queries are embedded with `all-mpnet-base-v2`, compared against previously seen
queries by cosine similarity, and served from cache when the nearest one clears
a threshold. The threshold is chosen from a measured sweep, and the cost saving
is measured rather than asserted — both are below, including where they come
out badly.

---

## What a cache hit actually saves

| Configuration | Hit p50 | Miss p50 | Speedup |
|---|---|---|---|
| Retrieval-only (default) | 47.9 ms | 40.0 ms | **~1×** |
| With a 500 ms generation call | 60.0 ms | 539.6 ms | **9.0×** |

**With the default retrieval-only backend, a cache hit is not measurably faster
than a miss.** Both paths must embed the query before they can do anything, and
that ~40 ms embedding pass dominates the microseconds of vector search that a
hit avoids.

The cache is only worth having when the work it skips is expensive. Install a
generator and a hit skips generation entirely:

```python
from gateway import InferenceEngineGenerator, LocalVectorStore, SemanticCacheGateway

gateway = SemanticCacheGateway(
    vector_store=store,
    generator=InferenceEngineGenerator(engine),  # anything with generate_inference()
)
```

`gateway/generator.py` defines the protocol. The default `RetrievalOnlyGenerator`
returns the retrieved passage verbatim and needs no model server, which is what
makes the repo runnable standalone — and also why its default speedup is ~1×.

Full numbers and hardware: [`results/benchmarks.md`](results/benchmarks.md).

---

## Choosing the threshold

The threshold is the entire design. It is picked from measurement, not by hand.

[`benchmarks/threshold_sweep.py`](benchmarks/threshold_sweep.py) scores 25
hand-labeled paraphrase pairs (same question, different words — these *should*
hit) against 25 near-miss pairs (different question, similar surface — these
must *not*), using the real encoder.

| Threshold | Hit rate | False-hit rate | Margin |
|---|---|---|---|
| 0.70 | 84% | 64% | 20% |
| 0.75 | 72% | 52% | 20% |
| 0.80 | 72% | 36% | 36% |
| **0.85** | **60%** | **20%** | **40%** |
| 0.90 | 16% | 16% | 0% |

![Threshold sweep](results/threshold_sweep.png)

**This repo defaults to 0.85**, the widest margin measured. An earlier version
defaulted to 0.75 and called it a "strict safety threshold"; it serves **52% of
near-misses** — returning the wrong regulatory answer more often than a coin
flip, confidently, with no signal to the caller.

### The uncomfortable part

Even 0.85 leaves a 20% false-hit rate, and the two distributions **fully
overlap**: all 25 near-misses score above the weakest genuine paraphrase
(0.3571). No threshold separates them cleanly. The worst collisions:

| Similarity | Pair |
|---|---|
| 0.9792 | transfers over **500,000** NOK vs. over **50,000** NOK |
| 0.9520 | **maximum** retention period vs. **minimum** retention period |
| 0.9343 | MFA for **internal** transfers vs. **external** transfers |

A 10× difference in a reporting threshold, and a pair of exact opposites, scored
as near-identical. Sentence embeddings encode topic; they do not reliably encode
the numeric and polarity distinctions that decide a compliance answer.

**Cosine similarity alone is not a sufficient gate for this domain.** A
production version needs a cheap guard in front of the cache — numeric-literal
and negation/polarity comparison between query and cache key — treating the
threshold as a coarse first filter rather than the decision. That guard is not
implemented here; the sweep is what shows it is needed.

The near-miss set is deliberately adversarial, so 20% is a worst case rather
than a production estimate.

---

## Design notes

- **One embedding per query.** The query is encoded once and the vector is
  threaded through lookup, search and insert. An earlier version embedded the
  same string up to three times per miss. Enforced by
  [`tests/test_encode_budget.py`](tests/test_encode_budget.py).
- **Lookup is a single matmul** over L2-normalized float32 vectors, so cosine
  similarity is a plain dot product. Storage is 3,072 B/entry (768 × 4); the
  encoder emits float32, so float64 storage would double memory for precision
  that was never in the data.
- **Bounded by LRU**, default 10,000 entries. Eviction overwrites the victim
  row in place rather than compacting.
- **Invalidated on corpus change.** `LocalVectorStore` carries a version
  counter that `ingest_document` bumps; the gateway drops the whole cache when
  it changes. Blunt on purpose — deciding which entries a new document affects
  costs more than repopulating on demand. Right for a write-rarely corpus,
  wrong for a write-heavy one.

---

## Known limitations

- **Lookup is O(n).** 0.07 ms at 100 entries, 58.9 ms at 100k — past ~10k the
  cache costs more than the embedding pass it saves. Needs ANN (FAISS IVF/HNSW)
  to go further; the 10k default bound keeps it inside the range where a full
  scan is defensible.
- **No relevance floor on retrieval.** `similarity_search` returns the top-k
  documents however poor the match, so "What is the capital of France?" returns
  a banking document rather than nothing. A minimum-score cutoff is missing.
- **No numeric or polarity guard**, per the threshold section above. This is the
  most important gap.
- **In-memory only.** Nothing is persisted; the cache and index are rebuilt on
  every start.
- **Single process, not thread-safe.** Concurrent `process_query` calls can race
  on cache insert.
- **The test suite runs against a lexical fake encoder**, not the real model, so
  it can verify the plumbing and the gate but not semantic quality. Semantic
  behaviour is measured by the benchmarks instead.

---

## Install

```bash
git clone https://github.com/grunundweiss/vector-cache-gateway.git
cd vector-cache-gateway
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,bench]"
```

## Run

```bash
python -m gateway.demo                  # paraphrase hit + near-miss rejection
python -m benchmarks.threshold_sweep    # regenerate the sweep and chart
python -m benchmarks.bench_cache        # latency, memory, scaling
pytest -v                               # test suite (offline, no model download)
```

The demo and benchmarks download `all-mpnet-base-v2` on first run. The test
suite does not: it substitutes a deterministic lexical fake
(`tests/conftest.py`), so CI stays offline and fast.

## Development

```bash
ruff check . && mypy gateway && pytest -q
```

CI runs the suite on Python 3.10, 3.11 and 3.12, plus ruff and mypy.

## License

MIT — see [LICENSE](LICENSE).
