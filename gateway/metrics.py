"""Counters for the three questions a live semantic cache has to answer.

    Is it saving anything?    hit rate, estimated latency saved
    Is it serving garbage?    mean similarity of hits, guard veto counts
    Is the threshold right?   the distribution of hit similarities, not just the rate

A benchmark answers these once, on a labeled set, on one machine. Production
answers them continuously, on traffic nobody labeled. A cache with no metrics
cannot distinguish "threshold too tight, saving nothing" from "threshold too
loose, quietly serving wrong answers" -- both look like a system that is up.

Latency samples are held in bounded deques rather than a growing list: a
long-lived process must not accumulate one float per request forever, which is
the same memory-leak-with-a-nice-name the cache itself is bounded against.
"""

import threading
from collections import Counter, deque

_SAMPLE_SIZE = 1024


def _percentile(samples: list[float], fraction: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    index = min(int(len(ordered) * fraction), len(ordered) - 1)
    return ordered[index]


def _summary(samples: deque) -> dict[str, float]:
    values = list(samples)
    if not values:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0, "n": 0}
    return {
        "p50_ms": _percentile(values, 0.5),
        "p95_ms": _percentile(values, 0.95),
        "mean_ms": sum(values) / len(values),
        "n": len(values),
    }


class CacheMetrics:
    """Thread-safe counters and bounded latency/similarity samples.

    Every method is cheap enough to call on the request path: an append to a
    bounded deque and a few integer increments, under a lock held for the length
    of that. Nothing here does I/O -- exporting is the caller's decision, via
    ``snapshot`` or ``prometheus``.
    """

    def __init__(self, sample_size: int = _SAMPLE_SIZE):
        self._lock = threading.Lock()
        self.sample_size = sample_size
        self.queries = 0
        self.hits = 0
        self.misses = 0
        self.bypasses = 0
        self.guard_vetoes = 0
        self.expired = 0
        self.evictions = 0
        self.invalidations = 0
        self.feedback_evictions = 0
        self.corpus_invalidations = 0
        self.veto_reasons: Counter = Counter()
        self.hit_scores: deque = deque(maxlen=sample_size)
        self.hit_latency_ms: deque = deque(maxlen=sample_size)
        self.miss_latency_ms: deque = deque(maxlen=sample_size)

    def record_hit(self, score: float, latency_ms: float) -> None:
        with self._lock:
            self.queries += 1
            self.hits += 1
            self.hit_scores.append(float(score))
            self.hit_latency_ms.append(float(latency_ms))

    def record_miss(self, latency_ms: float) -> None:
        with self._lock:
            self.queries += 1
            self.misses += 1
            self.miss_latency_ms.append(float(latency_ms))

    def record_bypass(self, reason: str = "") -> None:
        """A request that was never eligible for the cache -- a tool call, say."""
        with self._lock:
            self.queries += 1
            self.bypasses += 1
            if reason:
                self.veto_reasons[f"bypass:{reason}"] += 1

    def record_guard_veto(self, reason: str) -> None:
        with self._lock:
            self.guard_vetoes += 1
            self.veto_reasons[reason.split(":", 1)[0] or "unknown"] += 1

    def record_expired(self, count: int = 1) -> None:
        with self._lock:
            self.expired += count

    def record_eviction(self, count: int = 1) -> None:
        with self._lock:
            self.evictions += count

    def record_invalidation(self, count: int = 1, *, feedback: bool = False) -> None:
        with self._lock:
            self.invalidations += count
            if feedback:
                self.feedback_evictions += count

    def record_corpus_invalidation(self, dropped: int) -> None:
        with self._lock:
            self.corpus_invalidations += 1
            self.invalidations += dropped

    @property
    def hit_rate(self) -> float:
        """Hits over cacheable queries. Bypasses are excluded: a tool call was
        never a cache candidate, and counting it as a miss makes the cache look
        worse than it is while hiding how much traffic it never sees."""
        cacheable = self.hits + self.misses
        return self.hits / cacheable if cacheable else 0.0

    def snapshot(self) -> dict:
        with self._lock:
            scores = list(self.hit_scores)
            hit = _summary(self.hit_latency_ms)
            miss = _summary(self.miss_latency_ms)
            cacheable = self.hits + self.misses
            saved_per_hit = max(0.0, miss["p50_ms"] - hit["p50_ms"])
            return {
                "queries": self.queries,
                "hits": self.hits,
                "misses": self.misses,
                "bypasses": self.bypasses,
                "hit_rate": self.hits / cacheable if cacheable else 0.0,
                "guard_vetoes": self.guard_vetoes,
                "veto_reasons": dict(self.veto_reasons),
                "expired": self.expired,
                "evictions": self.evictions,
                "invalidations": self.invalidations,
                "feedback_evictions": self.feedback_evictions,
                "corpus_invalidations": self.corpus_invalidations,
                "hit_similarity": {
                    "mean": sum(scores) / len(scores) if scores else 0.0,
                    "min": min(scores) if scores else 0.0,
                    "p50": _percentile(scores, 0.5),
                    "n": len(scores),
                },
                "hit_latency": hit,
                "miss_latency": miss,
                # Counterfactual: what those hits would have cost at the measured
                # median miss latency. An estimate, and labeled as one -- the
                # miss that did not happen was never timed.
                "estimated_ms_saved": self.hits * saved_per_hit,
            }

    def render(self) -> str:
        """One-screen summary. What you want in a log line, not a dashboard."""
        s = self.snapshot()
        similarity = s["hit_similarity"]
        lines = [
            f"queries={s['queries']} hits={s['hits']} misses={s['misses']} "
            f"bypasses={s['bypasses']} hit_rate={s['hit_rate']:.1%}",
            f"hit similarity  mean={similarity['mean']:.4f} min={similarity['min']:.4f} "
            f"p50={similarity['p50']:.4f} n={similarity['n']}",
            f"latency         hit p50={s['hit_latency']['p50_ms']:.2f} ms  "
            f"miss p50={s['miss_latency']['p50_ms']:.2f} ms  "
            f"estimated saved={s['estimated_ms_saved'] / 1000:.2f} s",
            f"rejections      guard={s['guard_vetoes']} expired={s['expired']} "
            f"evictions={s['evictions']} invalidations={s['invalidations']} "
            f"(feedback={s['feedback_evictions']})",
        ]
        if s["veto_reasons"]:
            reasons = ", ".join(f"{k}={v}" for k, v in sorted(s["veto_reasons"].items()))
            lines.append(f"veto reasons    {reasons}")
        return "\n".join(lines)

    def prometheus(self, prefix: str = "semantic_cache") -> str:
        """Text-format exposition, so this can be scraped without a client library.

        Deliberately not a dependency: the format is six lines of string
        formatting, and pulling in a metrics client to emit counters a cache
        already holds is not a trade worth making.
        """
        s = self.snapshot()
        lines = [
            f"# TYPE {prefix}_queries_total counter",
            f"{prefix}_queries_total {s['queries']}",
            f"# TYPE {prefix}_hits_total counter",
            f"{prefix}_hits_total {s['hits']}",
            f"# TYPE {prefix}_misses_total counter",
            f"{prefix}_misses_total {s['misses']}",
            f"# TYPE {prefix}_bypasses_total counter",
            f"{prefix}_bypasses_total {s['bypasses']}",
            f"# TYPE {prefix}_guard_vetoes_total counter",
            f"{prefix}_guard_vetoes_total {s['guard_vetoes']}",
            f"# TYPE {prefix}_invalidations_total counter",
            f"{prefix}_invalidations_total {s['invalidations']}",
            f"# TYPE {prefix}_hit_rate gauge",
            f"{prefix}_hit_rate {s['hit_rate']:.6f}",
            f"# TYPE {prefix}_hit_similarity_mean gauge",
            f"{prefix}_hit_similarity_mean {s['hit_similarity']['mean']:.6f}",
            f"# TYPE {prefix}_hit_latency_p50_ms gauge",
            f"{prefix}_hit_latency_p50_ms {s['hit_latency']['p50_ms']:.6f}",
            f"# TYPE {prefix}_miss_latency_p50_ms gauge",
            f"{prefix}_miss_latency_p50_ms {s['miss_latency']['p50_ms']:.6f}",
        ]
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        with self._lock:
            self.queries = self.hits = self.misses = self.bypasses = 0
            self.guard_vetoes = self.expired = self.evictions = 0
            self.invalidations = self.feedback_evictions = self.corpus_invalidations = 0
            self.veto_reasons.clear()
            self.hit_scores.clear()
            self.hit_latency_ms.clear()
            self.miss_latency_ms.clear()
