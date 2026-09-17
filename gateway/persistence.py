"""Snapshotting the cache to disk, so a restart is not a cold start.

An in-memory cache loses everything on deploy. For a benchmark that is
irrelevant; for anything claiming to save inference cost over time it is the
whole claim, because the savings curve restarts at zero every time the process
does. A service that deploys twice a day and takes an hour to warm up spends a
meaningful fraction of its life paying full price.

Format: one ``.npz`` holding the vectors and parallel string arrays, plus a JSON
header. Explicitly **not** pickle. Loading a pickle executes whatever is in it,
and a cache file is exactly the kind of artifact that gets copied between
environments, restored from a backup, or mounted from somewhere you did not
write to. ``np.load(..., allow_pickle=False)`` cannot be talked into running
code.

What is stored is plaintext: the cache keys are user questions and the values
are answers derived from them. In a multi-tenant deployment that file is a
database of user content at rest, with the same handling requirements as any
other, and this module does nothing to encrypt it. Where that matters, put the
snapshot somewhere encrypted and say so in the deployment, rather than assuming
a cache file is ephemeral because the cache is.

A snapshot is refused on load when it does not match the running system --
different encoder, different dimensionality, different corpus version. Vectors
from another model are not comparable to vectors from this one; serving them
would be a silent, permanent false-hit generator.
"""

import json
import logging
import os
import time
from pathlib import Path

import numpy as np

from gateway.partition import PartitionedCache

logger = logging.getLogger(__name__)

SNAPSHOT_FORMAT = 1


class SnapshotMismatch(ValueError):
    """The snapshot does not belong to this cache: wrong model, dim or corpus."""


def save_snapshot(
    cache: PartitionedCache,
    path: str | Path,
    *,
    model: str,
    corpus_version: int | None = None,
    now: float | None = None,
) -> int:
    """Writes every live entry to ``path``. Returns how many were written.

    The write is atomic: a temporary file in the same directory, then
    ``os.replace``. A snapshot half-written by a process that died mid-deploy is
    a file that loads, validates, and serves truncated nonsense.
    """
    path = Path(path)
    now = time.time() if now is None else now

    vectors, namespaces, keys, answers, entry_ids, expires = [], [], [], [], [], []
    for namespace in cache.namespaces():
        partition = cache.partition(namespace, create=False)
        if partition is None:
            continue
        for entry in partition.entries():
            vectors.append(entry["vector"])
            namespaces.append(namespace)
            keys.append(entry["key"])
            answers.append(entry["answer"])
            entry_ids.append(entry["entry_id"])
            expires.append(np.nan if entry["expires_at"] is None else entry["expires_at"])

    matrix = (
        np.vstack(vectors).astype(np.float32)
        if vectors
        else np.zeros((0, 0), dtype=np.float32)
    )
    header = {
        "format": SNAPSHOT_FORMAT,
        "model": model,
        "dim": int(matrix.shape[1]) if matrix.size else 0,
        "corpus_version": corpus_version,
        "entries": len(keys),
        "saved_at": now,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with tmp.open("wb") as handle:
        np.savez(
            handle,
            meta=np.array(json.dumps(header)),
            vectors=matrix,
            namespaces=np.array(namespaces, dtype=np.str_),
            keys=np.array(keys, dtype=np.str_),
            answers=np.array(answers, dtype=np.str_),
            entry_ids=np.array(entry_ids, dtype=np.str_),
            expires_at=np.array(expires, dtype=np.float64),
        )
    os.replace(tmp, path)
    logger.info("wrote cache snapshot to %s (%d entries)", path, len(keys))
    return len(keys)


def read_header(path: str | Path) -> dict:
    """Returns a snapshot's header without loading its vectors."""
    with np.load(Path(path), allow_pickle=False) as data:
        return json.loads(str(data["meta"]))


def load_snapshot(
    cache: PartitionedCache,
    path: str | Path,
    *,
    model: str,
    corpus_version: int | None = None,
    strict: bool = True,
    now: float | None = None,
) -> int:
    """Restores entries into ``cache``. Returns how many were restored.

    Entries whose deadline has already passed are dropped rather than restored
    and immediately expired -- the deadlines are absolute, so a snapshot that
    sat on disk over a weekend comes back mostly empty, which is the point of
    having set a TTL.

    ``strict=False`` downgrades a mismatched snapshot from an exception to a
    warning and an empty cache, for the deployment that would rather start cold
    than fail to start.
    """
    path = Path(path)
    now = time.time() if now is None else now

    with np.load(path, allow_pickle=False) as data:
        header = json.loads(str(data["meta"]))
        problem = _mismatch(header, model=model, corpus_version=corpus_version)
        if problem is not None:
            if strict:
                raise SnapshotMismatch(f"{path}: {problem}")
            logger.warning("ignoring cache snapshot %s: %s", path, problem)
            return 0

        vectors = data["vectors"]
        namespaces = data["namespaces"]
        keys = data["keys"]
        answers = data["answers"]
        entry_ids = data["entry_ids"]
        expires_at = data["expires_at"]

    restored = 0
    skipped = 0
    for i in range(len(keys)):
        deadline = None if np.isnan(expires_at[i]) else float(expires_at[i])
        if deadline is not None and deadline <= now:
            skipped += 1
            continue
        partition = cache.partition(str(namespaces[i]))
        assert partition is not None
        partition.insert(
            str(keys[i]),
            vectors[i],
            str(answers[i]),
            entry_id=str(entry_ids[i]),
            expires_at=deadline,
            now=now,
        )
        restored += 1

    logger.info(
        "restored %d entries from %s (%d already expired)", restored, path, skipped
    )
    return restored


def _mismatch(header: dict, *, model: str, corpus_version: int | None) -> str | None:
    if header.get("format") != SNAPSHOT_FORMAT:
        return f"format {header.get('format')} != {SNAPSHOT_FORMAT}"
    if header.get("model") != model:
        return f"encoder {header.get('model')!r} != {model!r}"
    if corpus_version is not None and header.get("corpus_version") not in (None, corpus_version):
        return f"corpus version {header.get('corpus_version')} != {corpus_version}"
    return None
