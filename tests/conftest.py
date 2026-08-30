import re
import zlib

import numpy as np
import pytest


class FakeSentenceTransformer:
    """Deterministic offline stand-in that preserves lexical similarity.

    Hashes tokens into a fixed-width bag-of-words vector, so texts sharing
    vocabulary land close together and paraphrase behaviour is testable.

    Not semantically faithful -- it cannot model synonyms, so "car" and
    "automobile" are orthogonal under this fake where a real encoder would put
    them close. It exercises the similarity gate, which is what the previous
    random-vector fake could not do: random high-dimensional vectors are
    near-orthogonal, so every distinct string scored near zero and no
    paraphrase could ever clear the threshold.
    """

    _DIM = 256

    def __init__(self, *_args, **_kwargs):
        pass

    def encode(self, text, convert_to_numpy=True):
        vec = np.zeros(self._DIM)
        for token in re.findall(r"\w+", text.lower()):
            vec[zlib.crc32(token.encode("utf-8")) % self._DIM] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec


@pytest.fixture(autouse=True)
def fake_transformer(monkeypatch):
    monkeypatch.setattr(
        "gateway.vector_store.SentenceTransformer", FakeSentenceTransformer
    )
