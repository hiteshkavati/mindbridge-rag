from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np


class BM25:
    """Okapi BM25 over a fixed, pre-tokenised corpus."""

    def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(corpus)
        self.doc_len = np.array([len(d) for d in corpus], dtype=np.float64)
        self.avg_len = float(self.doc_len.mean()) if self.n_docs else 0.0

        self.tf: list[Counter[str]] = [Counter(d) for d in corpus]
        df: Counter[str] = Counter()
        for counts in self.tf:
            df.update(counts.keys())
        self.idf = {
            term: math.log(1.0 + (self.n_docs - n + 0.5) / (n + 0.5)) for term, n in df.items()
        }

    def scores(self, query: Sequence[str]) -> np.ndarray:
        out = np.zeros(self.n_docs, dtype=np.float64)
        if not self.n_docs or not self.avg_len:
            return out
        norm = self.k1 * (1.0 - self.b + self.b * self.doc_len / self.avg_len)
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, counts in enumerate(self.tf):
                f = counts.get(term)
                if f:
                    out[i] += idf * f * (self.k1 + 1.0) / (f + norm[i])
        return out
