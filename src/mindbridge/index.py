from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .bm25 import BM25
from .embeddings import Embedder
from .kb import Chunk
from .text import tokenize

Mode = Literal["hybrid", "bm25", "dense"]
_POOL = 40


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    dense: float
    bm25: float


def rrf(rankings: Sequence[Sequence[int]], weights: Sequence[float], k: int = 60) -> dict[int, float]:
    """Reciprocal rank fusion. Each ranking is a list of item ids, best first."""
    fused: dict[int, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking):
            fused[item] += weight / (k + rank + 1)
    return dict(fused)


class HybridRetriever:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: Embedder,
        *,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
    ):
        self.chunks = list(chunks)
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.weights = (bm25_weight, dense_weight)

        texts = [c.index_text for c in self.chunks]
        embedder.fit(texts)
        self._matrix = embedder.encode(texts)
        self._bm25 = BM25([tokenize(t) for t in texts])

    def search(
        self, query: str, k: int = 5, *, mode: Mode = "hybrid", per_doc: int = 2
    ) -> list[Hit]:
        bm = self._bm25.scores(tokenize(query))
        cos = self._matrix @ self.embedder.encode([query])[0]

        # a zero BM25 score carries no ordering information, so it stays out of the ranking
        bm_rank = [int(i) for i in np.argsort(-bm)[:_POOL] if bm[i] > 0]
        dense_rank = [int(i) for i in np.argsort(-cos)[:_POOL]]

        if mode == "bm25":
            scored = {i: float(bm[i]) for i in bm_rank}
        elif mode == "dense":
            scored = {i: float(cos[i]) for i in dense_rank}
        else:
            scored = rrf([bm_rank, dense_rank], self.weights, self.rrf_k)

        hits: list[Hit] = []
        seen: dict[str, int] = defaultdict(int)
        for i, score in sorted(scored.items(), key=lambda kv: -kv[1]):
            chunk = self.chunks[i]
            if seen[chunk.doc_id] >= per_doc:
                continue
            seen[chunk.doc_id] += 1
            hits.append(Hit(chunk, score, float(cos[i]), float(bm[i])))
            if len(hits) == k:
                break
        return hits

    def chunks_for(self, doc_id: str) -> list[Chunk]:
        return [c for c in self.chunks if c.doc_id == doc_id]

    def coverage(self, query: str) -> float:
        """IDF-weighted share of the query's content words that occur in the knowledge base.

        Unknown words count at the highest IDF, so a query made of words the KB has never
        seen scores near zero. Dense similarity alone is a poor out-of-scope signal with
        small corpora (it happily matches "today" in "price of gold today"), so this is the
        primary gate.
        """
        tokens = tokenize(query)
        if not tokens:
            return 0.0
        idf = self._bm25.idf
        worst = max(idf.values())
        known = sum(idf.get(t, 0.0) for t in tokens)
        return known / sum(idf.get(t, worst) for t in tokens)

    def is_supported(
        self,
        query: str,
        hits: Sequence[Hit],
        min_coverage: float,
        min_dense: float | None = None,
    ) -> bool:
        """True when the knowledge base can plausibly answer the query.

        A strong dense match (only meaningful with a real sentence-embedding model)
        can rescue a paraphrase whose wording the KB does not share.
        """
        if not hits:
            return False
        if self.coverage(query) >= min_coverage:
            return True
        return min_dense is not None and max(h.dense for h in hits) >= min_dense
