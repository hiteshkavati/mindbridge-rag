from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .text import tokenize

log = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str

    def fit(self, texts: Sequence[str]) -> None: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalised vectors, one row per text."""
        ...


def _normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class LsaEmbedder:
    """TF-IDF followed by truncated SVD (latent semantic analysis).

    Needs no model download, so it is what the tests and the offline eval use.
    It captures co-occurrence within the knowledge base only; it will not know
    that two unrelated words are synonyms unless the corpus shows it.
    """

    name = "lsa"

    def __init__(self, dims: int = 96, seed: int = 0):
        self.dims = dims
        self.seed = seed
        self._vec: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None

    def fit(self, texts: Sequence[str]) -> None:
        self._vec = TfidfVectorizer(
            tokenizer=tokenize,
            lowercase=False,
            token_pattern=None,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        x = self._vec.fit_transform(texts)
        n = max(2, min(self.dims, x.shape[0] - 1, x.shape[1] - 1))
        self._svd = TruncatedSVD(n_components=n, random_state=self.seed)
        self._svd.fit(x)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._vec is None or self._svd is None:
            raise RuntimeError("LsaEmbedder.fit() must be called first")
        return _normalise(self._svd.transform(self._vec.transform(texts)))


class SbertEmbedder:
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Run: pip install 'mindbridge-rag[sbert]'"
            ) from exc
        self.name = f"sbert:{model_name}"
        self._model = SentenceTransformer(model_name)

    def fit(self, texts: Sequence[str]) -> None:
        return None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


def make_embedder(backend: str, sbert_model: str) -> Embedder:
    if backend == "lsa":
        return LsaEmbedder()
    if backend == "sbert":
        return SbertEmbedder(sbert_model)
    try:
        return SbertEmbedder(sbert_model)
    except Exception as exc:  # missing package or no network for the model download
        log.warning("falling back to LSA embeddings (%s)", exc)
        return LsaEmbedder()
