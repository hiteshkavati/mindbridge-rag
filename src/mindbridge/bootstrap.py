from __future__ import annotations

import logging

from .config import Settings
from .embeddings import make_embedder
from .index import HybridRetriever
from .kb import build_chunks
from .llm import make_llm
from .pipeline import Pipeline
from .safety import load_helplines

log = logging.getLogger(__name__)


def build_retriever(settings: Settings, backend: str | None = None) -> HybridRetriever:
    chunks = build_chunks(settings.kb_dir)
    embedder = make_embedder(backend or settings.embedding_backend, settings.sbert_model)
    log.info("indexed %d chunks with %s", len(chunks), embedder.name)
    return HybridRetriever(
        chunks,
        embedder,
        rrf_k=settings.rrf_k,
        bm25_weight=settings.bm25_weight,
        dense_weight=settings.dense_weight,
    )


def build_pipeline(settings: Settings) -> Pipeline:
    return Pipeline(
        build_retriever(settings),
        make_llm(settings),
        load_helplines(settings.helplines_path),
        settings,
    )
