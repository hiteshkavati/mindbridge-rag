import numpy as np
import pytest

from mindbridge.index import rrf


def test_rrf_prefers_items_ranked_well_by_both_lists():
    fused = rrf([[1, 2, 3], [3, 1, 4]], weights=[1, 1], k=60)
    ranked = sorted(fused, key=fused.get, reverse=True)
    assert ranked[0] == 1
    assert set(fused) == {1, 2, 3, 4}


def test_rrf_weights_shift_the_outcome():
    a = rrf([[1, 2], [2, 1]], weights=[3, 1])
    assert a[1] > a[2]
    b = rrf([[1, 2], [2, 1]], weights=[1, 3])
    assert b[2] > b[1]


@pytest.mark.parametrize(
    "query, expected",
    [
        ("how do I stop my mind racing when I try to sleep", "racing-thoughts-at-night"),
        ("what is box breathing", "slow-breathing"),
        ("my parents keep pressuring me about marriage", "family-expectations"),
        ("how to cope with grief after losing someone", "grief-and-loss"),
    ],
)
def test_obvious_questions_find_their_document(retriever, query, expected):
    top = [h.chunk.doc_id for h in retriever.search(query, k=3, per_doc=1)]
    assert expected in top


def test_per_doc_cap_is_respected(retriever):
    hits = retriever.search("anxiety panic breathing", k=8, per_doc=1)
    docs = [h.chunk.doc_id for h in hits]
    assert len(docs) == len(set(docs))


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_all_modes_return_scored_hits(retriever, mode):
    hits = retriever.search("exam anxiety revision", k=3, mode=mode)
    assert hits
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_dense_vectors_are_unit_length(retriever):
    norms = np.linalg.norm(retriever._matrix, axis=1)
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-6)


def test_nonsense_query_does_not_crash(retriever):
    assert retriever.search("zzzzqqq xxyyzz", k=3) is not None
    assert retriever.search("", k=3) is not None


def test_coverage_separates_on_topic_from_off_topic(retriever):
    assert retriever.coverage("I can't sleep and my mind keeps racing") > 0.6
    assert retriever.coverage("who won the cricket match yesterday") < 0.3
    assert retriever.coverage("") == 0.0


def test_is_supported_uses_dense_rescue_only_when_configured(retriever):
    q = "who won the cricket match yesterday"
    hits = retriever.search(q, k=3)
    assert not retriever.is_supported(q, hits, min_coverage=0.5)
    assert retriever.is_supported(q, hits, min_coverage=0.5, min_dense=-1.0)
    assert not retriever.is_supported(q, [], min_coverage=0.0)
