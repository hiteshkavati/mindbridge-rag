from mindbridge.bm25 import BM25
from mindbridge.text import stem, tokenize


def test_stem_merges_common_inflections():
    assert stem("stressed") == stem("stressing") == stem("stress")
    assert stem("thoughts") == "thought"
    assert stem("anxiety") == "anxiety"


def test_stem_leaves_short_words_alone():
    assert stem("bus") == "bus"
    assert stem("is") == "is"


def test_tokenize_drops_stopwords_and_folds_case():
    toks = tokenize("The Thoughts are RACING")
    assert "the" not in toks
    assert toks[0] == "thought"
    assert "racing" in toks or "rac" in toks


def test_tokenize_handles_curly_apostrophes():
    assert tokenize("can\u2019t") == tokenize("can't")


def test_bm25_ranks_matching_document_first():
    corpus = [
        tokenize(t)
        for t in ["sleep routine and caffeine", "exam stress revision", "grief and loss"]
    ]
    bm = BM25(corpus)
    scores = bm.scores(tokenize("caffeine before sleep"))
    assert scores.argmax() == 0
    assert scores[1] == 0 and scores[2] == 0


def test_bm25_unknown_terms_score_zero():
    bm = BM25([["a", "b"], ["c"]])
    assert bm.scores(["zzz"]).sum() == 0


def test_bm25_empty_corpus_is_safe():
    assert BM25([]).scores(["x"]).shape == (0,)


def test_bm25_rewards_rarer_terms():
    bm = BM25([["common", "rare"], ["common"], ["common"], ["common"]])
    assert bm.scores(["rare"])[0] > bm.scores(["common"])[0]
