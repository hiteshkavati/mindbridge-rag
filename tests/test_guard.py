import pytest

from mindbridge.guard import check, clean_citations


@pytest.mark.parametrize(
    "text, rule",
    [
        ("You could try sertraline for this.", "medication"),
        ("Take 50 mg before bed.", "dosage"),
        ("You should stop taking your medication.", "treatment-change"),
        ("You probably have depression.", "diagnosis"),
        ("This is clearly an anxiety disorder.", "diagnosis"),
        ("This technique is guaranteed to work.", "guarantee"),
    ],
)
def test_unsafe_output_is_caught(text, rule):
    assert rule in check(text)


@pytest.mark.parametrize(
    "text",
    [
        "It sounds like you have a lot on your plate. Try a short walk [1].",
        "If this has lasted weeks, a doctor or Tele-MANAS (14416) can help.",
        "Depression is common and treatable, so it is worth talking to someone.",
    ],
)
def test_ordinary_supportive_text_passes(text):
    assert check(text) == []


def test_invalid_citations_are_removed_and_spacing_repaired():
    out = clean_citations("Try breathing [1] and a walk [7]. Also rest [2].", n_sources=2)
    assert "[7]" not in out
    assert "[1]" in out and "[2]" in out
    assert "  " not in out
    assert " ." not in out


def test_zero_sources_strips_every_marker():
    assert clean_citations("Do this [1].", 0) == "Do this."
