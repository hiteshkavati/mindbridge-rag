import pytest

from mindbridge import evaluate as ev
from mindbridge.safety import Level, assess, crisis_response, load_helplines, supportive_footer

from .conftest import ROOT


@pytest.mark.parametrize(
    "text",
    [
        "I want to kill myself",
        "I'm going to end my life tonight",
        "I've decided to take my own life",
        "I wrote a goodbye note",
        "I've been cutting myself again and can't stop",
        "suicide karne ka soch raha hoon",
    ],
)
def test_explicit_intent_is_high(text):
    assert assess(text).level == Level.HIGH


@pytest.mark.parametrize(
    "text",
    [
        "I don't want to be here anymore",
        "I wish I could just disappear",
        "I feel like a burden to everyone",
        "what's the point of living anymore",
        "I feel hopeless and worthless",
    ],
)
def test_passive_ideation_is_elevated(text):
    assert assess(text).level == Level.ELEVATED


@pytest.mark.parametrize(
    "text",
    [
        "this deadline is killing me",
        "I'm dead tired",
        "I could die of embarrassment",
        "I'm dying to see that film",
        "my boss will kill me if I'm late",
        "I killed it in my presentation",
        "I hurt myself at the gym",
        "I want to end this relationship",
        "I don't want to live in this hostel anymore",
        "I'm giving up on my diet",
        "",
    ],
)
def test_common_idioms_and_everyday_text_are_not_flagged(text):
    assert assess(text).level == Level.NONE


def test_negation_suppresses_a_match():
    assert assess("I would never hurt myself").level == Level.NONE
    assert assess("I don't want to die, I'm just scared of flying").level == Level.NONE


def test_negation_followed_by_contrast_is_not_suppressed():
    assert (
        assess("I don't want to hurt myself but I keep thinking about it").level >= Level.ELEVATED
    )


def test_past_tense_downgrades_high_to_elevated():
    result = assess("Years ago I wanted to kill myself, but that was a long time back")
    assert result.level == Level.ELEVATED


def test_third_party_risk_is_flagged_and_marked():
    result = assess("my brother told me he wants to kill himself")
    assert result.level == Level.ELEVATED
    assert result.about_other


def test_first_person_wins_over_third_party_in_the_same_message():
    result = assess("my friend wants to end her life. I want to kill myself too")
    assert result.level == Level.HIGH
    assert not result.about_other


def test_matching_is_case_and_unicode_insensitive():
    assert assess("I WANT TO KILL MYSELF").level == Level.HIGH
    assert assess("I don’t want to be here anymore").level == Level.ELEVATED


def test_soft_signals_need_to_cluster():
    assert assess("I feel hopeless about this bug").level == Level.NONE
    assert assess("I feel hopeless and worthless").level == Level.ELEVATED


def test_crisis_response_contains_numbers_and_no_method_language():
    h = load_helplines(ROOT / "data" / "helplines.json")
    text = crisis_response(h)
    assert "112" in text and "14416" in text
    assert "Are you safe" in text
    assert supportive_footer(h).count("-") >= 2


def test_helplines_fall_back_to_built_in_numbers(tmp_path):
    h = load_helplines(tmp_path / "missing.json")
    assert h.emergency_number == "112"
    assert any("14416" in n for line in h.lines for n in line.numbers)
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert load_helplines(broken).lines


def test_labelled_set_has_no_missed_high_risk_cases():
    """Regression guard over the eval file: never miss an explicit-intent message."""
    rows = ev.load_jsonl(ROOT / "eval" / "safety.jsonl")
    for split in ("dev", "test"):
        metrics = ev.safety_metrics([r for r in rows if r["split"] == split])
        assert metrics["recall_high"] == 1.0, metrics["misses"]
        assert metrics["false_flag_rate"] <= 0.1, metrics["misses"]
