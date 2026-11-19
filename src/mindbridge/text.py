from __future__ import annotations

import re

_WORD = re.compile(r"[a-z][a-z0-9]*")

STOPWORDS = frozenset(
    """a an and are as at be been but by can for from had has have how i if in into is it its
    me my of on or so than that the their them then there these they this to up was we were
    what when where which who why will with would you your do does did doing make makes get gets
    got just like im ive id dont cant wont very really also about some any more most much many
    too ill""".split()
)

_SUFFIXES = (
    ("ies", "y"),
    ("sses", "ss"),
    ("ing", ""),
    ("edly", ""),
    ("ed", ""),
    ("ly", ""),
    ("s", ""),
)


def stem(word: str) -> str:
    """Very light suffix stripping. Good enough to match stress/stressed/stressing
    without pulling in a full stemmer dependency."""
    if len(word) <= 4:
        return word
    for suffix, repl in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            if suffix == "s" and word.endswith("ss"):
                continue
            return word[: -len(suffix)] + repl
    return word


def tokenize(text: str) -> list[str]:
    text = text.lower().replace("’", "'")
    text = text.replace("'", "")
    return [stem(w) for w in _WORD.findall(text) if w not in STOPWORDS]
