"""Checks applied to model output before it reaches the user."""

from __future__ import annotations

import re

_DRUGS = (
    "sertraline|fluoxetine|escitalopram|citalopram|paroxetine|venlafaxine|bupropion|"
    "amitriptyline|alprazolam|clonazepam|diazepam|lorazepam|zolpidem|propranolol|lithium|"
    "quetiapine|olanzapine|risperidone|melatonin|ssri|benzodiazepine"
)

_CHECKS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("medication", re.compile(rf"\b(?:{_DRUGS})s?\b", re.I)),
    ("dosage", re.compile(r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|ml|tablets?|pills?)\b", re.I)),
    (
        "treatment-change",
        re.compile(
            r"\b(?:stop|skip|change|increase|reduce|double) (?:taking )?(?:your )?"
            r"(?:medication|meds|medicine|dose|dosage)\b",
            re.I,
        ),
    ),
    (
        "diagnosis",
        re.compile(
            r"\byou (?:probably |likely |definitely |clearly )?"
            r"(?:have|are suffering from|suffer from) (?:clinical |major )?"
            r"(?:depression|depressive disorder|an anxiety disorder|bipolar|ptsd|adhd|ocd|"
            r"panic disorder|schizophrenia)\b"
            r"|\b(?:this|that) is (?:clearly |definitely |likely )?"
            r"(?:depression|an anxiety disorder|bipolar|ptsd|adhd|ocd)\b"
            r"|\bi(?:'ve| have) diagnosed\b",
            re.I,
        ),
    ),
    (
        "guarantee",
        re.compile(r"\b(?:guarantee[sd]?|will definitely (?:cure|fix)|cure[sd]? your)\b", re.I),
    ),
)


def check(text: str) -> list[str]:
    """Return the names of every rule the text trips; empty means it passed."""
    return [name for name, rx in _CHECKS if rx.search(text)]


def clean_citations(text: str, n_sources: int) -> str:
    """Drop [n] markers that point at passages that were not provided."""

    def keep(m: re.Match[str]) -> str:
        return m.group(0) if 1 <= int(m.group(1)) <= n_sources else ""

    out = re.sub(r"\[(\d+)\]", keep, text)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return re.sub(r"[ \t]+([.,;!?])", r"\1", out).strip()
