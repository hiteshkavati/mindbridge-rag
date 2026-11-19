"""Crisis-language detection.

Rule based on purpose: it is fast, has no network dependency, and every decision
can be traced to a named rule. The cost is recall on unusual phrasing, which is
why the pipeline can optionally ask an LLM for a second opinion on borderline
messages (see Pipeline._llm_second_opinion) and why helplines are shown for the
ELEVATED level as well.

Levels
  HIGH      explicit intent, plan or active self-harm. The pipeline skips the LLM
            entirely and returns a fixed response.
  ELEVATED  passive ideation, hopelessness, or concern about another person.
            Generation is allowed, helplines are always appended.
  NONE      everything else, including common idioms ("this deadline is killing me").
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path


class Level(IntEnum):
    NONE = 0
    ELEVATED = 1
    HIGH = 2


@dataclass(frozen=True)
class Assessment:
    level: Level
    about_other: bool = False
    rules: tuple[str, ...] = ()
    soft_signals: int = 0


@dataclass(frozen=True)
class Rule:
    name: str
    rx: re.Pattern[str]
    level: Level
    neg_ok: bool = False  # the pattern already encodes its own negation
    cue: re.Pattern[str] | None = None  # must also appear somewhere in the sentence


def _r(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.I)


H, E = Level.HIGH, Level.ELEVATED

_SELF_HARM_CUE = _r(
    r"\b(?:want(?:ed)?|wanna|going|gonna|thinking|think|thought|urges?|plan(?:ning)?|again|"
    r"can'?t stop|keep|feel like|tempted|need to|trying|tried|tonight|right now|ways?)\b"
)
_FIRST_PERSON = _r(r"\b(?:i|im|ive|ill|i'm|i've|i'll|my)\b")

_RELATIONS = (
    "he|she|they|friend|brother|sister|mom|mother|dad|father|cousin|classmate|colleague|"
    "roommate|boyfriend|girlfriend|partner|son|daughter|student|teammate|someone"
)

FIRST_PERSON_RULES: tuple[Rule, ...] = (
    Rule(
        "kill-self",
        _r(r"\bkill(?:ing)? myself\b(?! (?:laughing|working|studying|trying|doing))"),
        H,
    ),
    Rule("end-life", _r(r"\bend(?:ing)? (?:my (?:own )?life|it all)\b"), H),
    Rule("take-life", _r(r"\btak(?:e|ing) my (?:own )?life\b"), H),
    Rule(
        "suicide-act",
        _r(r"\b(?:commit(?:ting)?|attempt(?:ing)?) suicide\b"),
        H,
        cue=_FIRST_PERSON,
    ),
    Rule(
        "want-die",
        _r(
            r"\b(?:want|wanna|ready|decided|plan(?:ning)?)(?: \w+){0,2} (?:to )?die\b"
            r"(?! (?:of|from|laughing|inside|for|if|when|before|without))"
        ),
        H,
    ),
    Rule(
        "alive-tomorrow",
        _r(
            r"\b(?:won'?t|will not|not) (?:be )?alive (?:tomorrow|tonight|by|after|much longer)\b"
            r"|\bdon'?t think i(?:'ll| will) (?:be )?(?:alive|here|make it) "
            r"(?:tomorrow|tonight|much longer)\b"
        ),
        H,
        neg_ok=True,
    ),
    Rule(
        "plan-act",
        _r(
            r"\bplan(?:ning|ned)? to (?:end (?:it all|my life|things for good)|kill myself|die|"
            r"hurt myself|harm myself)\b"
        ),
        H,
    ),
    Rule(
        "note",
        _r(r"\b(?:wrote|writing|written|leaving|left|made)(?: \w+){0,3} (?:goodbye|suicide|farewell) (?:note|letter|message)\b"),
        H,
    ),
    Rule(
        "overdose",
        _r(
            r"\b(?:took|taken|swallowed) (?:too many|a lot of|all (?:my|the|of the)) "
            r"(?:pills|tablets|medication|meds)\b[^.!?]{0,30}\b(?:on purpose|deliberately|to die|to end)\b"
            r"|\boverdos(?:ed|ing|e) on purpose\b"
        ),
        H,
    ),
    Rule(
        "self-harm",
        _r(r"\b(?:hurt(?:ing)?|harm(?:ing)?|cutting|burning|injur(?:e|ing)) myself\b"),
        H,
        cue=_SELF_HARM_CUE,
    ),
    Rule(
        "self-harm-habit",
        _r(r"\bi(?:'ve| have|'m| am)? (?:been |started |started to )?self[- ]?harm(?:ing)?\b"),
        H,
    ),
    Rule(
        "hi-act",
        _r(
            r"\bsuicide kar(?:ne| ?loonga| ?lunga| ?lungi| ?na chah(?:ta|ti))\b|\bja(?:a)?n dene\b|"
            r"\bkhud ko khatam\b|\bmar (?:jana|jaana|na) chah(?:ta|ti)\b|\bzindagi khatam kar"
        ),
        H,
    ),
    Rule(
        "passive-here",
        _r(
            r"\b(?:don'?t|do not|no longer) want to (?:be here|be alive|exist|wake up|go on|live)\b"
            r"(?! (?:in|with|near|at|there|alone|on|here at))"
        ),
        E,
        neg_ok=True,
    ),
    Rule(
        "wish-gone",
        _r(
            r"\bwish (?:i|that i)(?: was| were| could be)? (?:dead|gone|never born|not alive|not here)\b"
            r"|\bwish i (?:could )?(?:just )?(?:disappear|vanish|die|not exist|didn'?t exist)\b"
        ),
        E,
    ),
    Rule("never-wake", _r(r"\b(?:never|not) (?:to )?wake up\b(?! (?:late|early|on time))"), E, neg_ok=True),
    Rule(
        "no-point",
        _r(
            r"\b(?:no|not any) (?:point|reason|purpose) (?:in |to |of )?(?:living|live|life|going on|continuing)\b"
            r"|\bpoint of (?:living|life|going on)\b|\blife (?:isn'?t|is not|is no longer) worth\b"
            r"|\b(?:isn'?t|not) worth living\b"
        ),
        E,
        neg_ok=True,
    ),
    Rule(
        "burden",
        _r(r"\bburden (?:to|on) (?:everyone|everybody|my family|my parents|others|them|all|people)\b"),
        E,
    ),
    Rule(
        "better-off",
        _r(
            r"\bbetter off without me\b|\bbetter off if i (?:was|were|wasn'?t|weren'?t) "
            r"(?:gone|dead|here|around|alive|born)\b|\b(?:no ?one|nobody) would (?:even )?"
            r"(?:notice|miss)(?: me)? if i (?:was|were|am|wasn'?t|disappeared|left|died)\b"
        ),
        E,
        neg_ok=True,
    ),
    Rule(
        "cant-go-on",
        _r(
            r"\bcan'?t (?:go on|keep going)\b|\bcannot go on\b|"
            r"\bwant (?:it|this|everything|the pain) to stop\b|\bgiv(?:e|ing) up on (?:life|everything)\b"
        ),
        E,
    ),
    Rule(
        "dark-thoughts",
        _r(r"\bdark thoughts\b|\bthoughts of (?:dying|death|suicide|ending (?:it|my life))\b"),
        E,
    ),
    Rule(
        "suicidal-self",
        _r(
            r"\bi(?:'m| am|'ve been| have been| feel| am feeling|'m feeling)? (?:feeling )?suicidal\b"
            r"|\bthinking (?:about|of) suicide\b|\bsuicidal (?:thoughts|feelings|ideation)\b"
        ),
        E,
    ),
    Rule("hi-passive", _r(r"\bjeene ka (?:mann|man|mood) nahi\b|\bjeena nahi chah"), E),
    Rule("slang", _r(r"\b(?:kms|unalive (?:myself|me))\b"), E),
)

THIRD_PARTY_RULES: tuple[Rule, ...] = (
    Rule(
        "other-self-harm",
        _r(r"\b(?:kill|hurt|harm|end)(?:ing)? (?:himself|herself|themselves|themself|(?:his|her|their) (?:own )?life)\b"),
        E,
    ),
    Rule("other-take-life", _r(r"\btak(?:e|ing) (?:his|her|their) (?:own )?life\b"), E),
    Rule("other-suicidal", _r(rf"\b(?:{_RELATIONS})\b[^.!?]{{0,40}}\bsuicidal\b"), E),
    Rule("other-wants-die", _r(r"\bwants to die\b"), E),
)

_SOFT = {
    "hopeless": _r(r"\bhopeless(?:ness)?\b"),
    "worthless": _r(r"\bworthless\b"),
    "trapped": _r(r"\btrapped\b"),
    "no-way-out": _r(r"\bno way out\b"),
    "nothing-matters": _r(r"\bnothing (?:even )?matters\b"),
    "cant-take": _r(r"\bcan'?t (?:take|do) (?:it|this) anymore\b"),
    "giving-up": _r(r"\bgiv(?:e|ing) up\b"),
    "empty": _r(r"\bempty inside\b"),
    "tired-of-all": _r(r"\btired of (?:everything|life|living|it all)\b"),
    "burden-word": _r(r"\bburden\b"),
}

_NEGATION = _r(
    r"(?:\bnot\b|\bdon'?t\b|\bdoesn'?t\b|\bwon'?t\b|\bwouldn'?t\b|\bnever\b|\bno longer\b|\bwithout\b)"
    r"[^.!?]{0,40}$"
)
_CONTRAST = _r(r"\b(?:but|though|however|yet|still)\b")
_PAST = _r(
    r"\b(?:used to|years ago|months ago|back in|when i was|in (?:school|college|high school|"
    r"the past)|last year|as a (?:kid|teen|teenager|child))\b"
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+|;")


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("’", "'").replace("`", "'")
    return re.sub(r"[ \t]+", " ", text)


def _negated(sentence: str, start: int) -> bool:
    return bool(_NEGATION.search(sentence[:start]))


def _assess_sentence(sent: str) -> tuple[Level, bool, list[str]]:
    level, other, names = Level.NONE, False, []

    def bump(new_level: Level, new_other: bool, name: str) -> None:
        nonlocal level, other
        names.append(name)
        if new_level > level or (new_level == level and other and not new_other):
            level, other = new_level, new_other

    for rule in THIRD_PARTY_RULES:
        if rule.rx.search(sent):
            bump(rule.level, True, rule.name)

    for rule in FIRST_PERSON_RULES:
        match = rule.rx.search(sent)
        if not match or (rule.cue and not rule.cue.search(sent)):
            continue
        lvl = rule.level
        if not rule.neg_ok and _negated(sent, match.start()):
            if _CONTRAST.search(sent[match.end():]):
                lvl = Level.ELEVATED  # "I don't want to hurt myself but ..."
            else:
                continue
        if lvl == Level.HIGH and _PAST.search(sent):
            lvl = Level.ELEVATED
        bump(lvl, False, rule.name)
    return level, other, names


def assess(text: str) -> Assessment:
    text = normalise(text)
    level, other, rules = Level.NONE, False, []
    for sent in filter(None, (s.strip() for s in _SENTENCE.split(text))):
        lvl, oth, names = _assess_sentence(sent)
        rules.extend(names)
        if lvl > level or (lvl == level and other and not oth):
            level, other = lvl, oth

    soft = [name for name, rx in _SOFT.items() if rx.search(text)]
    if level < Level.ELEVATED and len(soft) >= 2:
        level, other = Level.ELEVATED, False
        rules.append("soft-cluster")
    return Assessment(level, other, tuple(rules), len(soft))


# ---------------------------------------------------------------- helplines


@dataclass(frozen=True)
class Helpline:
    name: str
    numbers: tuple[str, ...]
    hours: str


@dataclass(frozen=True)
class Helplines:
    emergency_number: str
    lines: tuple[Helpline, ...]

    def block(self) -> str:
        rows = [f"- {h.name}: {' or '.join(h.numbers)} ({h.hours})" for h in self.lines]
        return f"Emergency services: {self.emergency_number}\n" + "\n".join(rows)


# The crisis path must never depend on a file being present, so the same
# numbers are compiled in. data/helplines.json overrides them when it loads.
_FALLBACK = Helplines(
    "112",
    (
        Helpline("Tele-MANAS", ("14416", "1-800-891-4416"), "24x7, free, many languages"),
        Helpline("iCall (TISS)", ("9152987821",), "Mon-Sat, 10 AM to 8 PM"),
    ),
)


def load_helplines(path: Path | str | None) -> Helplines:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))  # type: ignore[arg-type]
        lines = tuple(
            Helpline(x["name"], tuple(x["numbers"]), x["hours"]) for x in data["lines"]
        )
        return Helplines(data["emergency"]["number"], lines)
    except (OSError, ValueError, KeyError, TypeError):
        return _FALLBACK


def crisis_response(helplines: Helplines) -> str:
    return (
        "I'm really glad you told me. What you're describing sounds heavy, and your safety "
        "matters most right now.\n\n"
        f"If you are in immediate danger, or you have already hurt yourself or taken something, "
        f"please call {helplines.emergency_number} now or go to the nearest emergency room.\n\n"
        "You don't have to be sure about anything to reach out. These lines are free to call "
        "and are staffed by trained people:\n"
        + "\n".join(
            f"- {h.name}: {' or '.join(h.numbers)} ({h.hours})" for h in helplines.lines
        )
        + "\n\nIf you can, tell someone near you that you are not okay and ask them to stay with "
        "you for now. If there is anything within reach that you could use to hurt yourself, "
        "try to put some distance between you and it.\n\n"
        "I'm here and happy to keep talking. Are you safe right now?"
    )


def supportive_footer(helplines: Helplines) -> str:
    return (
        "\n\nIf things ever feel like too much, you can talk to someone at any time:\n"
        + "\n".join(f"- {h.name}: {' or '.join(h.numbers)} ({h.hours})" for h in helplines.lines)
        + f"\nIn an emergency, call {helplines.emergency_number}."
    )
