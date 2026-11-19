from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from . import guard, prompts
from .config import Settings
from .index import Hit, HybridRetriever
from .llm import LLM, LLMError
from .safety import (
    Assessment,
    Helplines,
    Level,
    assess,
    crisis_response,
    supportive_footer,
)
from .sessions import Turn

log = logging.getLogger(__name__)

Mode = Literal["crisis", "grounded", "empathy", "smalltalk", "out_of_scope", "extractive"]

_GREETING = re.compile(
    r"^\s*(?:hi+|hello+|hey+|namaste|good (?:morning|afternoon|evening))\b[\s!.,]*$", re.I
)
_THANKS = re.compile(r"^\s*(?:thanks?|thank you|thx|ty)\b[\s!.,a-z]{0,20}$", re.I)
_EMOTIONAL = re.compile(
    r"\b(?:feel|feeling|felt|stress\w*|anxi\w*|sad|depress\w*|lonely|alone|overwhelm\w*|tired|"
    r"exhaust\w*|worr\w*|scared|afraid|angry|upset|cry\w*|hopeless|lost|stuck|struggl\w*|panic\w*|"
    r"grie\w*|miss\w*|hurt\w*|numb|empty)\b",
    re.I,
)

GREETING_REPLY = (
    "Hi, I'm MindBridge. I can share practical ideas about stress, sleep, mood, work and "
    "relationships. What's on your mind?"
)
THANKS_REPLY = "You're welcome. Take care, and come back any time."
OUT_OF_SCOPE_REPLY = (
    "That's outside what I can help with. I'm set up for questions about stress, mood, sleep, "
    "work and relationships. If something is weighing on you, tell me about it and I'll do what "
    "I can."
)
EMPATHY_FALLBACK = (
    "That sounds hard, and I'm glad you said it out loud. I don't have specific material on this, "
    "but if it keeps weighing on you, a counsellor or Tele-MANAS (14416, free, 24x7) can help. "
    "What part of it is hardest right now?"
)


@dataclass(frozen=True)
class Source:
    doc_id: str
    title: str
    section: str
    snippet: str
    score: float


@dataclass
class Reply:
    text: str
    mode: Mode
    level: Level = Level.NONE
    about_other: bool = False
    sources: list[Source] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)


def _sources(hits: Sequence[Hit]) -> list[Source]:
    out: list[Source] = []
    for h in hits:
        snippet = " ".join(h.chunk.text.split())
        out.append(
            Source(
                h.chunk.doc_id,
                h.chunk.title,
                h.chunk.section,
                snippet[:180] + ("..." if len(snippet) > 180 else ""),
                round(h.score, 4),
            )
        )
    return out


def _extractive(hits: Sequence[Hit]) -> str:
    top = hits[0].chunk
    words = " ".join(top.text.split()).split()
    body = " ".join(words[:90]) + ("..." if len(words) > 90 else "")
    return (
        f"Here is something from the library that fits what you described ({top.title}) [1]:\n\n"
        f"{body}\n\nIf you tell me a bit more about what is going on, I can point you to "
        "something more specific."
    )


class Pipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLM,
        helplines: Helplines,
        settings: Settings,
    ):
        self.retriever = retriever
        self.llm = llm
        self.helplines = helplines
        self.cfg = settings

    # ---------------------------------------------------------------- public

    def answer(self, message: str, history: Sequence[Turn] = ()) -> Reply:
        t0 = time.perf_counter()
        timings: dict[str, float] = {}

        verdict = assess(message)
        if self.cfg.safety_llm_check and verdict.level == Level.NONE and verdict.soft_signals == 1:
            verdict = self._llm_second_opinion(message, verdict)
        timings["safety"] = _ms(t0)

        if verdict.level == Level.HIGH and not verdict.about_other:
            log.info("crisis path rules=%s", ",".join(verdict.rules))
            return Reply(
                crisis_response(self.helplines), "crisis", verdict.level, False, timings_ms=timings
            )

        if _GREETING.match(message):
            return Reply(GREETING_REPLY, "smalltalk", verdict.level, timings_ms=timings)
        if _THANKS.match(message):
            return Reply(THANKS_REPLY, "smalltalk", verdict.level, timings_ms=timings)

        t1 = time.perf_counter()
        hits = self.retriever.search(self._query(message, history), k=self.cfg.top_k)
        hits = self._with_required_docs(hits, verdict)
        hits = hits[: self.cfg.top_k + 1]
        timings["retrieval"] = _ms(t1)

        query = self._query(message, history)
        supported = verdict.about_other or self.retriever.is_supported(
            query, hits, self.cfg.min_coverage, self.cfg.min_dense_score
        )
        distressed = verdict.level >= Level.ELEVATED or bool(_EMOTIONAL.search(message))

        t2 = time.perf_counter()
        if supported:
            text, mode = self._grounded(message, history, hits, verdict)
            sources = _sources(hits)
        elif distressed:
            text, mode = self._empathy(message, history, verdict), "empathy"
            sources = []
        else:
            text, mode, sources = OUT_OF_SCOPE_REPLY, "out_of_scope", []
        timings["generation"] = _ms(t2)

        if verdict.level >= Level.ELEVATED:
            text += supportive_footer(self.helplines)
        timings["total"] = _ms(t0)
        return Reply(text, mode, verdict.level, verdict.about_other, sources, timings)

    # ---------------------------------------------------------------- steps

    @staticmethod
    def _query(message: str, history: Sequence[Turn]) -> str:
        """Short follow-ups ("what about at night?") are searched together with the previous
        user message so they keep their topic."""
        if len(message.split()) <= 6:
            for turn in reversed(history):
                if turn.role == "user":
                    return f"{turn.content} {message}"
        return message

    def _with_required_docs(self, hits: list[Hit], verdict: Assessment) -> list[Hit]:
        """Make sure the right guidance is in context when the message signals risk."""
        if verdict.about_other:
            first = self.retriever.chunks_for("supporting-a-friend")[:1]
            rest = [h for h in hits if h.chunk.doc_id != "supporting-a-friend"]
            return [Hit(c, 1.0, 1.0, 0.0) for c in first] + rest
        if verdict.level >= Level.ELEVATED:
            needed = "when-to-see-a-professional"
            if all(h.chunk.doc_id != needed for h in hits):
                extra = self.retriever.chunks_for(needed)[:1]
                hits = hits + [Hit(c, 0.0, 0.0, 0.0) for c in extra]
        return hits

    def _messages(self, message: str, history: Sequence[Turn]) -> list[dict[str, str]]:
        recent = list(history)[-2 * self.cfg.history_turns :]
        return [{"role": t.role, "content": t.content} for t in recent] + [
            {"role": "user", "content": message}
        ]

    def _grounded(
        self, message: str, history: Sequence[Turn], hits: list[Hit], verdict: Assessment
    ) -> tuple[str, Mode]:
        system = prompts.GROUNDED
        if verdict.about_other:
            system += prompts.OTHER_NOTE
        elif verdict.level >= Level.ELEVATED:
            system += prompts.ELEVATED_NOTE
        system += "\n\nContext\n\n" + prompts.format_context(hits)
        try:
            text = self.llm.generate(
                system,
                self._messages(message, history),
                max_tokens=self.cfg.llm_max_tokens,
                temperature=self.cfg.llm_temperature,
            )
        except LLMError as exc:
            log.warning("generation unavailable, answering extractively: %s", exc)
            return _extractive(hits), "extractive"

        problems = guard.check(text)
        if problems:
            log.warning("output guard tripped: %s", ",".join(problems))
            return _extractive(hits), "extractive"
        return guard.clean_citations(text, len(hits)), "grounded"

    def _empathy(self, message: str, history: Sequence[Turn], verdict: Assessment) -> str:
        system = prompts.NO_CONTEXT
        if verdict.level >= Level.ELEVATED:
            system += prompts.ELEVATED_NOTE
        try:
            text = self.llm.generate(
                system,
                self._messages(message, history),
                max_tokens=200,
                temperature=self.cfg.llm_temperature,
            )
        except LLMError:
            return EMPATHY_FALLBACK
        return EMPATHY_FALLBACK if guard.check(text) else text

    def _llm_second_opinion(self, message: str, verdict: Assessment) -> Assessment:
        """Borderline wording (a single hopelessness cue, no explicit rule) gets screened by
        the model. It can raise the level but never lower it, and any failure is ignored."""
        try:
            raw = self.llm.generate(
                prompts.CLASSIFIER,
                [{"role": "user", "content": message}],
                max_tokens=20,
                temperature=0.0,
            )
            risk = json.loads(raw[raw.index("{") : raw.rindex("}") + 1]).get("risk", "none")
        except (LLMError, ValueError, AttributeError):
            return verdict
        level = {"high": Level.HIGH, "elevated": Level.ELEVATED}.get(risk)
        if level is None:
            return verdict
        return Assessment(
            level, verdict.about_other, (*verdict.rules, "llm-check"), verdict.soft_signals
        )


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 1)
