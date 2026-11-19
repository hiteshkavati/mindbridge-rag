from mindbridge.llm import LLMError, LLMUnavailable
from mindbridge.pipeline import Pipeline
from mindbridge.safety import Level
from mindbridge.sessions import Turn


def test_crisis_message_never_reaches_the_model(pipeline, fake_llm):
    reply = pipeline.answer("I want to kill myself")
    assert reply.mode == "crisis"
    assert reply.level == Level.HIGH
    assert fake_llm.calls == []
    assert "112" in reply.text and "14416" in reply.text
    assert reply.sources == []


def test_grounded_answer_carries_sources_and_valid_citations(pipeline, fake_llm):
    fake_llm.reply = "Write the worries down earlier in the evening [1]. Does that feel doable? [9]"
    reply = pipeline.answer("my mind races at night and I can't sleep")
    assert reply.mode == "grounded"
    assert reply.sources
    assert "[9]" not in reply.text
    assert "[1]" in reply.text
    system = fake_llm.calls[0]["system"]
    assert "Context" in system and "[1]" in system


def test_model_failure_falls_back_to_extractive_answer(pipeline, fake_llm):
    fake_llm.error = LLMUnavailable("down")
    reply = pipeline.answer("my mind races at night and I can't sleep")
    assert reply.mode == "extractive"
    assert reply.sources
    assert "library" in reply.text


def test_unexpected_provider_error_also_falls_back(pipeline, fake_llm):
    fake_llm.error = LLMError("bad request")
    assert pipeline.answer("how do I handle exam stress").mode == "extractive"


def test_output_guard_replaces_unsafe_generation(pipeline, fake_llm):
    fake_llm.reply = "You probably have depression. Take 50 mg of sertraline."
    reply = pipeline.answer("I have felt low and tired for weeks and lost interest in things")
    assert reply.mode == "extractive"
    assert "sertraline" not in reply.text


def test_out_of_scope_question_is_declined_without_calling_the_model(pipeline, fake_llm):
    reply = pipeline.answer("who won the cricket match yesterday")
    assert reply.mode == "out_of_scope"
    assert fake_llm.calls == []


def test_distress_without_matching_material_gets_an_empathic_reply(pipeline, fake_llm):
    fake_llm.reply = "That sounds really hard. What part is weighing on you most?"
    reply = pipeline.answer("my cat ran away and I feel awful")
    assert reply.mode == "empathy"
    assert reply.sources == []
    assert "Context" not in fake_llm.calls[0]["system"]


def test_empathy_fallback_when_model_is_down(pipeline, fake_llm):
    fake_llm.error = LLMUnavailable("down")
    reply = pipeline.answer("my cat ran away and I feel awful")
    assert reply.mode == "empathy"
    assert "Tele-MANAS" in reply.text


def test_elevated_risk_still_generates_but_always_adds_helplines(pipeline, fake_llm):
    reply = pipeline.answer("I feel hopeless and worthless and I'm exhausted by work stress")
    assert reply.level == Level.ELEVATED
    assert "Tele-MANAS" in reply.text and "112" in reply.text
    assert "gentle" in fake_llm.calls[0]["system"] or "distress" in fake_llm.calls[0]["system"]
    assert any(s.doc_id == "when-to-see-a-professional" for s in reply.sources)


def test_message_about_someone_else_uses_supporting_a_friend_guidance(pipeline, fake_llm):
    reply = pipeline.answer("my friend told me he wants to end his life, what should I do")
    assert reply.about_other
    assert reply.sources[0].doc_id == "supporting-a-friend"
    assert "someone else" in fake_llm.calls[0]["system"]
    assert "Tele-MANAS" in reply.text


def test_greeting_and_thanks_are_handled_locally(pipeline, fake_llm):
    assert pipeline.answer("hi").mode == "smalltalk"
    assert pipeline.answer("thanks!").mode == "smalltalk"
    assert fake_llm.calls == []


def test_short_follow_up_keeps_the_previous_topic(pipeline):
    history = [Turn("user", "I can't sleep at night"), Turn("assistant", "Try a routine.")]
    assert "sleep" in pipeline._query("what about caffeine?", history)
    assert pipeline._query("what about caffeine?", []) == "what about caffeine?"
    long_msg = "this is a much longer message about something entirely different today"
    assert pipeline._query(long_msg, history) == long_msg


def test_history_is_trimmed_to_configured_turns(pipeline, fake_llm, settings):
    history = [Turn("user" if i % 2 == 0 else "assistant", f"m{i}") for i in range(40)]
    pipeline.answer("how do I deal with exam stress", history)
    sent = fake_llm.calls[0]["messages"]
    assert len(sent) <= 2 * settings.history_turns + 1
    assert sent[-1]["content"] == "how do I deal with exam stress"


def test_llm_second_opinion_can_raise_but_not_lower(retriever, helplines, settings, fake_llm):
    cfg = settings.model_copy(update={"safety_llm_check": True})
    pipe = Pipeline(retriever, fake_llm, helplines, cfg)
    fake_llm.reply = '{"risk": "elevated"}'
    reply = pipe.answer("I feel hopeless about everything lately")
    assert reply.level == Level.ELEVATED

    fake_llm.reply = '{"risk": "none"}'
    assert pipe.answer("I feel hopeless about everything lately").level == Level.NONE

    fake_llm.reply = "not json at all"
    assert pipe.answer("I feel hopeless about everything lately").level == Level.NONE


def test_second_opinion_is_skipped_by_default(pipeline, fake_llm):
    pipeline.answer("I feel hopeless about this bug")
    assert all("screen messages" not in c["system"] for c in fake_llm.calls)
