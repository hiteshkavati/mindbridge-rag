# Safety design

This is a self-help assistant. It is not therapy and not a crisis service. The safety layer
exists so that a person in distress is pointed to people who can help, quickly and without
depending on a language model behaving well.

## Levels

- `HIGH`: explicit intent, plan or act. The pipeline skips retrieval and the LLM and returns a
  fixed message with Indian helplines (Tele-MANAS 14416 / 1-800-891-4416, iCall 9152987821,
  emergency 112) and asks whether the person is safe right now.
- `ELEVATED`: passive ideation, hopelessness, burden language, or a worry about someone else.
  The normal answer is produced with a gentler system note, `when-to-see-a-professional` is
  added to the context, and a helpline footer is appended.
- `NONE`: everything else.

## What the rules handle

Negation ("I'm not going to kill myself"), contrast ("I felt like ending it but I'm okay now"),
past tense, third-party risk ("my friend wants to die", routed to `supporting-a-friend`), common
idioms ("this traffic is killing me"), Hinglish phrasings, and clusters of weaker signals
(two or more soft cues together escalate to `ELEVATED`).

## Limits

- Rules are fitted to phrasings the author thought of. Real messages are more varied, so
  recall on unseen text will be lower than the eval report suggests. See the caveats there.
- English and a small Hinglish set only.
- A negation or past-tense downgrade can be wrong in either direction.
- An optional LLM second opinion (`MB_SAFETY_LLM_CHECK`) runs only when no rule fired and exactly
  one weak signal was present. It can raise the level and never lowers one. It is off by default.

## Helplines

`data/helplines.json` carries a `last_verified` date. Numbers and hours change, so check them
against the official sources before any real deployment and update the file. A compiled-in
fallback is used if the file is missing.

## Knowledge base

The documents are plain-language summaries of widely taught self-help techniques. They have
not been reviewed by a clinician. That review is required before anyone relies on this.
