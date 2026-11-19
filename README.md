# mindbridge-rag

A retrieval-grounded self-help chat assistant for everyday stress, sleep, burnout, exam and
relationship problems, with a rule-based crisis layer and India-specific helplines. It
answers from a small authored knowledge base, shows its sources, and says so when a question
is outside what it covers.

This is a rebuild of an earlier college project, a mental health chatbot trained on a Q&A CSV.
That version had three problems I wanted to fix properly: the dataset collapsed to 53 unique
question-answer pairs once duplicates were removed, passwords were stored in plain text, and
there was no handling at all for someone writing about harming themselves.

This is a self-help tool. It is not therapy and not a crisis service, and the knowledge base
has not been reviewed by a clinician.

## What it does

- **Hybrid retrieval.** A from-scratch Okapi BM25 and a dense retriever, merged with
  Reciprocal Rank Fusion, with a per-document cap so one article cannot fill the context.
- **Out-of-scope gate.** IDF-weighted lexical coverage decides whether the knowledge base
  can answer. If not, the assistant says that instead of guessing.
- **Crisis layer.** Rules for explicit intent, passive ideation and third-party worry,
  handling negation, past tense, idioms and Hinglish. High risk bypasses the LLM and returns
  a fixed reply with Tele-MANAS, iCall and 112.
- **Output guard.** Generated text that names medication, doses, diagnoses or guarantees is
  replaced with an extractive answer built from the retrieved text. Invalid citations are removed.
- **Works without an LLM.** With no provider configured, answers are extractive. Anthropic,
  OpenAI-compatible and Ollama backends are supported over plain httpx.
- **Service.** FastAPI, scrypt password hashing, JWT, per-user and per-IP rate limits,
  in-memory sessions bound to their owner, strict CSP, and a small vanilla JS chat UI.

## Quickstart

```bash
pip install -e ".[dev]"
export MB_JWT_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
mindbridge serve            # http://127.0.0.1:8000
```

Terminal chat without auth: `mindbridge chat`. Check the knowledge base: `mindbridge kb-check`.

To generate with a model, set `MB_LLM_PROVIDER` to `anthropic`, `openai` or `ollama` and the
matching key. For better embeddings, `pip install -e ".[sbert]"` and set
`MB_EMBEDDING_BACKEND=sbert`. `auto` tries sentence-transformers and falls back to the offline
LSA embedder. See `.env.example`. Docker: `docker compose up --build`.

## API

```bash
curl -X POST localhost:8000/api/auth/register -H 'content-type: application/json' \
  -d '{"username":"asha","password":"a-long-passphrase"}'

curl -X POST localhost:8000/api/auth/login -H 'content-type: application/json' \
  -d '{"username":"asha","password":"a-long-passphrase"}'
# -> {"access_token": "...", ...}   (export it as TOKEN)

curl -X POST localhost:8000/api/chat -H "authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"message":"I freeze during exams even when I have studied"}'
```

The response contains `reply`, `mode` (`grounded`, `extractive`, `crisis`, ...), `safety`,
`sources` with titles and snippets, `session_id` for follow-ups, and per-stage `timings_ms`.
Other routes: `/api/health`, `/api/helplines`, `/api/auth/login`, `DELETE /api/sessions/{id}`.

## Measured results

Offline LSA embedder, 134 chunks from 30 documents. Full tables in
[docs/eval-report.md](docs/eval-report.md), regenerate with `make eval`.

| holdout (30 in-scope, 8 out-of-scope) | hit@1 | hit@3 | hit@5 | MRR@10 |
|---|---|---|---|---|
| BM25 | 47% | 73% | 77% | 0.62 |
| dense (LSA) | 57% | 77% | 83% | 0.69 |
| hybrid | 53% | 80% | 83% | 0.68 |

At the shipped coverage cutoff the holdout set answers 83% of in-scope queries and declines
88% of out-of-scope ones.

Read these with care:

- The holdout was written after the system was tuned. Dev and test reach 93-100% retrieval
  only because I added keyword metadata to the documents while looking at those queries.
- Hybrid is not clearly better than the best single retriever with LSA. I have not run the
  sentence-transformers path, so there are no numbers for it.
- Crisis detection scored 100% on my dev and test cases, which I wrote knowing the rules. A
  fresh holdout found real gaps (recall of 70% high and 44% elevated), which I fixed, so it is
  no longer a clean measurement. Expect lower recall on real messages.
- Generation quality is not measured automatically.

## Layout

```
src/mindbridge/   retrieval, safety, guard, llm, pipeline, api, auth, cli, static UI
data/kb/          30 knowledge base documents
data/helplines.json
eval/             labelled retrieval and crisis sets (dev, test, holdout)
docs/             architecture, safety design, design decisions, eval report
tests/            pytest suite
```

## Privacy

Conversations are held in memory with a time limit and are tied to the user who started them.
Only usernames and password hashes are stored on disk. Set `MB_ENV=prod` and a long
`MB_JWT_SECRET` before exposing the service; it refuses to start in prod with a weak secret.
If you enable an LLM provider, messages are sent to that provider.

## Before real use

Have a clinician review the knowledge base, check `data/helplines.json` against official
sources, and test the crisis layer against real-world phrasing. If you or someone near you is
in danger in India, call 112, or reach Tele-MANAS on 14416 / 1-800-891-4416.

Earlier version: [Chatbot-for-Mental-Health-Support](https://github.com/hiteshkavati/Chatbot-for-Mental-Health-Support).

## License

MIT
