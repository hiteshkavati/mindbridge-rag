# Design decisions

**Hybrid retrieval with RRF.** Queries are short and informal. BM25 catches exact words,
dense retrieval catches paraphrase. RRF needs no score calibration between the two. On the
offline LSA embedder, hybrid is roughly level with the better single retriever (see
`eval-report.md`); the fusion is kept because the dense side is expected to matter more with
sentence-transformers, which I could not run in the environment this was built in.

**Own BM25.** Small enough to write and test directly, and it keeps the offline path free of
extra dependencies.

**Coverage gate instead of a dense-score cutoff.** LSA cosine scores do not separate in-scope
from out-of-scope queries. IDF-weighted coverage of query terms by the retrieved chunks does a
better job. The cutoff (0.51) was chosen before keyword metadata was added to the documents and
is not re-fitted, because re-fitting rewards that metadata. The gate trades two errors: too
low and off-topic questions get a grounded-looking answer, too high and real questions are
declined. The holdout numbers show both errors exist.

**Rule-based safety with a fixed reply.** A crisis response should not depend on a model's
mood, latency or an API outage. Rules are auditable and testable. The cost is recall on
unusual phrasing; see `safety.md`.

**Output guard and extractive fallback.** If the generated text names a medication, gives a
dose, changes treatment, offers a diagnosis or promises an outcome, the answer is replaced by
an extractive one built from the retrieved text. The same path serves when no LLM is
configured, so the system is usable offline.

**Privacy defaults.** Conversations live in memory with a TTL and are bound to the user who
created them. Nothing is written to disk except hashed credentials. Sessions can be deleted
through the API.

**Generation quality is checked by hand.** Groundedness and tone need a live model and human
reading, so they are not in the automated report.
