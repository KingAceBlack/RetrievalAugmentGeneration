# enrag — step 1: an English corpus the model has never seen

A large, recent, public-domain English corpus from the US Federal Register.

## Why this source

You asked for documents unlikely to be in training data. **Recency is the only
guarantee that actually holds.** Obscurity doesn't — obscure text still gets
scraped. But a document published last week cannot be in a model trained
before it existed.

The Federal Register publishes every federal business day, roughly 200–250
documents daily. Filter by date and you have a corpus that is definitionally
unseen, and one that keeps growing for free.

It also happens to be the right *shape* for this test:

- **Public domain.** US government work. Redistribute it, show it to
  prospects, put it in a demo. No licensing conversation.
- **Structurally like enterprise documents.** Dry, procedural, sectioned,
  with SUMMARY / DATES / ADDRESSES / SUPPLEMENTARY INFORMATION headings. It
  reads like an internal policy estate, not like prose.
- **Dense with checkable specifics.** Docket numbers, CFR citations, comment
  deadlines, effective dates, named contacts, page ranges. These make good
  evaluation questions precisely because no model can guess them — the answer
  is either retrieved or it is wrong.
- **No API key.** No registration, no quota negotiation.

## Run it

```bash
pip install -r requirements.txt

python fetch_fr.py --probe                 # inspect the API's field names
python fetch_fr.py                         # last 30 days, all types
python fetch_fr.py --start 2026-06-01 --end 2026-08-01
python fetch_fr.py --types RULE PRORULE    # denser, more substantial documents
python fetch_fr.py --resume                # continue an interrupted run
python fetch_fr.py --stats                 # summarise what you got
```

Default range is the last 30 days: about 21 business days and 4,000-5,000
documents, several million words. Enough that retrieval quality actually
matters - a corpus of 50 documents makes any retriever look good.

**Expect a 20-40 minute run.** The listing endpoint ignores `fields[]` and
returns a fixed ten keys, none of them the full-text link, so each document
costs two requests: one to the single-document endpoint, one for the text.
That is not wasted effort - the detail endpoint is where the useful metadata
lives. Use `--resume` if the run breaks; it skips what is already on disk.

## What you get

`corpus/documents.jsonl`, one record per document:

```json
{
  "document_number": "2026-17250",
  "title": "Request for Information: Categories Used in Federal Vaccine ...",
  "type": "Notice",
  "action": "Request for information.",
  "abstract": "The Department of Health and Human Services ...",
  "publication_date": "2026-08-24",
  "comments_close_on": "2026-10-23",
  "citation": "91 FR 54001",
  "start_page": 54001,
  "docket_ids": ["HHS-OS-2026-0012"],
  "cfr_references": ["42 CFR 100"],
  "topics": ["Vaccines"],
  "significant": true,
  "agency_names": ["Health and Human Services Department"],
  "html_url": "https://www.federalregister.gov/documents/...",
  "text": "...full document text...",
  "word_count": 3820
}
```

`comments_close_on`, `effective_on`, `docket_ids`, `cfr_references` and
`citation` are the fields to build ground truth from. Each is a single
unambiguous fact that lives in exactly one document and that no model can
guess. `--stats` reports what percentage of the corpus carries each one.

## Corrections are excluded by default

Documents numbered `C1-`/`C2-`, or carrying a `correction_of` field, restate a
few lines of an earlier document. They are short, near-duplicate, and would
pollute a retrieval benchmark with almost-identical passages. Pass
`--keep-corrections` if you want them.

## Choosing a date range
## Choosing a date range

Pick a start date **after your model's training cutoff**, with a margin. If
you benchmark a newer model later, move the window forward — a corpus that was
unseen in August is not necessarily unseen next year. Record the window you
used alongside your results, because it is part of the experimental setup.

One honest caveat: significant rules attract news coverage, so a model may
know the *gist* of a major regulation. It will not know the docket number, the
comment deadline, or the wording of paragraph four. Build evaluation questions
around those specifics rather than around the policy topic, and the corpus
stays sound.

## Filtering for quality

Many Federal Register documents are two-paragraph meeting notices. They're
noise for retrieval testing, so `--min-words 300` drops them by default, and
`--stats` reports what fraction was short.

For a denser corpus, `--types RULE PRORULE` keeps rules and proposed rules,
which run to thousands of words with real internal structure and are much
closer to what an enterprise document estate looks like.

## Verify

```bash
python -m pytest tests/ -q
```

Tests cover text extraction, entity decoding, weekend skipping, and the
`raw_text_url` fallback. They run offline against fixtures — no network.

## Inspecting the corpus

```bash
python inspect_corpus.py                  # report only
python inspect_corpus.py --dev            # also write corpus/dev.jsonl
python inspect_corpus.py --threshold 0.5  # catch looser near-duplicates
```

Reports three things before any retrieval code exists:

**Length distribution**, flagging any document holding more than 5% of the
corpus on its own. A single very long rule distorts an index and makes
document-level ground truth meaningless - "the answer is in document X" says
nothing when X is 800,000 words. Those need section-level retrieval.

**Exact duplicates.** Same text under two document numbers means two correct
answers to one question, which silently corrupts a benchmark. Rare, but it
happens with reissued rules, and it is invisible until you look.

**Near-duplicate clusters**, plus a nearest-neighbour similarity distribution.
Do not treat near-duplicates as noise to remove: near-identical documents
differing in a few tokens are the hard retrieval case, and the one that
matters commercially - it is the same shape as an enterprise estate holding
forty versions of one policy where only one is current.

The distribution exists because no fixed threshold is right for every corpus.
Boilerplate-heavy documents cluster at a very different cutoff from distinct
ones, so the histogram shows where the natural gap falls and you set
`--threshold` just below the cluster you want to catch.

Similarity uses a bottom-k sketch of word 5-gram hashes - the k smallest
hashes stand in for the full shingle set, which is far cheaper than MinHash
with k permutations and exact whenever a document has fewer than k unique
shingles. No third-party dependencies.

## The development slice

`--dev` writes `corpus/dev.jsonl`: about 40 documents, excluding anything over
`--max-words`, spread round-robin across agencies, and **seeded with a
near-duplicate cluster on purpose**. A dev set of only distinct documents
makes every retriever look competent; keeping the hard case in view while
iterating is the entire point.

Use the slice to develop the chunker and retriever - it runs in seconds. Use
the full corpus for any number you would show a client. Same pipeline, one
flag, nothing re-downloaded.

## Step 2: the evaluation set

```bash
python make_eval.py                    # from corpus/dev.jsonl
python make_eval.py --source corpus/documents.jsonl --n 60
python make_eval.py --report
```

Writes `eval/questions.jsonl`. Build this **before** the retriever, or you
will tune on impressions and have no number to show anyone.

### The rule that makes it valid

A candidate answer is kept only if it appears **verbatim in its own document**
and **in no other document in the corpus**.

Metadata alone is not enough. `comments_close_on: "2026-10-23"` is a true fact
about a document, but if that date is never written in the prose then
retrieval cannot find it - the question is unanswerable while looking
answerable, and your retriever gets blamed for a corpus defect. And if forty
documents share the date, two answers are "correct" and the score is fiction.

Both checks run against the **full** corpus even when questions are drawn from
the dev slice, because a question unique within 40 documents may be ambiguous
within 415.

Dates are checked in written form (`October 23, 2026`), never ISO, since that
is how the prose states them.

### Three question types

**fact** - one specific value from one document. Easy, and most of the set.

**discriminator** - drawn from near-duplicate families, where several
documents look alike and only one carries the answer. These are the questions
that actually test retrieval, and the sibling document numbers are recorded as
`distractors` so you can see what the retriever confused it with.

**unanswerable** - plausible questions with no answer in the corpus. The
correct response is a refusal. This category is usually skipped and it is the
one enterprise buyers care most about: a system that answers everything is
worse than one that admits a gap.

### Two design details worth knowing

Discriminators are generated **first**, and fact questions then skip those
documents. Both builders phrase questions identically for a given document, so
generating facts first silently deletes the discriminators as duplicates - the
most valuable questions in the set, gone without a warning.

CFR references are never used as answers. They are shared by hundreds of
documents and can never be unique.

Questions name the agency and the distinguishing part of the title, never the
document number - otherwise the question is a trivial string match and
measures nothing.

### Hand-review before trusting it

The phrasing is templated. A generated question counts only if a person would
recognise it as something a real user might ask. Read the file, delete the
awkward ones, and rewrite a few by hand. `--report` flags any answer that is
not unique in the corpus; that count should always be zero.

## Answering questions

The retrieval side needs no model. Generation needs one, and any of these work:

| Provider | Env var | Free? |
|---|---|---|
| NVIDIA NIM | `NVIDIA_API_KEY` | free credits, OpenAI-compatible |
| Groq | `GROQ_API_KEY` | yes, no card, 30 req/min and 6k tokens/min |
| DeepSeek | `DEEPSEEK_API_KEY` | very cheap |
| OpenRouter | `OPENROUTER_API_KEY` | some `:free` models |
| Together | `TOGETHER_API_KEY` | trial credit |
| OpenAI | `OPENAI_API_KEY` | paid |
| Anthropic | `ANTHROPIC_API_KEY` | paid |
| Local | `OLLAMA_HOST` | free, runs on your machine |

Set one and it is detected automatically. `--backend` forces a specific one,
`--model` overrides the default model.

```bash
export NVIDIA_API_KEY=nvapi-...       # or GROQ_API_KEY=gsk_...
python rag.py --ask "What is the docket number for the Olcott safety zone?"
python evaluate.py --k 3 --generate --rpm 3 --by-type
```

### Why `--rpm` and `--k 3` on a free tier

Groq's free tier allows 30 requests a minute but only **6,000 tokens** a
minute. One RAG request carrying five passages is roughly 1,800 tokens, so
the token budget runs out long before the request count does. `--rpm 3` paces
the run to fit, and `--k 3` sends three passages instead of five, which nearly
halves the tokens. Sixty questions then take about twenty minutes.

If the provider still pushes back, the client honours the `Retry-After`
header and backs off rather than failing the run.

### Reasoning models

`deepseek-v4-flash` on NVIDIA is a reasoning model. Two things follow.

Its scratchpad arrives in a separate `reasoning_content` field, which is never
the answer - the code reads `content` only, so the thinking cannot leak into a
scored answer.

More importantly, **reasoning tokens count against `max_tokens`**. A model that
thinks for 500 tokens with a 500-token budget returns an empty answer, and the
run scores zero for reasons that have nothing to do with retrieval. So thinking
is off by default: the answer is sitting in the passages already, and extraction
does not benefit from deliberation. If content ever comes back empty while
reasoning is present, the code says so rather than silently scoring a blank.

`--thinking` turns it on (at low effort) if you want to measure whether it
helps. Raise `--max-tokens` to about 3000 if you do.

### Comparing models is the point

The same command with a different `--backend` scores the same 60 questions
against a different model. That is how you find out what a small self-hosted
model actually costs you in answer quality versus a frontier one - a number
worth having if you ever pitch an on-premises deployment.

## Hand-written questions

```bash
python write_eval.py            # 15 documents, one at a time
python write_eval.py --review   # check what you wrote
python write_eval.py --merge    # combine with the generated set
```

### Why bother, when questions can be generated

Generated questions are built from the title, so they ask in the document's
own words. Real users do not. Someone asks "how long can I be off after
having a baby"; the document says "parental leave provisions". Retrieval is
hard *because* of that gap, and a title-derived benchmark never tests it.

This matters more than it sounds. It is also why **prepending metadata to
chunks cannot be measured on the generated set**: putting the title into every
chunk of a document, when the questions were made from that title, inflates
the score without improving retrieval. It is the fingerprint leak coming back
through the documents instead of the questions. Twenty hand-written questions
give you a set where that change can be judged honestly.

### What the tool enforces

The answer must appear verbatim in the document, checked as you type it - an
answer the document never states cannot be retrieved, and would show up later
as a retrieval failure that is really a benchmark defect.

Words reused from the title are flagged. The question still gets saved; you
just know it is easier than a real one, and `--review` counts them.

Work is saved as you go, and rerunning skips documents already covered, so
Ctrl-C is safe and you can do this in several sittings.

## Embeddings, and beating the baseline

```bash
pip install fastembed numpy
python embed.py --build                    # ~10-20 min once, then cached
python evaluate.py --k 5 --retriever bm25   --questions eval/questions_all.jsonl --by-type
python evaluate.py --k 5 --retriever dense  --questions eval/questions_all.jsonl --by-type
python evaluate.py --k 5 --retriever hybrid --questions eval/questions_all.jsonl --by-type
```

Three retrievers, one interface, same questions. That is the whole point: the
BM25 number is the bar, and dense or hybrid either clears it or does not.

**Model: `BAAI/bge-small-en-v1.5` via fastembed.** 133MB, 384 dimensions, and
it handles 512 tokens - which matters, because chunks here run to about 300
tokens and the more famous `all-MiniLM-L6-v2` truncates at 256. It would
silently discard the tail of every chunk, making answers near the end
unfindable for reasons nothing in the output would explain.

fastembed rather than sentence-transformers because the latter pulls in
PyTorch, roughly 2GB on Windows, to do arithmetic that ONNX Runtime does in
50MB.

### Fusion by rank, not by score

Hybrid uses Reciprocal Rank Fusion. BM25 scores are unbounded and shift with
the corpus; cosine similarity sits between -1 and 1. Adding them means the
balance drifts as the corpus changes, and any weight you "tune" is really
tuning that scale mismatch. RRF throws the magnitudes away and uses only the
positions, so there is nothing to tune and nothing to drift.

Fusion happens over the top 50 from each retriever, not the top 5 - otherwise
a chunk ranked sixth by both, which is exactly the kind of broad agreement
worth surfacing, could never appear.

### Caching

Vectors are cached to `corpus/embeddings.npz`, keyed on the model name and a
hash of the chunk text. Change the chunker and the cache invalidates itself.
Without that check you would score new chunks against old vectors and get
numbers that look plausible and mean nothing.

### For Finnish, later

Swap one string: `--embed-model intfloat/multilingual-e5-small` or
`BAAI/bge-m3`. This is where dense retrieval should win clearly, because
Finnish inflection breaks keyword matching in a way English does not. Same
harness, two languages, and the contrast is the sales argument.

## Reranking

```bash
python evaluate.py --k 5 --retriever bm25 --rerank \
    --questions eval/questions_all.jsonl --by-type
```

`--rerank` wraps whichever first pass you chose, so it composes with all of
them. No extra install - fastembed ships the cross-encoders.

### Why this is different from embeddings

An embedding model reads the question and the passage **separately**, turns
each into numbers, and compares them. It never sees the pair together, so it
can tell that two things are about similar subjects but not that one answers
the other. That is why it confused A53 with A372.

A cross-encoder reads the question and the passage **as one input** and scores
them jointly. Much more accurate, and far too slow for 30,000 chunks - but
fine over the 50 the first pass hands it.

So it is a second stage, never a replacement. It attacks **ranking**, not
recall: if the right document is not among the candidates, no reranker can
recover it. Which is exactly the gap here - recall@5 is 94% while rank-1 is
83%.

### Two details that matter

The per-document cap is **not** applied before reranking. Capping first can
throw away the passage that would have won, because the first pass ranked a
weaker passage from the same document above it.

There is **no cache**. Scores depend on the question, so every query pays for
50 cross-encoder passes. That is the main reason to keep `--rerank-depth` at
50 rather than 500, and it is a real latency cost in production - unlike
embeddings, which are computed once.

### Models

`Xenova/ms-marco-MiniLM-L-6-v2` is the default: ~80MB, fast.
`BAAI/bge-reranker-base` is larger and usually better.
`jinaai/jina-reranker-v2-base-multilingual` is the one for the Finnish corpus.

## Reviewing answers

```bash
python evaluate.py --k 5 --generate --questions eval/questions_all.jsonl
python review.py                 # adjudicate the disagreements
python review.py --report        # scores with verdicts applied
python review.py --drop-bad      # drop questions whose ground truth is wrong
```

### Why a human step at all

Exact-match scoring has a ceiling, and it is lower than it looks. Real
examples from one run, all scored wrong, all correct:

    "must vote unanimously"                vs  "Unanimous vote"
    "Sec. 648.102(c)(2) of 50 CFR"         vs  "50 CFR 648.102(c)(2)"
    "OI Docket No. 24-523 and MD ..."      vs  "OI Docket No. 24-523, MD ..."

Each could be fixed by loosening the matcher - stemming, citation parsing,
list comparison. Every loosening also makes it likelier that a genuinely wrong
answer is accepted, and a benchmark that silently accepts wrong answers is
worse than one that rejects a few right ones.

So the matcher stays strict and a person settles the disputes. It takes two
minutes and is more accurate than any regex.

### Verdicts persist, but only while the answer does

Verdicts are keyed on the question **and the generated text**. Change the
chunker, rerun, and any answer that came out differently returns to the review
queue - a stale "correct" cannot mask a regression. Answers that did not change
keep their verdict, so you review a handful rather than all of them.

### Four verdicts, and why refusal is separate

    correct       the matcher was wrong
    wrong         the matcher was right
    refusal       the model correctly declined - retrieval failed, not the model
    bad-question  the ground truth itself is wrong; drop it

`refusal` matters because answering wrongly and correctly declining are
opposite behaviours that a single accuracy number hides. The report separates
them, so you can state both the overall figure and the figure for questions
whose answer was actually retrieved.

`bad-question` earns its place too: in one run the model answered "R-1896"
where the ground truth said "Regulations O and Y" - the model was right and
the generated question had pulled a regulation name out of a docket field.
Without a way to record that, the same wrong question fails forever.

## Next

Step 2 is the evaluation set: 30–50 questions whose answers are single
specific facts inside single documents, each recorded with its
`document_number` as ground truth. Write those before the retriever exists.
The retriever will be easy; knowing whether it works will not be.
