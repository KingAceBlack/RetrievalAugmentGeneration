#!/usr/bin/env python3
"""Score the pipeline against eval/questions.jsonl.

Retrieval and answer accuracy are reported separately, and that separation is
the point. A shared answer string - a date that appears in thirty documents -
means a system can retrieve entirely the wrong passage and still emit the
right text. Reporting one number would hide that. So:

  retrieval    was the gold document among the retrieved passages?
               scored on every answerable question.

  passage      did any retrieved passage actually CONTAIN the answer?
               this is the number that predicts whether generation can work.
               Document-level retrieval can be perfect while the passages
               handed to the model hold no answer at all - the right document
               found, the wrong part of it retrieved.

  answer       did the generated text contain the expected value?
               scored only where `corpus_occurrences` is low enough that a
               right answer implies the right source (--max-occurrences).

  refusal      did the system decline on unanswerable questions?
               a system that answers everything is worse than one that
               admits a gap, and this is the number enterprise buyers ask
               about first.

Usage:
    python evaluate.py                       # retrieval only, no API needed
    python evaluate.py --generate            # also score answers and refusals
    python evaluate.py --k 10 --by-type
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
import unicodedata
from collections import defaultdict

from rag import REFUSAL, backend_name, build_index, generate, set_rate_limit

QUESTIONS = pathlib.Path("eval/questions.jsonl")
RESULTS = pathlib.Path("eval/results.jsonl")


def load_questions(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} not found - run make_eval.py first")
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


# Models emit typographic variants the source never used: a non-breaking
# hyphen inside a docket number, a curly apostrophe, an en dash in a date
# range. Folding them prevents a correct answer being scored wrong.
_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015"
                                 "\u2018\u2019\u201c\u201d\u00a0"), " ")


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "").translate(_DASHES)
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


# Words that only label an identifier. A model writing "the docket number is"
# where the metadata says "Docket No." has not changed the answer, so these are
# not required to appear. The identifier itself still must.
LABEL_WORDS = {"docket", "no", "nos", "number", "numbers", "rin", "id",
               "identifier", "regulation", "case", "sequence", "accession"}


def _digit_core(tokens: list[str]) -> list[str]:
    """The span from the first digit-bearing token to the last.

    For "docket number uscg 2026 1030" that is ["uscg", "2026", "1030"] -
    the part that identifies something, as opposed to the label describing it.
    """
    positions = [i for i, t in enumerate(tokens) if any(c.isdigit() for c in t)]
    if not positions:
        return []
    start = positions[0]
    # include an immediately preceding alphabetic prefix, so "uscg 2026 1030"
    # is kept whole rather than reduced to "2026 1030"
    if start and not any(c.isdigit() for c in tokens[start - 1]):
        if len(tokens[start - 1]) <= 6:
            start -= 1
    return tokens[start:positions[-1] + 1]


def contains_answer(generated: str, expected: str) -> bool:
    """Is the expected answer stated, allowing for natural phrasing?

    Exact containment is too strict. Expected answers taken from metadata
    carry their label - "Docket Number USCG-2026-1030" - while a model writes
    "The docket number is USCG-2026-1030". The inserted "is" breaks a
    contiguous match on an answer that is completely correct.

    So: exact containment first, then a fallback requiring every expected
    token to appear somewhere AND the identifying core - the digits and their
    prefix - to appear contiguously. That accepts rephrasing of the label
    while still rejecting a wrong identifier, since a changed digit breaks the
    core.
    """
    exp_norm, gen_norm = normalise(expected), normalise(generated)
    if not exp_norm:
        return False
    if exp_norm in gen_norm:
        return True

    exp_tokens, gen_tokens = exp_norm.split(), gen_norm.split()
    required = {t for t in exp_tokens if t not in LABEL_WORDS}
    if not required or not required <= set(gen_tokens):
        return False

    core = _digit_core(exp_tokens)
    if not core:
        return False                     # no identifier to anchor on
    return " ".join(core) in gen_norm


def is_refusal(generated: str) -> bool:
    g = normalise(generated)
    return (normalise(REFUSAL) in g
            or "not found in the available documents" in g
            or g.startswith("i cannot find") or g.startswith("i could not find"))


def pct(part: int, whole: int) -> str:
    return f"{part}/{whole} ({part / whole:.0%})" if whole else "n/a"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=pathlib.Path,
                    default=pathlib.Path("corpus/documents.jsonl"))
    ap.add_argument("--questions", type=pathlib.Path, default=QUESTIONS)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--per-doc", type=int, default=2,
                    help="max chunks from any one document")
    ap.add_argument("--generate", action="store_true",
                    help="also score answers (needs a generation backend)")
    ap.add_argument("--model")
    ap.add_argument("--backend")
    ap.add_argument("--rpm", type=float, default=0,
                    help="throttle requests per minute (Groq free tier: 3)")
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max-occurrences", type=int, default=5,
                    help="answer scoring only counts questions whose expected "
                         "value appears in at most this many documents")
    ap.add_argument("--retriever", default="bm25",
                    choices=["bm25", "dense", "hybrid"],
                    help="bm25 is the baseline; compare the others against it")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--rerank", nargs="?", const="Xenova/ms-marco-MiniLM-L-6-v2",
                    default=None, metavar="MODEL",
                    help="rerank the first pass with a cross-encoder")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="per-request seconds before retrying")
    ap.add_argument("--rerank-depth", type=int, default=50,
                    help="candidates to rerank (cost scales with this)")
    ap.add_argument("--by-type", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[:args.limit]

    set_rate_limit(args.rpm)
    if args.generate and backend_name(args.backend) is None:
        print("no generation backend - set GROQ_API_KEY, DEEPSEEK_API_KEY, "
              "ANTHROPIC_API_KEY, OLLAMA_HOST, or pass --backend",
              file=sys.stderr)
        return 1

    if args.retriever == "bm25" and not args.rerank:
        index = build_index(args.corpus)
    else:
        from embed import make_retriever
        index = make_retriever(args.retriever, args.corpus, args.embed_model,
                               rerank_model=args.rerank,
                               rerank_depth=args.rerank_depth)
    results: list[dict] = []
    started = time.monotonic()

    if args.generate:
        print(f"generating answers for {len(questions)} questions - "
              f"expect a few seconds each", file=sys.stderr, flush=True)

    for i, q in enumerate(questions, 1):
        hits = index.search(q["question"], args.k, args.per_doc)
        retrieved = []
        for chunk, _ in hits:
            if chunk.doc_id not in retrieved:
                retrieved.append(chunk.doc_id)

        rec = {
            "id": q.get("id"),
            "type": q["type"],
            "question": q["question"],
            "gold_doc": q.get("gold_doc"),
            "retrieved": retrieved,
            "corpus_occurrences": q.get("corpus_occurrences"),
            # Recording the expected answer beside the generated one is what
            # makes a wrong score debuggable without rerunning anything.
            "expected": q.get("answer"),
        }
        # Does any retrieved passage actually contain the expected answer?
        # A model cannot state what it was never shown.
        if q.get("answer"):
            rec["answer_in_context"] = any(
                contains_answer(chunk.text, q["answer"]) for chunk, _ in hits)
            rec["answer_in_gold_context"] = any(
                contains_answer(chunk.text, q["answer"])
                for chunk, _ in hits if chunk.doc_id == q.get("gold_doc"))

        if q.get("gold_doc"):
            rec["hit"] = q["gold_doc"] in retrieved
            rec["rank"] = (retrieved.index(q["gold_doc"]) + 1
                           if rec["hit"] else None)
            # which look-alike outranked the right document?
            rec["confused_with"] = (retrieved[0] if not rec["hit"] and retrieved
                                    else None)

        if args.generate:
            # One failed question must not end the run. Record the failure and
            # carry on; the summary reports how many were lost.
            try:
                answer = generate(q["question"], hits, args.model,
                                  args.backend, args.max_tokens,
                                  args.thinking, args.timeout) or ""
            except Exception as exc:
                print(f"    generation failed: {type(exc).__name__}",
                      file=sys.stderr, flush=True)
                rec["generation_error"] = f"{type(exc).__name__}: {exc}"[:200]
                answer = ""
            rec["generated"] = answer
            rec["refused"] = is_refusal(answer) if answer else False
            if q.get("answer") and "generation_error" not in rec:
                rec["answer_correct"] = contains_answer(answer, q["answer"])

        results.append(rec)

        # With generation each question takes seconds, so report every one.
        # Silence during the slowest step looks identical to a hang.
        if args.generate:
            elapsed = time.monotonic() - started
            eta = (elapsed / i) * (len(questions) - i)
            mark = ("ERR" if rec.get("generation_error")
                    else "ok " if rec.get("answer_correct")
                    else "ref" if rec.get("refused")
                    else "   ")
            print(f"  {i:3d}/{len(questions)} {mark} "
                  f"{q['type'][:6]:6s} {elapsed:5.0f}s elapsed, "
                  f"~{eta:.0f}s left", file=sys.stderr, flush=True)
        elif i % 10 == 0:
            print(f"  {i}/{len(questions)}", file=sys.stderr, flush=True)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---------------- reporting ----------------
    answerable = [r for r in results if r.get("gold_doc")]
    unanswerable = [r for r in results if r["type"] == "unanswerable"]

    label = args.retriever + (f" + rerank({args.rerank.split('/')[-1]})"
                              if args.rerank else "")
    print(f"\n{len(results)} questions, k={args.k}, retriever={label}\n")

    hits_n = sum(1 for r in answerable if r["hit"])
    print(f"retrieval  gold document in top {args.k}: "
          f"{pct(hits_n, len(answerable))}")
    ranks = [r["rank"] for r in answerable if r.get("rank")]
    if ranks:
        mrr = sum(1 / r for r in ranks) / len(answerable)
        print(f"           MRR {mrr:.3f}, "
              f"rank 1 for {pct(sum(1 for r in ranks if r == 1), len(answerable))}")

    # The gap between these two numbers is where answers get lost.
    with_ctx = [r for r in answerable if r.get("answer_in_context") is not None]
    if with_ctx:
        n_ctx = sum(1 for r in with_ctx if r["answer_in_context"])
        n_gold = sum(1 for r in with_ctx if r["answer_in_gold_context"])
        print(f"\npassage    answer text present in a retrieved passage: "
              f"{pct(n_ctx, len(with_ctx))}")
        print(f"           present in a passage from the GOLD document: "
              f"{pct(n_gold, len(with_ctx))}")
        gap = hits_n - n_gold
        if gap > 0:
            print(f"           {gap} questions retrieved the right document "
                  f"but not the part containing the answer")

    if args.by_type:
        print("\n           by question type:")
        by_type: dict[str, list[dict]] = defaultdict(list)
        for r in answerable:
            by_type[r["type"]].append(r)
        for t, rows in sorted(by_type.items()):
            n = sum(1 for r in rows if r["hit"])
            print(f"             {t:16s} {pct(n, len(rows))}")

    if args.generate:
        scorable = [r for r in answerable
                    if r.get("answer_correct") is not None
                    and (r.get("corpus_occurrences") or 1) <= args.max_occurrences]
        correct = sum(1 for r in scorable if r["answer_correct"])
        print(f"\nanswer     correct: {pct(correct, len(scorable))}"
              f"  (of questions whose answer appears in "
              f"<={args.max_occurrences} documents)")
        excluded = len(answerable) - len(scorable)
        if excluded:
            print(f"           {excluded} excluded - answer string too common "
                  f"to attribute to the right source")

        errors = [r for r in results if r.get("generation_error")]
        if errors:
            print(f"\n{len(errors)} question(s) failed to generate and are "
                  f"excluded from the scores above:")
            for r in errors[:3]:
                print(f"  {r['id']}  {r['generation_error'][:70]}")

        refused = sum(1 for r in unanswerable if r["refused"])
        print(f"\nrefusal    declined when there was no answer: "
              f"{pct(refused, len(unanswerable))}")
        wrong_refusals = sum(1 for r in answerable if r.get("refused"))
        if wrong_refusals:
            print(f"           {wrong_refusals} refusals on answerable "
                  f"questions (retrieval failed, model behaved correctly)")

    starved = [r for r in answerable
               if r.get("hit") and r.get("answer_in_gold_context") is False]
    if starved:
        print(f"\n{len(starved)} right-document-wrong-passage cases:")
        for r in starved[:5]:
            print(f"  {r['id']} ({r['type']}) {r['gold_doc']} rank "
                  f"{r.get('rank')}")
            print(f"     {r['question'][:88]}")

    misses = [r for r in answerable if not r["hit"]]
    if misses:
        print(f"\n{len(misses)} retrieval misses - the useful ones to read:")
        for r in misses[:5]:
            print(f"  {r['id']} ({r['type']}) gold {r['gold_doc']}, "
                  f"top hit {r['confused_with']}")
            print(f"     {r['question'][:88]}")

    print(f"\nfull results -> {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
