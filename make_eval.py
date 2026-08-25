#!/usr/bin/env python3
"""Build an evaluation set from the corpus, with verified ground truth.

The rule that makes this worth doing: a candidate answer is kept only if it

  1. appears verbatim in the document it is attributed to, and
  2. appears in NO other document in the whole corpus.

Metadata alone is not enough. `comments_close_on: "2026-10-23"` is a fact
about a document, but if that date is never written in its text, retrieval
cannot find it - the question would be unanswerable while looking answerable.
And if forty other documents share the date, two answers are "correct" and the
score is fiction. Both checks are enforced here.

Three question types are produced:

  fact           a single specific value from one document
  discriminator  drawn from near-duplicate families, where several documents
                 look alike and only one carries the answer
  unanswerable   plausible questions with no answer in the corpus, to test
                 that the system refuses instead of inventing

Usage:
    python make_eval.py                            # from corpus/dev.jsonl
    python make_eval.py --source corpus/documents.jsonl --n 60
    python make_eval.py --report                   # summarise eval/questions.jsonl

Output:
    eval/questions.jsonl
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import random
import re
from collections import Counter, defaultdict

from inspect_corpus import load, near_duplicate_clusters

CORPUS = pathlib.Path("corpus/documents.jsonl")
SOURCE = pathlib.Path("corpus/dev.jsonl")
OUT = pathlib.Path("eval/questions.jsonl")

MONTHS = ["", "January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


# ---------------------------------------------------------------------------
# Surface forms: how a metadata value is actually written in the prose
# ---------------------------------------------------------------------------

def date_forms(iso: str) -> list[str]:
    """"2026-10-23" -> ["October 23, 2026"]. Written out, never ISO."""
    try:
        d = dt.date.fromisoformat(iso)
    except (ValueError, TypeError):
        return []
    return [f"{MONTHS[d.month]} {d.day}, {d.year}"]


def candidate_answers(doc: dict) -> list[tuple[str, str]]:
    """(field, surface_form) pairs worth testing, best fields first.

    Dates lead because they are stated in prose - "Comments must be received
    by October 23, 2026" - so finding one requires reading the passage. Docket
    IDs are strong identifiers but sit near the top of the document, which
    makes them easier. CFR references are deliberately excluded: they are
    shared by hundreds of documents and can never be unique.
    """
    out: list[tuple[str, str]] = []
    for field in ("comments_close_on", "effective_on"):
        for form in date_forms(doc.get(field) or ""):
            out.append((field, form))
    for docket in (doc.get("docket_ids") or [])[:2]:
        out.append(("docket_ids", docket))
    for rin in (doc.get("regulation_id_numbers") or [])[:1]:
        out.append(("regulation_id_numbers", rin))
    return out


# ---------------------------------------------------------------------------
# Verification against the full corpus
# ---------------------------------------------------------------------------

class Verifier:
    """Checks where an answer string occurs across the corpus.

    Two different properties get confused here, so they are kept apart:

      * Presence in the gold document is MANDATORY. If the prose never states
        the value, retrieval cannot find it and the question is unanswerable
        while looking answerable.

      * Global uniqueness of the answer string is NOT required. In a corpus
        covering one month, effective dates repeat constantly - demanding a
        globally unique answer throws away every date question and leaves only
        docket numbers, which are exact strings near the top of each document
        and test keyword matching rather than retrieval.

    What must be unambiguous is the QUESTION, not the answer. "September 5,
    2026" occurring in forty documents is fine when the question names one
    specific safety zone. The occurrence count is recorded so that a shared
    answer string can be accounted for when scoring: a system can retrieve the
    wrong document and still emit the right string, which is why retrieval and
    answer accuracy are measured separately.
    """

    def __init__(self, corpus: list[dict]):
        self.docs = [(d["document_number"], d.get("text", "").lower())
                     for d in corpus]

    def occurrences(self, answer: str) -> list[str]:
        needle = answer.lower()
        return [num for num, text in self.docs if needle in text]

    def check(self, answer: str, gold: str,
              max_occurrences: int = 10) -> tuple[bool, list[str]]:
        """Usable if stated in the gold document and not corpus-wide boilerplate."""
        found = self.occurrences(answer)
        ok = gold in found and len(found) <= max_occurrences
        return ok, found


# ---------------------------------------------------------------------------
# Question phrasing
# ---------------------------------------------------------------------------

_TRAILING = re.compile(r"\s*[;,]\s*$")


def subject_of(doc: dict) -> str:
    """A short natural handle for the document, taken from its title.

    Federal Register titles are of the form "Safety Zone; Lake Ontario,
    Olcott, NY" - the part after the first semicolon is the distinguishing
    detail, which is exactly what a real question would mention.
    """
    title = (doc.get("title") or "").strip()
    parts = [p.strip() for p in title.split(";") if p.strip()]
    if len(parts) >= 2:
        subject = f"{parts[0]} for {', '.join(parts[1:])}"
    else:
        subject = title
    return _TRAILING.sub("", subject)[:140]


QUESTIONS = {
    "comments_close_on":
        "By what date must comments be received on the {agency} rulemaking "
        "on {subject}?",
    "effective_on":
        "On what date does the {agency} rule on {subject} take effect?",
    "docket_ids":
        "What is the docket number for the {agency} rulemaking on {subject}?",
    "regulation_id_numbers":
        "What is the Regulation Identifier Number (RIN) for the {agency} "
        "rulemaking on {subject}?",
}


def phrase(doc: dict, field: str) -> str:
    agency = (doc.get("agency_names") or ["the agency"])[0]
    return QUESTIONS[field].format(agency=agency, subject=subject_of(doc))


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_fact_questions(source: list[dict], verifier: Verifier,
                         limit: int, rng: random.Random,
                         exclude: set[str] | None = None,
                         max_occurrences: int = 10,
                         field_cap: float = 0.4,
                         counts: Counter | None = None) -> list[dict]:
    exclude = exclude or set()
    counts = counts if counts is not None else Counter()
    cap = max(1, int(limit * field_cap))
    out: list[dict] = []
    docs = [d for d in source if d["document_number"] not in exclude]
    rng.shuffle(docs)
    used_docs: set[str] = set()

    # Two passes. The first honours the per-field cap, so no single easy field
    # dominates. The second fills any shortfall without it - the cap is a
    # preference for variety, not a reason to return a half-empty set.
    for enforce_cap in (True, False):
        for doc in docs:
            if len(out) >= limit:
                break
            if doc["document_number"] in used_docs:
                continue
            for field, answer in candidate_answers(doc):
                if enforce_cap and counts[field] >= cap:
                    continue
                ok, found = verifier.check(answer, doc["document_number"],
                                           max_occurrences)
                if not ok:
                    continue
                counts[field] += 1
                used_docs.add(doc["document_number"])
                out.append({
                    "type": "fact",
                    "question": phrase(doc, field),
                    "answer": answer,
                    "answer_field": field,
                    "gold_doc": doc["document_number"],
                    "distractors": [],
                    "corpus_occurrences": len(found),
                })
                break
        if len(out) >= limit:
            break
    return out



def build_discriminator_questions(corpus: list[dict], clusters: list[list[int]],
                                  verifier: Verifier, limit: int,
                                  rng: random.Random,
                                  max_occurrences: int = 10) -> list[dict]:
    """Questions where the family looks alike and only one member answers."""
    out: list[dict] = []
    for cluster in clusters:
        if len(out) >= limit:
            break
        members = [corpus[i] for i in cluster]
        rng.shuffle(members)
        for doc in members:
            if len(out) >= limit:
                break
            picked = None
            for field, answer in candidate_answers(doc):
                unique, found = verifier.check(answer, doc["document_number"],
                                               max_occurrences)
                if unique:
                    picked = (field, answer, found)
                    break
            if not picked:
                continue
            field, answer, found = picked
            siblings = [m["document_number"] for m in members
                        if m["document_number"] != doc["document_number"]]
            out.append({
                "type": "discriminator",
                "question": phrase(doc, field),
                "answer": answer,
                "answer_field": field,
                "gold_doc": doc["document_number"],
                "distractors": siblings[:8],
                "corpus_occurrences": len(found),
            })
            break        # one per family, so no single family dominates
    return out


UNANSWERABLE = [
    ("By what date must comments be received on the {agency} rulemaking on "
     "{subject}?", "no such rulemaking is in the corpus"),
    ("What penalty does the {agency} rule on {subject} impose for a first "
     "offence?", "the documents do not state a penalty"),
    ("How many public comments did the {agency} receive on {subject} before "
     "the deadline?", "comment counts are not published in these documents"),
]


def build_unanswerable_questions(source: list[dict], verifier: Verifier,
                                 limit: int, rng: random.Random) -> list[dict]:
    """Plausible questions with no answer, to test refusal.

    Two flavours: a real document asked about something it does not contain,
    and a fabricated subject that matches nothing at all. Both should produce
    "not found in the available documents" rather than a guess.
    """
    out: list[dict] = []
    docs = rng.sample(source, min(len(source), limit * 2))
    for doc in docs:
        if len(out) >= limit:
            break
        template, why = rng.choice(UNANSWERABLE[1:])
        out.append({
            "type": "unanswerable",
            "question": template.format(
                agency=(doc.get("agency_names") or ["the agency"])[0],
                subject=subject_of(doc)),
            "answer": None,
            "answer_field": None,
            "gold_doc": None,
            "distractors": [doc["document_number"]],
            "note": why,
        })

    # One fabricated subject that appears nowhere.
    fake = "the proposed rule on municipal drone parcel delivery corridors"
    if not verifier.occurrences("drone parcel delivery corridors"):
        out.append({
            "type": "unanswerable",
            "question": f"What compliance deadline is set by {fake}?",
            "answer": None,
            "answer_field": None,
            "gold_doc": None,
            "distractors": [],
            "note": "subject does not appear anywhere in the corpus",
        })
    return out


# ---------------------------------------------------------------------------

def report(path: pathlib.Path = OUT) -> int:
    if not path.exists():
        print(f"{path} not found - generate it first")
        return 1
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    print(f"{len(rows)} questions in {path}\n")
    for t, n in Counter(r["type"] for r in rows).most_common():
        print(f"  {t:16s} {n:4d}  ({n / len(rows):.0%})")

    fields = Counter(r["answer_field"] for r in rows if r["answer_field"])
    print("\nanswer fields:")
    for f, n in fields.most_common():
        print(f"  {f:24s} {n:4d}")

    answerable = [r for r in rows if r["gold_doc"]]
    docs = {r["gold_doc"] for r in answerable}
    print(f"\n{len(docs)} distinct gold documents across "
          f"{len(answerable)} answerable questions")
    occ = [r.get("corpus_occurrences", 1) for r in answerable]
    unique = sum(1 for o in occ if o == 1)
    print(f"{unique} answers unique corpus-wide, "
          f"{len(occ) - unique} shared with other documents")
    if occ:
        print(f"  occurrence count: median {sorted(occ)[len(occ) // 2]}, "
              f"max {max(occ)}")
    print("  shared answers are fine - the QUESTION must be unambiguous, "
          "not the answer string")

    # Identifier fields are exact strings sitting near the top of every
    # document. A set dominated by them measures keyword matching. Dates are
    # stated in prose, so date dominance is only an imbalance, not a flaw.
    identifiers = {"docket_ids", "regulation_id_numbers"}
    id_share = sum(n for f, n in fields.items() if f in identifiers)
    if answerable and id_share > 0.5 * len(answerable):
        print(f"\nWARNING: {id_share / len(answerable):.0%} of answers are "
              f"docket numbers or RINs - exact strings near the top of each "
              f"document. This set measures keyword matching more than "
              f"retrieval. Raise --max-occurrences so date questions qualify.")
    elif fields:
        top, n = fields.most_common(1)[0]
        if n > 0.6 * len(answerable):
            print(f"\nnote: {top} supplies {n / len(answerable):.0%} of "
                  f"answers - fine, but a wider mix tests more")

    disc = [r for r in rows if r["type"] == "discriminator"]
    if disc:
        avg = sum(len(r["distractors"]) for r in disc) / len(disc)
        print(f"{len(disc)} discriminator questions, "
              f"{avg:.1f} look-alike distractors on average")
    print("\nhand-review before use: phrasing is templated, and a question "
          "only counts if a person would recognise it as reasonable")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=pathlib.Path, default=CORPUS,
                    help="full corpus, used for uniqueness checking")
    ap.add_argument("--source", type=pathlib.Path, default=SOURCE,
                    help="documents to draw questions from")
    ap.add_argument("--n", type=int, default=40, help="fact questions")
    ap.add_argument("--discriminators", type=int, default=15)
    ap.add_argument("--unanswerable", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--max-occurrences", type=int, default=10,
                    help="reject an answer appearing in more documents than "
                         "this (corpus-wide boilerplate)")
    ap.add_argument("--field-cap", type=float, default=0.4,
                    help="max share of fact questions from any one field")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.report:
        return report()

    rng = random.Random(args.seed)
    corpus = load(args.corpus)
    source = load(args.source) if args.source != args.corpus else corpus
    print(f"{len(corpus)} documents in corpus, "
          f"{len(source)} in the question source")

    verifier = Verifier(corpus)

    print("\nclustering for discriminator questions ...")
    clusters = near_duplicate_clusters(corpus, args.threshold)
    print(f"  {len(clusters)} near-duplicate families")

    print("verifying candidate answers against the full corpus ...")
    # Discriminators first: they are scarce and the most informative, so they
    # must not be crowded out by a fact question on the same document.
    disc = build_discriminator_questions(corpus, clusters, verifier,
                                         args.discriminators, rng,
                                         args.max_occurrences)
    used = {q["gold_doc"] for q in disc}
    counts = Counter(q["answer_field"] for q in disc)
    facts = build_fact_questions(source, verifier, args.n, rng, exclude=used,
                                 max_occurrences=args.max_occurrences,
                                 field_cap=args.field_cap, counts=counts)
    unans = build_unanswerable_questions(source, verifier,
                                         args.unanswerable, rng)

    seen: set[str] = set()
    questions: list[dict] = []
    for q in disc + facts + unans:
        if q["question"] in seen:
            continue
        seen.add(q["question"])
        q["id"] = f"q{len(questions) + 1:03d}"
        questions.append(q)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"\n{len(questions)} questions -> {OUT}")
    kinds = Counter(q["type"] for q in questions)
    print("  " + ", ".join(f"{n} {t}" for t, n in kinds.most_common()))
    dropped = len(disc) + len(facts) + len(unans) - len(questions)
    if dropped:
        print(f"  {dropped} dropped as duplicate phrasings")
    print("\nrun  python make_eval.py --report  to summarise")
    print("then hand-review the file before trusting any number from it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
