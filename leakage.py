#!/usr/bin/env python3
"""Measure how much of the retrieval score is a fingerprint match.

Questions generated from document titles inherit the title's rare vocabulary.
"the Coast Guard rulemaking on Safety Zone for Lake Ontario, Olcott, NY"
contains "Olcott", which occurs in exactly one document out of 415. BM25 does
not have to understand anything to answer that - it matches a token back to
its only source. The retriever scores 100% and the number means nothing.

This reports, for every question, the rarest token it shares with its gold
document. A token occurring in one document is a fingerprint; the question
identifies the answer rather than describing it.

`--ablate` then writes a harder set with those fingerprint tokens removed.
The resulting questions read awkwardly - they are a diagnostic, not a
deliverable - but re-running the evaluation against them shows what retrieval
achieves when it cannot cheat. The gap between the two scores is the portion
of your headline number that was never real.

Usage:
    python leakage.py                       # report
    python leakage.py --max-df 3            # widen what counts as rare
    python leakage.py --ablate              # write eval/questions_hard.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import Counter, defaultdict

from rag import load_corpus, strip_boilerplate, tokenize

CORPUS = pathlib.Path("corpus/documents.jsonl")
QUESTIONS = pathlib.Path("eval/questions.jsonl")
HARD = pathlib.Path("eval/questions_hard.jsonl")

# Words that are rare but carry real meaning rather than identity.
STOP = {"the", "a", "an", "of", "for", "on", "in", "by", "to", "and", "is",
        "what", "when", "which", "does", "must", "date", "number", "rule",
        "rulemaking", "docket", "comments", "received", "effective", "take",
        "effect", "regulation", "identifier", "rin"}


def document_frequency(docs: list[dict]) -> dict[str, set[str]]:
    df: dict[str, set[str]] = defaultdict(set)
    for doc in docs:
        text = strip_boilerplate(doc.get("text", ""))
        for term in set(tokenize(f"{doc.get('title', '')} {text}")):
            df[term].add(doc["document_number"])
    return df


def analyse(question: dict, df: dict[str, set[str]],
            max_df: int) -> dict:
    gold = question.get("gold_doc")
    terms = [t for t in set(tokenize(question["question"]))
             if t not in STOP and not t.isdigit()]

    rare = []
    for term in terms:
        docs = df.get(term, set())
        if gold and gold in docs and len(docs) <= max_df:
            rare.append((len(docs), term))
    rare.sort()

    fingerprints = [t for n, t in rare if n == 1]
    return {
        "id": question.get("id"),
        "type": question["type"],
        "gold_doc": gold,
        "rarest_df": rare[0][0] if rare else None,
        "fingerprint_terms": fingerprints,
        "rare_terms": [t for _, t in rare],
    }


def content_words(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOP and not t.isdigit()]


def is_degenerate(text: str, min_content: int = 3) -> bool:
    """Has ablation removed so much that nothing identifies the document?

    "On what date does the Department rule on Rule take effect?" is not a hard
    question - it is an impossible one, and identical for every document.
    Scoring against stubs like this produces a fake-low number, exactly as
    misleading as the fake-high one it was meant to correct.
    """
    return len(set(content_words(text))) < min_content


def ablate(question: dict, drop: list[str]) -> str:
    """Remove fingerprint tokens from the question text."""
    text = question["question"]
    for term in sorted(drop, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(term)}\b", "", text, flags=re.I)
    text = re.sub(r"\s*,\s*,", ",", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.?])", r"\1", text)
    text = re.sub(r"(for|on|in)\s*\?", "?", text, flags=re.I)
    return text.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=pathlib.Path, default=CORPUS)
    ap.add_argument("--questions", type=pathlib.Path, default=QUESTIONS)
    ap.add_argument("--max-df", type=int, default=3,
                    help="a term in this many documents or fewer counts rare")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--min-content", type=int, default=3,
                    help="drop an ablated question left with fewer distinct "
                         "content words than this")
    args = ap.parse_args()

    docs = load_corpus(args.corpus)
    questions = [json.loads(l) for l in args.questions.open(encoding="utf-8")
                 if l.strip()]
    print(f"indexing {len(docs)} documents for term frequencies ...")
    df = document_frequency(docs)

    rows = [analyse(q, df, args.max_df) for q in questions
            if q.get("gold_doc")]

    fingerprinted = [r for r in rows if r["fingerprint_terms"]]
    rare_only = [r for r in rows
                 if not r["fingerprint_terms"] and r["rarest_df"]]
    clean = [r for r in rows if r["rarest_df"] is None]

    print(f"\n{len(rows)} answerable questions\n")
    print(f"  {len(fingerprinted):3d} contain a term unique to the gold "
          f"document ({len(fingerprinted) / len(rows):.0%})")
    print(f"  {len(rare_only):3d} share only terms appearing in "
          f"2-{args.max_df} documents")
    print(f"  {len(clean):3d} share no rare term - retrieval must actually work")

    by_type: dict[str, list] = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    print("\n  by question type:")
    for t, rs in sorted(by_type.items()):
        n = sum(1 for r in rs if r["fingerprint_terms"])
        print(f"    {t:16s} {n}/{len(rs)} fingerprinted")

    common = Counter(t for r in fingerprinted for t in r["fingerprint_terms"])
    if common:
        print("\n  example fingerprint terms:")
        for term, _ in common.most_common(8):
            print(f"    {term}")

    if not args.ablate:
        print("\nrun with --ablate to write a harder set with these removed")
        return 0

    out, degenerate = [], []
    for q in questions:
        row = next((r for r in rows if r["id"] == q.get("id")), None)
        hard = dict(q)
        if row and row["fingerprint_terms"]:
            rewritten = ablate(q, row["fingerprint_terms"])
            if is_degenerate(rewritten, args.min_content):
                degenerate.append(q.get("id"))
                continue          # drop rather than score an impossible stub
            hard["question"] = rewritten
            hard["removed_terms"] = row["fingerprint_terms"]
        out.append(hard)

    HARD.parent.mkdir(parents=True, exist_ok=True)
    with HARD.open("w", encoding="utf-8") as fh:
        for q in out:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")

    changed = sum(1 for q in out if q.get("removed_terms"))
    print(f"\n{len(out)} questions -> {HARD} ({changed} altered)")
    if degenerate:
        print(f"  {len(degenerate)} dropped - ablation left nothing to "
              f"identify the document: {', '.join(degenerate[:6])}")
        print("  a high drop count means the topic words ARE the identity, "
              "and paraphrasing is the only real fix")
    print("\nexamples:")
    for q in out:
        if q.get("removed_terms"):
            print(f"  removed {q['removed_terms']}")
            print(f"    {q['question'][:96]}")
            if sum(1 for x in out[:out.index(q) + 1]
                   if x.get("removed_terms")) >= 3:
                break
    print(f"\nnow compare:\n"
          f"  python evaluate.py --k 5\n"
          f"  python evaluate.py --k 5 --questions {HARD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
