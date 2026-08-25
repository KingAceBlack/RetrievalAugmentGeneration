#!/usr/bin/env python3
"""Import hand-written questions and verify them against the corpus.

Input is a plain text file of three-line records:

    DOC: 2026-15368
    Q: How long must GSA wait after notifying Congress?
    A: 30 calendar days

Every answer is checked against the document it claims to come from. An answer
the document never states cannot be retrieved, and would later show up as a
retrieval failure that is really a transcription error - so it is rejected
here rather than quietly corrupting the score.

Matching ignores punctuation, case and whitespace, so a phone number written
"(800) 552-6458" matches "800-552-6458" in the text, and coordinates with
typographic quotes match plain ones.

Usage:
    python import_questions.py                       # verify only
    python import_questions.py --write               # write the eval file
    python import_questions.py --file mine.txt --write

Output:
    eval/questions_manual.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter

from rag import load_corpus, strip_boilerplate

SOURCE = pathlib.Path("manual_questions.txt")
CORPUS = pathlib.Path("corpus/documents.jsonl")
OUT = pathlib.Path("eval/questions_manual.jsonl")

RECORD = re.compile(
    r"^DOC:\s*(?P<doc>\S+)\s*\n"
    r"^Q:\s*(?P<q>.+?)\s*\n"
    r"^A:\s*(?P<a>.+?)\s*$",
    re.M)

# Typographic variants that appear in questions but not in the source text.
UNIFY = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u00a0": " ", "\u2032": "'",
    "\u2033": "''", "\u00b0": " ",
}


def normalise(text: str) -> str:
    for src, dst in UNIFY.items():
        text = text.replace(src, dst)
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def parse(text: str) -> list[dict]:
    records = []
    for m in RECORD.finditer(text):
        answer = m.group("a").strip()
        records.append({
            "gold_doc": m.group("doc").strip(),
            "question": m.group("q").strip(),
            "answer": None if answer.upper() == "NONE" else answer,
        })
    return records


def title_words(title: str) -> set[str]:
    stop = {"the", "of", "for", "and", "a", "an", "on", "in", "to", "rule",
            "rules", "proposed", "notice", "amendment", "establishment",
            "final", "act", "regulation", "regulations"}
    return {w for w in normalise(title).split()
            if w not in stop and len(w) > 3}


def verify(records: list[dict], docs: list[dict]) -> tuple[list[dict], list[dict]]:
    by_id = {d["document_number"]: d for d in docs}
    haystacks = {d["document_number"]: normalise(strip_boilerplate(d["text"]))
                 for d in docs}
    corpus_norm = list(haystacks.items())

    good, bad = [], []
    for rec in records:
        doc = by_id.get(rec["gold_doc"])
        if doc is None:
            rec["problem"] = "document not in corpus"
            bad.append(rec)
            continue

        if rec["answer"] is None:                       # deliberate refusal
            rec.update(type="unanswerable", answer_field=None,
                       gold_doc=None, distractors=[rec["gold_doc"]])
            good.append(rec)
            continue

        needle = normalise(rec["answer"])
        if needle not in haystacks[rec["gold_doc"]]:
            rec["problem"] = "answer text not found in that document"
            bad.append(rec)
            continue

        occurrences = [n for n, hay in corpus_norm if needle in hay]
        leak = sorted(set(normalise(rec["question"]).split())
                      & title_words(doc.get("title", "")))
        rec.update(
            type="manual",
            answer_field="manual",
            distractors=[],
            corpus_occurrences=len(occurrences),
            title_words_reused=leak,
        )
        good.append(rec)
    return good, bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=pathlib.Path, default=SOURCE)
    ap.add_argument("--corpus", type=pathlib.Path, default=CORPUS)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    if not args.file.exists():
        print(f"{args.file} not found", file=sys.stderr)
        return 1

    records = parse(args.file.read_text(encoding="utf-8"))
    if not records:
        print("no DOC/Q/A records found - check the file format")
        return 1

    docs = load_corpus(args.corpus)
    good, bad = verify(records, docs)

    print(f"{len(records)} questions across "
          f"{len({r['gold_doc'] for r in records})} documents\n")

    if bad:
        print(f"{len(bad)} REJECTED:\n")
        for rec in bad:
            print(f"  {rec['gold_doc']}  {rec['problem']}")
            print(f"    Q: {rec['question'][:78]}")
            print(f"    A: {(rec['answer'] or '')[:78]}")
            print()

    answerable = [r for r in good if r.get("gold_doc")]
    print(f"{len(good)} accepted ({len(answerable)} answerable, "
          f"{len(good) - len(answerable)} refusal tests)")

    leaky = [r for r in answerable if r.get("title_words_reused")]
    if leaky:
        print(f"\n{len(leaky)} reuse words from their document title "
              f"(easier than a real question):")
        for r in leaky:
            print(f"  {r['gold_doc']}  {', '.join(r['title_words_reused'])}")
            print(f"    {r['question'][:74]}")

    shared = [r for r in answerable if r.get("corpus_occurrences", 1) > 1]
    if shared:
        print(f"\n{len(shared)} answers also appear in other documents:")
        for r in shared:
            print(f"  {r['gold_doc']}  in {r['corpus_occurrences']} documents"
                  f"  '{r['answer'][:40]}'")
        print("  fine for retrieval scoring - the question still names one "
              "document")

    spread = Counter(r["gold_doc"] for r in answerable)
    print(f"\nquestions per document: "
          f"{', '.join(f'{d}={n}' for d, n in spread.most_common(5))}")

    if not args.write:
        print("\nrun with --write to save")
        return 0 if not bad else 1

    for i, rec in enumerate(good, 1):
        rec["id"] = f"h{i:03d}"
        rec.pop("problem", None)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for rec in good:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n{len(good)} questions -> {OUT}")
    print("\nnext:\n  python write_eval.py --merge\n"
          "  python evaluate.py --k 5 --questions eval/questions_all.jsonl "
          "--by-type")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
