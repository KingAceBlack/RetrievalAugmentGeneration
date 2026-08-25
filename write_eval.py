#!/usr/bin/env python3
"""Write evaluation questions by hand, quickly.

Generated questions borrow the document's own vocabulary - they are built from
the title, so they ask in the language the document already uses. Real users
do not. Someone asks "how long can I be off after having a baby"; the document
says "parental leave provisions". Retrieval is hard precisely because of that
gap, and a benchmark made of title-derived questions never tests it.

This tool shows you one document at a time - title, agency, and the passages
most likely to contain a checkable fact - and you type a question in your own
words. It records the answer, the gold document, and where in the text the
answer appears, so the result is verified rather than asserted.

Rules it enforces so the set stays honest:

  * the answer must appear verbatim in the document, or it is rejected
  * words you reuse from the title are flagged, because they are the leak
  * you can skip any document that has nothing worth asking about

Usage:
    python write_eval.py                          # 15 documents from the dev slice
    python write_eval.py --n 25 --corpus corpus/documents.jsonl
    python write_eval.py --review                 # check what you wrote
    python write_eval.py --merge                  # combine with generated set

Output:
    eval/questions_manual.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import sys

from rag import load_corpus, strip_boilerplate

SOURCE = pathlib.Path("corpus/dev.jsonl")
OUT = pathlib.Path("eval/questions_manual.jsonl")
GENERATED = pathlib.Path("eval/questions_clean.jsonl")
MERGED = pathlib.Path("eval/questions_all.jsonl")

# Lines that usually carry a checkable fact.
FACT_MARKERS = re.compile(
    r"^\s*(DATES?|ADDRESSES|EFFECTIVE|COMMENTS?|FOR FURTHER INFORMATION|"
    r"SUMMARY|ACTION|Docket|RIN)\b", re.I | re.M)


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def title_words(title: str) -> set[str]:
    stop = {"the", "of", "for", "and", "a", "an", "on", "in", "to", "rule",
            "proposed", "rules", "notice", "amendment", "establishment"}
    return {w for w in normalise(title).split() if w not in stop and len(w) > 2}


def leaked_words(question: str, title: str) -> list[str]:
    return sorted(set(normalise(question).split()) & title_words(title))


def interesting_passages(text: str, limit: int = 6) -> list[str]:
    """Lines likely to hold a specific, checkable value."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    picked = [l for l in lines if FACT_MARKERS.match(l)]
    if len(picked) < limit:
        for line in lines:
            if line in picked:
                continue
            if re.search(r"\b(19|20)\d{2}\b|\b\d{1,2}:\d{2}\b|\$\d", line):
                picked.append(line)
            if len(picked) >= limit:
                break
    return picked[:limit]


def show(doc: dict, index: int, total: int) -> None:
    text = strip_boilerplate(doc.get("text", ""))
    print("\n" + "=" * 74)
    print(f"[{index}/{total}]  {doc['document_number']}   "
          f"{doc.get('word_count', 0):,} words")
    print(f"title  : {doc.get('title', '')}")
    print(f"agency : {', '.join(doc.get('agency_names') or ['?'])}")
    for field in ("comments_close_on", "effective_on", "docket_ids",
                  "cfr_references", "citation"):
        if doc.get(field):
            print(f"{field:14s}: {doc[field]}")
    print("-" * 74)
    for line in interesting_passages(text):
        print(f"  {line[:150]}")
    print("=" * 74)
    print("Ask what a real user would ask - avoid the title's words.")


def prompt(label: str) -> str:
    try:
        return input(label).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nstopping")
        raise SystemExit(0)


def capture(doc: dict) -> dict | None:
    """One question/answer pair, verified against the document text."""
    text = strip_boilerplate(doc.get("text", ""))

    question = prompt("\nquestion (blank to skip)  > ")
    if not question:
        return None

    leak = leaked_words(question, doc.get("title", ""))
    if leak:
        print(f"  ! reuses title words: {', '.join(leak)}")
        print("    that is the leak this set exists to avoid.")
        if prompt("    rephrase? [Y/n] > ").lower() not in ("n", "no"):
            question = prompt("question  > ") or question
            leak = leaked_words(question, doc.get("title", ""))

    while True:
        answer = prompt("answer (exact text from the document)  > ")
        if not answer:
            print("  skipped")
            return None
        if normalise(answer) in normalise(text):
            break
        print("  ! that text does not appear in the document.")
        print("    an answer the document never states cannot be retrieved.")
        if prompt("    try again? [Y/n] > ").lower() in ("n", "no"):
            return None

    return {
        "type": "manual",
        "question": question,
        "answer": answer,
        "answer_field": "manual",
        "gold_doc": doc["document_number"],
        "distractors": [],
        "title_words_reused": leak,
    }


def review(path: pathlib.Path = OUT) -> int:
    if not path.exists():
        print(f"{path} not found - write some questions first")
        return 1
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    print(f"{len(rows)} hand-written questions\n")
    leaky = [r for r in rows if r.get("title_words_reused")]
    for r in rows:
        flag = "!" if r.get("title_words_reused") else " "
        print(f" {flag} {r['gold_doc']}  {r['question'][:78]}")
        print(f"     -> {r['answer'][:70]}")
    print(f"\n{len(rows) - len(leaky)} clean, {len(leaky)} reuse title words")
    if leaky:
        print("the flagged ones still work, they are just easier than real "
              "questions")
    return 0


def merge(manual: pathlib.Path = OUT, generated: pathlib.Path = GENERATED,
          out: pathlib.Path = MERGED) -> int:
    rows: list[dict] = []
    for path in (generated, manual):
        if path.exists():
            rows += [json.loads(l) for l in path.open(encoding="utf-8")
                     if l.strip()]
        else:
            print(f"  {path} not found, skipping")
    for i, r in enumerate(rows, 1):
        r["id"] = f"m{i:03d}"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} questions -> {out}")
    print(f"\nscore with:\n  python evaluate.py --k 5 --questions {out} --by-type")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=pathlib.Path, default=SOURCE)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-words", type=int, default=12_000,
                    help="skip documents longer than this - too slow to read")
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    if args.review:
        return review()
    if args.merge:
        return merge()

    docs = [d for d in load_corpus(args.corpus)
            if d.get("word_count", 0) <= args.max_words]
    if not docs:
        print("no documents short enough - raise --max-words")
        return 1

    done = set()
    if OUT.exists():
        done = {json.loads(l)["gold_doc"]
                for l in OUT.open(encoding="utf-8") if l.strip()}
        print(f"{len(done)} questions already written, skipping those documents")

    docs = [d for d in docs if d["document_number"] not in done]
    random.Random(args.seed).shuffle(docs)
    docs = docs[:args.n]

    print(f"\n{len(docs)} documents. Blank question skips. Ctrl-C stops - "
          f"answers are saved as you go.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with OUT.open("a", encoding="utf-8") as fh:
        for i, doc in enumerate(docs, 1):
            show(doc, i, len(docs))
            record = capture(doc)
            if record is None:
                continue
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            written += 1
            print(f"  saved ({written} so far)")

    print(f"\n{written} questions -> {OUT}")
    print("\nnext:\n  python write_eval.py --review\n"
          "  python write_eval.py --merge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
