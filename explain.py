#!/usr/bin/env python3
"""Diagnose why retrieval ranks what it ranks.

Two checks, in the order worth running them.

`--duplicates` finds questions that are identical or near-identical to each
other after ablation. Two questions with the same text but different gold
documents are not hard - they are impossible, and no retriever can separate
them. A tie broken the same way every time then looks exactly like a ranking
failure, and chasing it wastes effort on a benchmark defect.

`--why` breaks a single query's BM25 score into per-term contributions across
candidate documents, so "why does this document keep winning" becomes a table
rather than a guess. Usually the answer is length normalisation: a short
document repeating the query terms outscores a long one that says the same
thing once.

Usage:
    python explain.py --duplicates --questions eval/questions_hard.jsonl
    python explain.py --why "Pipeline Safety Standards Update ASTM" \\
                      --docs 2026-15815 2026-15570 2026-15584
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
from collections import Counter, defaultdict

from rag import build_index, tokenize

QUESTIONS = pathlib.Path("eval/questions.jsonl")


def load_questions(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} not found")
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def normalise_question(text: str) -> str:
    """Compare questions as a person would read them.

    The index tokenizer keeps "update-astm" and "a53/a53m" whole, which is
    right for retrieval but wrong here: two questions differing only in
    punctuation are the same question. Split on everything non-alphanumeric.
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    return " ".join(sorted(set(words)))


def find_duplicates(questions: list[dict]) -> list[list[dict]]:
    """Group questions whose token sets are identical."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        if q.get("gold_doc"):
            groups[normalise_question(q["question"])].append(q)
    return [g for g in groups.values() if len(g) > 1]


def report_duplicates(questions: list[dict]) -> int:
    answerable = [q for q in questions if q.get("gold_doc")]
    dupes = find_duplicates(questions)

    if not dupes:
        print(f"{len(answerable)} answerable questions, all distinct")
        print("no impossible questions - ranking failures are real")
        return 0

    affected = sum(len(g) for g in dupes)
    print(f"{len(answerable)} answerable questions\n")
    print(f"{len(dupes)} group(s) of identical questions covering "
          f"{affected} questions ({affected / len(answerable):.0%})\n")
    print("These have the same text but different correct documents, so at "
          "most one\ncan be answered. They are not hard questions - they are "
          "impossible ones,\nand they drag the score down for a reason that "
          "has nothing to do with retrieval.\n")

    for group in sorted(dupes, key=len, reverse=True)[:5]:
        ids = ", ".join(q.get("id", "?") for q in group)
        golds = ", ".join(q["gold_doc"] for q in group)
        print(f"  {len(group)} questions: {ids}")
        print(f"    text : {group[0]['question'][:96]}")
        print(f"    golds: {golds}")
        removed = group[0].get("removed_terms")
        if removed:
            print(f"    ablation removed: {removed}")
        print()

    print("Fix: exclude these, or rewrite them so each names something its "
          "siblings do not.")
    print("     --drop-duplicates writes a cleaned set you can re-score.")
    return 0


CLEAN = pathlib.Path("eval/questions_clean.jsonl")


def drop_duplicates(questions: list[dict], out: pathlib.Path = CLEAN) -> int:
    """Write a set with impossible questions removed.

    A question whose text is shared with a different gold document cannot be
    answered by any retriever. Leaving it in does not make the benchmark
    harder, it makes it wrong - the score is reduced by an amount that has
    nothing to do with the system being measured.
    """
    dupes = find_duplicates(questions)
    drop = {q.get("id") for group in dupes for q in group}

    kept = [q for q in questions if q.get("id") not in drop]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for q in kept:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")

    answerable = sum(1 for q in kept if q.get("gold_doc"))
    print(f"\n{len(kept)} questions -> {out}")
    print(f"  {len(drop)} impossible questions removed: "
          f"{', '.join(sorted(x for x in drop if x))}")
    print(f"  {answerable} answerable questions remain")
    print(f"\nre-score with:\n  python evaluate.py --k 5 --questions {out}")
    return 0


def explain_ranking(index, query: str, doc_ids: list[str], top: int = 3) -> int:
    """Break a query's BM25 score into per-term contributions per document."""
    n = len(index.chunks)
    terms = sorted(set(tokenize(query)))

    per_doc: dict[str, list] = defaultdict(list)
    for i, chunk in enumerate(index.chunks):
        if doc_ids and chunk.doc_id not in doc_ids:
            continue
        counts = Counter(tokenize(chunk.text))
        breakdown, total = {}, 0.0
        for term in terms:
            posting = index.postings.get(term)
            if not posting:
                continue
            freq = counts.get(term, 0)
            if not freq:
                continue
            idf = math.log(1 + (n - len(posting) + 0.5) / (len(posting) + 0.5))
            norm = 1 - index.b + index.b * (index.lengths[i] / (index.avg_len or 1))
            score = idf * (freq * (index.k1 + 1)) / (freq + index.k1 * norm)
            breakdown[term] = (freq, score)
            total += score
        if total:
            per_doc[chunk.doc_id].append((total, chunk, breakdown, index.lengths[i]))

    if not per_doc:
        print("no chunk from those documents matched any query term")
        return 1

    best = sorted(((max(v)[0], d) for d, v in per_doc.items()), reverse=True)

    print(f"query: {query}")
    print(f"terms: {', '.join(terms)}\n")
    print(f"{'document':16s} {'score':>7s} {'chunk len':>10s}  top terms")
    print("-" * 74)
    for total, doc_id in best[:10]:
        score, chunk, breakdown, length = max(per_doc[doc_id])
        top_terms = sorted(breakdown.items(), key=lambda kv: -kv[1][1])[:4]
        detail = "  ".join(f"{t}x{f}={s:.1f}" for t, (f, s) in top_terms)
        print(f"{doc_id:16s} {score:7.2f} {length:10d}  {detail}")

    print(f"\naverage chunk length in the index: {index.avg_len:.0f} tokens")
    lengths = [max(per_doc[d])[3] for _, d in best[:10]]
    if lengths and min(lengths) < 0.6 * index.avg_len:
        print("\nthe leading chunk is well below average length - BM25 divides "
              "by\nlength, so a short chunk repeating the query terms "
              "outscores a longer\none that says the same thing once")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=pathlib.Path,
                    default=pathlib.Path("corpus/documents.jsonl"))
    ap.add_argument("--questions", type=pathlib.Path, default=QUESTIONS)
    ap.add_argument("--duplicates", action="store_true")
    ap.add_argument("--drop-duplicates", action="store_true",
                    help="write eval/questions_clean.jsonl without them")
    ap.add_argument("--why", metavar="QUERY")
    ap.add_argument("--docs", nargs="*", default=[])
    args = ap.parse_args()

    if args.duplicates or args.drop_duplicates:
        questions = load_questions(args.questions)
        report_duplicates(questions)
        if args.drop_duplicates:
            return drop_duplicates(questions)
        return 0
    if args.why:
        return explain_ranking(build_index(args.corpus), args.why, args.docs)

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
