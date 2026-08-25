#!/usr/bin/env python3
"""Record human verdicts on answers, so a review survives the next run.

Exact-match scoring has a ceiling. It rejects answers that are correct but
phrased differently - "must vote unanimously" against "Unanimous vote",
"Sec. 648.102(c)(2) of 50 CFR" against "50 CFR 648.102(c)(2)". Each of those
could be fixed by loosening the matcher, and every loosening makes it more
likely that a genuinely wrong answer slips through unnoticed. A benchmark that
silently accepts wrong answers is worse than one that rejects a few right ones.

So the matcher stays strict and a person adjudicates the disagreements. This
tool shows you only those, records what you decide in `eval/verdicts.json`,
and replays your decisions afterwards - keyed on the question and the exact
generated text, so a verdict only applies while the answer is unchanged.

Four verdicts:

    correct     the automated score was wrong; the answer is right
    wrong       the automated score was right
    refusal     the model correctly declined - retrieval failed, not the model
    bad-question  the ground truth itself is wrong; drop the question

Usage:
    python review.py                 # adjudicate what the matcher disagreed on
    python review.py --all           # go through every answered question
    python review.py --report        # scores with verdicts applied
    python review.py --drop-bad      # write an eval set without bad questions
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

RESULTS = pathlib.Path("eval/results.jsonl")
VERDICTS = pathlib.Path("eval/verdicts.json")
QUESTIONS = pathlib.Path("eval/questions_all.jsonl")
CLEANED = pathlib.Path("eval/questions_reviewed.jsonl")

CHOICES = {
    "c": "correct",
    "w": "wrong",
    "r": "refusal",
    "b": "bad-question",
}


def answer_key(row: dict) -> str:
    """Identity of a question/answer pair.

    Keyed on the generated text as well as the question, so a verdict stops
    applying the moment the model says something different. Otherwise a stale
    "correct" would mask a regression.
    """
    raw = f"{row.get('id')}|{row.get('question')}|{row.get('generated', '')}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def load_results(path: pathlib.Path = RESULTS) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} not found - run evaluate.py --generate first")
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def load_verdicts(path: pathlib.Path = VERDICTS) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_verdicts(verdicts: dict, path: pathlib.Path = VERDICTS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(verdicts, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def needs_review(row: dict) -> bool:
    """Only the disagreements: scored wrong, but an answer was produced."""
    return (row.get("answer_correct") is False
            and bool(row.get("generated"))
            and not row.get("generation_error"))


def effective_correct(row: dict, verdicts: dict) -> bool | None:
    """Automated score, overridden by a human verdict where one exists."""
    verdict = verdicts.get(answer_key(row), {}).get("verdict")
    if verdict == "correct":
        return True
    if verdict in ("wrong", "refusal"):
        return False
    if verdict == "bad-question":
        return None
    return row.get("answer_correct")


def show(row: dict, index: int, total: int) -> None:
    print("\n" + "=" * 74)
    print(f"[{index}/{total}]  {row.get('id')}  ({row.get('type')})")
    print(f"Q    : {row['question'][:150]}")
    print(f"WANT : {row.get('expected')}")
    print(f"GOT  : {str(row.get('generated', ''))[:300]}")
    print("=" * 74)


def adjudicate(rows: list[dict], verdicts: dict) -> int:
    pending = [r for r in rows if answer_key(r) not in verdicts]
    if not pending:
        print(f"nothing new to review ({len(rows)} already judged)")
        return 0

    print(f"{len(pending)} to review "
          f"({len(rows) - len(pending)} already judged)\n")
    print("  [c]orrect   the matcher was wrong, the answer is right")
    print("  [w]rong     the matcher was right")
    print("  [r]efusal   correctly declined - retrieval failed, not the model")
    print("  [b]ad       the ground truth is wrong; drop this question")
    print("  [s]kip      decide later      [q]uit - answers are saved")

    for i, row in enumerate(pending, 1):
        show(row, i, len(pending))
        while True:
            try:
                choice = input("verdict [c/w/r/b/s/q] > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nstopping - verdicts saved")
                return 0
            if choice == "q":
                print("saved")
                return 0
            if choice == "s":
                break
            if choice in CHOICES:
                verdicts[answer_key(row)] = {
                    "verdict": CHOICES[choice],
                    "id": row.get("id"),
                    "question": row.get("question"),
                    "expected": row.get("expected"),
                    "generated": row.get("generated"),
                }
                save_verdicts(verdicts)
                print(f"  -> {CHOICES[choice]}")
                break
            print("  c, w, r, b, s or q")
    return 0


def report(rows: list[dict], verdicts: dict) -> int:
    answered = [r for r in rows if r.get("answer_correct") is not None]
    auto = sum(1 for r in answered if r.get("answer_correct"))

    reviewed, refusals, bad = 0, 0, 0
    for row in answered:
        verdict = verdicts.get(answer_key(row), {}).get("verdict")
        if verdict == "correct":
            reviewed += 1
        elif verdict == "refusal":
            refusals += 1
        elif verdict == "bad-question":
            bad += 1

    scorable = len(answered) - bad
    final = auto + reviewed

    print(f"{len(rows)} results, {len(answered)} scored for answers\n")
    print(f"  automated        {auto}/{len(answered)} "
          f"({auto / max(len(answered), 1):.0%})")
    if reviewed:
        print(f"  + reviewed right {reviewed}  (correct, phrased differently)")
    if bad:
        print(f"  - bad questions  {bad}  (ground truth wrong)")
    print(f"  reviewed score   {final}/{max(scorable, 1)} "
          f"({final / max(scorable, 1):.0%})")

    if refusals:
        print(f"\n  {refusals} correct refusals counted as answer failures.")
        print(f"  Excluding them: {final}/{max(scorable - refusals, 1)} "
              f"({final / max(scorable - refusals, 1):.0%}) of questions "
              f"whose answer was actually retrieved.")

    unjudged = [r for r in answered if needs_review(r)
                and answer_key(r) not in verdicts]
    if unjudged:
        print(f"\n  {len(unjudged)} disagreements not yet reviewed - "
              f"run without --report")
    return 0


def drop_bad(rows: list[dict], verdicts: dict,
             questions: pathlib.Path = QUESTIONS,
             out: pathlib.Path = CLEANED) -> int:
    bad_ids = {v["id"] for v in verdicts.values()
               if v.get("verdict") == "bad-question"}
    if not questions.exists():
        print(f"{questions} not found", file=sys.stderr)
        return 1

    kept = [json.loads(l) for l in questions.open(encoding="utf-8")
            if l.strip() and json.loads(l).get("id") not in bad_ids]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for q in kept:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"{len(kept)} questions -> {out}")
    print(f"  {len(bad_ids)} dropped as bad ground truth: "
          f"{', '.join(sorted(bad_ids))}" if bad_ids else "  none dropped")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=pathlib.Path, default=RESULTS)
    ap.add_argument("--questions", type=pathlib.Path, default=QUESTIONS)
    ap.add_argument("--all", action="store_true",
                    help="review every answered question, not just disputes")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--drop-bad", action="store_true")
    args = ap.parse_args()

    rows = load_results(args.results)
    verdicts = load_verdicts()

    if args.report:
        return report(rows, verdicts)
    if args.drop_bad:
        return drop_bad(rows, verdicts, args.questions)

    subject = [r for r in rows
               if (r.get("answer_correct") is not None and bool(r.get("generated")))
               if args.all or needs_review(r)]
    if not subject:
        print("no answers to review")
        return 0
    adjudicate(subject, verdicts)
    print("\nnow run:  python review.py --report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
