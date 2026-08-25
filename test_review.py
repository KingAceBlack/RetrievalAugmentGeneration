import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from review import (  # noqa: E402
    answer_key, drop_bad, effective_correct, load_verdicts, needs_review,
    save_verdicts,
)


def row(rid, question, expected, generated, correct, rtype="fact"):
    return {"id": rid, "type": rtype, "question": question,
            "expected": expected, "generated": generated,
            "answer_correct": correct}


DISPUTED = row("m1", "What vote is required?", "Unanimous vote",
               "The ERC must vote unanimously.", False)
AGREED = row("m2", "When effective?", "July 6, 2026",
             "It is effective July 6, 2026.", True)
REFUSED = row("m3", "What is the docket?", "ML25268A122",
              "Not found in the available documents.", False)


# --- keying ----------------------------------------------------------------

def test_key_changes_when_the_answer_changes():
    """A stale verdict must not mask a regression."""
    other = dict(DISPUTED, generated="Something completely different.")
    assert answer_key(DISPUTED) != answer_key(other)


def test_key_changes_when_the_question_changes():
    other = dict(DISPUTED, question="A different question entirely?")
    assert answer_key(DISPUTED) != answer_key(other)


def test_key_is_stable_for_the_same_pair():
    assert answer_key(DISPUTED) == answer_key(dict(DISPUTED))


# --- what needs review -----------------------------------------------------

def test_disagreement_needs_review():
    assert needs_review(DISPUTED)


def test_agreement_does_not():
    assert not needs_review(AGREED)


def test_refusal_needs_review():
    """It scored wrong, but the model may have behaved correctly."""
    assert needs_review(REFUSED)


def test_empty_answer_skipped():
    assert not needs_review(dict(DISPUTED, generated=""))


def test_generation_error_skipped():
    assert not needs_review(dict(DISPUTED, generation_error="Timeout"))


# --- applying verdicts -----------------------------------------------------

def test_correct_verdict_overrides_the_matcher():
    verdicts = {answer_key(DISPUTED): {"verdict": "correct"}}
    assert effective_correct(DISPUTED, verdicts) is True


def test_wrong_verdict_confirms_the_matcher():
    verdicts = {answer_key(DISPUTED): {"verdict": "wrong"}}
    assert effective_correct(DISPUTED, verdicts) is False


def test_refusal_verdict_counts_as_not_correct():
    verdicts = {answer_key(REFUSED): {"verdict": "refusal"}}
    assert effective_correct(REFUSED, verdicts) is False


def test_bad_question_excludes_it_entirely():
    verdicts = {answer_key(DISPUTED): {"verdict": "bad-question"}}
    assert effective_correct(DISPUTED, verdicts) is None


def test_unjudged_falls_back_to_the_matcher():
    assert effective_correct(AGREED, {}) is True
    assert effective_correct(DISPUTED, {}) is False


def test_verdict_stops_applying_when_the_answer_changes():
    verdicts = {answer_key(DISPUTED): {"verdict": "correct"}}
    changed = dict(DISPUTED, generated="A totally different answer now.")
    assert effective_correct(changed, verdicts) is False


# --- persistence -----------------------------------------------------------

def test_verdicts_round_trip(tmp_path):
    path = tmp_path / "v.json"
    data = {answer_key(DISPUTED): {"verdict": "correct", "id": "m1"}}
    save_verdicts(data, path)
    assert load_verdicts(path) == data


def test_missing_verdict_file_is_empty(tmp_path):
    assert load_verdicts(tmp_path / "nope.json") == {}


# --- dropping bad questions ------------------------------------------------

def test_drop_bad_removes_only_flagged_questions(tmp_path):
    questions = tmp_path / "q.jsonl"
    questions.write_text("\n".join(json.dumps({"id": i, "question": "q"})
                                   for i in ("m1", "m2", "m3")) + "\n",
                         encoding="utf-8")
    out = tmp_path / "clean.jsonl"
    verdicts = {answer_key(DISPUTED): {"verdict": "bad-question", "id": "m1"}}

    assert drop_bad([DISPUTED], verdicts, questions, out) == 0
    kept = [json.loads(l) for l in out.open(encoding="utf-8")]
    assert [k["id"] for k in kept] == ["m2", "m3"]


def test_drop_bad_is_a_noop_without_flags(tmp_path):
    questions = tmp_path / "q.jsonl"
    questions.write_text(json.dumps({"id": "m1", "question": "q"}) + "\n",
                         encoding="utf-8")
    out = tmp_path / "clean.jsonl"
    drop_bad([DISPUTED], {}, questions, out)
    assert len(list(out.open(encoding="utf-8"))) == 1
