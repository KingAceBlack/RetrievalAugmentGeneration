import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from explain import find_duplicates, normalise_question  # noqa: E402


def q(qid, text, gold):
    return {"id": qid, "question": text, "gold_doc": gold, "type": "fact"}


PIPELINE = ("What is the docket number for the Transportation Department "
            "rulemaking on Pipeline Safety: Standards Update-ASTM ?")


def test_identical_questions_grouped():
    """The ablation case: the ASTM designator removed leaves twins."""
    questions = [q("q002", PIPELINE, "2026-15570"),
                 q("q047", PIPELINE, "2026-15580"),
                 q("q009", "What is the docket for the Olcott safety zone?",
                   "2026-15300")]
    groups = find_duplicates(questions)
    assert len(groups) == 1
    assert {x["id"] for x in groups[0]} == {"q002", "q047"}


def test_distinct_questions_not_grouped():
    questions = [q("a", "Docket for the Olcott safety zone?", "1"),
                 q("b", "Docket for the Aurora safety zone?", "2")]
    assert find_duplicates(questions) == []


def test_word_order_and_case_ignored():
    assert (normalise_question("Docket for the Olcott zone")
            == normalise_question("olcott ZONE the for docket"))


def test_punctuation_differences_ignored():
    questions = [q("a", "Pipeline Safety: Standards Update-ASTM ?", "1"),
                 q("b", "Pipeline Safety Standards Update ASTM", "2")]
    assert len(find_duplicates(questions)) == 1


def test_unanswerable_questions_excluded():
    """They share no gold document, so identical text is fine."""
    questions = [{"id": "u1", "question": "What penalty applies?",
                  "gold_doc": None, "type": "unanswerable"},
                 {"id": "u2", "question": "What penalty applies?",
                  "gold_doc": None, "type": "unanswerable"}]
    assert find_duplicates(questions) == []


def test_three_way_duplicate_grouped_together():
    questions = [q(f"q{i}", PIPELINE, f"doc{i}") for i in range(3)]
    groups = find_duplicates(questions)
    assert len(groups) == 1 and len(groups[0]) == 3


# --- dropping impossible questions -----------------------------------------

def test_drop_removes_every_member_of_a_duplicate_group(tmp_path):
    """All members go, not just the extras - none of them is answerable."""
    import json
    from explain import drop_duplicates

    questions = [q("q002", PIPELINE, "2026-15570"),
                 q("q047", PIPELINE, "2026-15580"),
                 q("q049", PIPELINE, "2026-15581"),
                 q("q009", "Docket for the Olcott safety zone?", "2026-15300")]
    out = tmp_path / "clean.jsonl"
    assert drop_duplicates(questions, out) == 0

    kept = [json.loads(l) for l in out.open(encoding="utf-8")]
    assert [k["id"] for k in kept] == ["q009"]


def test_drop_keeps_unanswerable_questions(tmp_path):
    """They have no gold document, so identical text is not a defect."""
    import json
    from explain import drop_duplicates

    questions = [q("q1", "Docket for the Olcott zone?", "1"),
                 {"id": "u1", "question": "What penalty applies?",
                  "gold_doc": None, "type": "unanswerable"},
                 {"id": "u2", "question": "What penalty applies?",
                  "gold_doc": None, "type": "unanswerable"}]
    out = tmp_path / "clean.jsonl"
    drop_duplicates(questions, out)
    kept = [json.loads(l) for l in out.open(encoding="utf-8")]
    assert {k["id"] for k in kept} == {"q1", "u1", "u2"}


def test_drop_is_a_noop_when_all_distinct(tmp_path):
    import json
    from explain import drop_duplicates

    questions = [q("a", "Docket for Olcott?", "1"),
                 q("b", "Docket for Aurora?", "2")]
    out = tmp_path / "clean.jsonl"
    drop_duplicates(questions, out)
    assert len(list(out.open(encoding="utf-8"))) == 2
