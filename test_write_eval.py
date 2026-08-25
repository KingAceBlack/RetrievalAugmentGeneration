import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from write_eval import (  # noqa: E402
    interesting_passages, leaked_words, merge, normalise, title_words,
)

TEXT = """AGENCY: Coast Guard, DHS.

SUMMARY: The Coast Guard is establishing a temporary safety zone.

DATES: This rule is effective September 5, 2026. Comments must be received
by October 23, 2026.

ADDRESSES: Docket Number USCG-2026-0861.

SUPPLEMENTARY INFORMATION: Background follows in ordinary prose without any
particular marker at the start of the line."""


# --- leak detection --------------------------------------------------------

def test_title_words_drop_boilerplate():
    words = title_words("Safety Zone; Lake Ontario, Olcott, NY")
    assert "olcott" in words and "ontario" in words
    assert "the" not in words and "for" not in words


def test_reused_title_words_flagged():
    leak = leaked_words("What is the docket for the Olcott safety zone?",
                        "Safety Zone; Lake Ontario, Olcott, NY")
    assert "olcott" in leak and "safety" in leak


def test_question_in_user_language_is_clean():
    """The question a real person asks shares no vocabulary with the title."""
    leak = leaked_words(
        "When do boaters near the Niagara County shoreline need to stay out "
        "of the water for the fireworks?",
        "Safety Zone; Lake Ontario, Olcott, NY")
    assert leak == []


def test_short_words_ignored():
    assert leaked_words("Is it in NY?", "Safety Zone; Olcott, NY") == []


# --- passage selection -----------------------------------------------------

def test_marker_lines_surfaced():
    lines = interesting_passages(TEXT)
    assert any(l.startswith("DATES:") for l in lines)
    assert any(l.startswith("ADDRESSES:") for l in lines)
    assert any(l.startswith("SUMMARY:") for l in lines)


def test_prose_without_markers_is_deprioritised():
    lines = interesting_passages(TEXT, limit=3)
    assert not any("Background follows" in l for l in lines)


def test_lines_with_years_included_when_markers_run_out():
    text = "Some prose here.\nA payment of $500 applies.\nMore prose."
    assert any("$500" in l for l in interesting_passages(text))


def test_limit_respected():
    assert len(interesting_passages(TEXT, limit=2)) == 2


# --- normalisation ---------------------------------------------------------

def test_normalise_strips_punctuation():
    assert normalise("USCG-2026/0861.") == "uscg 2026 0861"


def test_answer_verification_is_substring_after_normalising():
    assert normalise("October 23, 2026") in normalise(TEXT)
    assert normalise("October 24, 2026") not in normalise(TEXT)


# --- merging ---------------------------------------------------------------

def test_merge_combines_and_renumbers(tmp_path):
    gen = tmp_path / "gen.jsonl"
    man = tmp_path / "man.jsonl"
    out = tmp_path / "all.jsonl"
    gen.write_text(json.dumps({"id": "q001", "type": "fact",
                               "gold_doc": "1", "question": "a",
                               "answer": "x"}) + "\n", encoding="utf-8")
    man.write_text(json.dumps({"type": "manual", "gold_doc": "2",
                               "question": "b", "answer": "y"}) + "\n",
                   encoding="utf-8")
    assert merge(man, gen, out) == 0
    rows = [json.loads(l) for l in out.open(encoding="utf-8")]
    assert [r["id"] for r in rows] == ["m001", "m002"]
    assert {r["type"] for r in rows} == {"fact", "manual"}


def test_merge_tolerates_a_missing_file(tmp_path):
    man = tmp_path / "man.jsonl"
    out = tmp_path / "all.jsonl"
    man.write_text(json.dumps({"type": "manual", "gold_doc": "2",
                               "question": "b", "answer": "y"}) + "\n",
                   encoding="utf-8")
    assert merge(man, tmp_path / "missing.jsonl", out) == 0
    assert len(list(out.open(encoding="utf-8"))) == 1
