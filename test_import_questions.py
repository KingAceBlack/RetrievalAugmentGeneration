import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from import_questions import normalise, parse, verify  # noqa: E402


def doc(num, text, title="A Rule"):
    return {"document_number": num, "title": title, "text": text,
            "agency_names": ["Agency"]}


# --- parsing ---------------------------------------------------------------

def test_parses_three_line_records():
    text = ("DOC: 2026-15368\n"
            "Q: How long must GSA wait?\n"
            "A: 30 calendar days\n\n"
            "DOC: 2026-15377\n"
            "Q: When is the review deadline?\n"
            "A: September 28, 2026\n")
    out = parse(text)
    assert len(out) == 2
    assert out[0] == {"gold_doc": "2026-15368",
                      "question": "How long must GSA wait?",
                      "answer": "30 calendar days"}


def test_none_answer_becomes_a_refusal_test():
    out = parse("DOC: 2026-1\nQ: What penalty applies?\nA: NONE\n")
    assert out[0]["answer"] is None


def test_blank_input_yields_nothing():
    assert parse("") == []
    assert parse("just some prose\n") == []


# --- normalisation ---------------------------------------------------------

def test_phone_number_formats_match():
    assert normalise("(800) 552-6458") == normalise("800-552-6458")


def test_typographic_quotes_match_plain_ones():
    assert normalise("42\u00b021'41.22'' N") == normalise("42 21'41.22'' N")


def test_thousands_separators_preserved():
    assert normalise("139,175 acres") == "139 175 acres"


def test_cfr_citation_normalises():
    assert normalise("50 CFR 622.193(z)(1)(i)") == "50 cfr 622 193 z 1 i"


# --- verification ----------------------------------------------------------

def test_answer_present_is_accepted():
    docs = [doc("1", "GSA waits 30 calendar days after notifying Congress.")]
    records = parse("DOC: 1\nQ: How long is the wait?\nA: 30 calendar days\n")
    good, bad = verify(records, docs)
    assert len(good) == 1 and not bad
    assert good[0]["type"] == "manual"
    assert good[0]["corpus_occurrences"] == 1


def test_answer_absent_is_rejected():
    docs = [doc("1", "This document states nothing about waiting periods.")]
    records = parse("DOC: 1\nQ: How long is the wait?\nA: 30 calendar days\n")
    good, bad = verify(records, docs)
    assert not good and len(bad) == 1
    assert "not found" in bad[0]["problem"]


def test_unknown_document_is_rejected():
    records = parse("DOC: 9999\nQ: Anything?\nA: something\n")
    good, bad = verify(records, [doc("1", "text")])
    assert not good and bad[0]["problem"] == "document not in corpus"


def test_answer_shared_across_documents_is_counted():
    docs = [doc("1", "Effective September 28, 2026 for this rule."),
            doc("2", "Also effective September 28, 2026 elsewhere.")]
    records = parse("DOC: 1\nQ: When effective?\nA: September 28, 2026\n")
    good, _ = verify(records, docs)
    assert good[0]["corpus_occurrences"] == 2


def test_title_word_reuse_flagged():
    docs = [doc("1", "The ozone standard is Rule 3173.",
                title="Air Plan Approval; Ozone Standard")]
    records = parse("DOC: 1\nQ: Which ozone rule applies?\nA: Rule 3173\n")
    good, _ = verify(records, docs)
    assert "ozone" in good[0]["title_words_reused"]


def test_question_in_plain_language_is_clean():
    docs = [doc("1", "The smog limit is Rule 3173.",
                title="Air Plan Approval; Ozone Standard")]
    records = parse("DOC: 1\nQ: Which measure caps smog?\nA: Rule 3173\n")
    good, _ = verify(records, docs)
    assert good[0]["title_words_reused"] == []


def test_refusal_record_has_no_gold_document():
    docs = [doc("1", "Ordinary text.")]
    records = parse("DOC: 1\nQ: What penalty applies?\nA: NONE\n")
    good, bad = verify(records, docs)
    assert not bad
    assert good[0]["type"] == "unanswerable"
    assert good[0]["gold_doc"] is None
    assert good[0]["distractors"] == ["1"]


def test_boilerplate_is_stripped_before_matching():
    """The masthead is removed at index time, so it must be here too."""
    docs = [doc("1", "[Federal Register Volume 91]\n\nThe answer is 42 acres.")]
    records = parse("DOC: 1\nQ: How big?\nA: 42 acres\n")
    good, bad = verify(records, docs)
    assert good and not bad
