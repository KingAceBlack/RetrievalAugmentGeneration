import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from make_eval import (  # noqa: E402
    Verifier, build_discriminator_questions, build_fact_questions,
    build_unanswerable_questions, candidate_answers, date_forms, phrase,
    subject_of,
)


def doc(num, text, **kw):
    base = {
        "document_number": num,
        "title": kw.pop("title", f"Safety Zone; Location {num}"),
        "type": "Rule",
        "agency_names": kw.pop("agency_names", ["Coast Guard"]),
        "text": text,
        "word_count": len(text.split()),
    }
    base.update(kw)
    return base


# --- date surface forms ----------------------------------------------------

def test_date_written_out_not_iso():
    assert date_forms("2026-10-23") == ["October 23, 2026"]


def test_single_digit_day_has_no_leading_zero():
    """Windows strftime lacks %-d, so the day is formatted by hand."""
    assert date_forms("2026-09-05") == ["September 5, 2026"]


def test_invalid_date_yields_nothing():
    assert date_forms("") == []
    assert date_forms("not-a-date") == []
    assert date_forms(None) == []


# --- candidate selection ---------------------------------------------------

def test_dates_come_before_dockets():
    d = doc("1", "t", comments_close_on="2026-10-23",
            docket_ids=["USCG-2026-0001"])
    fields = [f for f, _ in candidate_answers(d)]
    assert fields.index("comments_close_on") < fields.index("docket_ids")


def test_cfr_references_never_offered():
    """They are shared by hundreds of documents and can never be unique."""
    d = doc("1", "t", cfr_references=["33 CFR 165"],
            docket_ids=["USCG-2026-0001"])
    assert "cfr_references" not in [f for f, _ in candidate_answers(d)]


def test_no_candidates_when_metadata_is_bare():
    assert candidate_answers(doc("1", "t")) == []


# --- verification ----------------------------------------------------------

def test_unique_answer_accepted():
    corpus = [doc("1", "Comments due by October 23, 2026 in this docket."),
              doc("2", "An unrelated rule with different dates entirely.")]
    ok, found = Verifier(corpus).check("October 23, 2026", "1")
    assert ok and found == ["1"]


def test_shared_answer_string_is_accepted():
    """Dates repeat constantly in a one-month corpus.

    Requiring a globally unique answer discards every date question and leaves
    only docket numbers, which test keyword matching rather than retrieval.
    """
    corpus = [doc("1", "Comments due by October 23, 2026."),
              doc("2", "Also due October 23, 2026 for a separate matter.")]
    ok, found = Verifier(corpus).check("October 23, 2026", "1")
    assert ok and set(found) == {"1", "2"}


def test_corpus_wide_boilerplate_rejected():
    """A string in almost every document is not a usable answer."""
    corpus = [doc(str(i), "Comments due by October 23, 2026.")
              for i in range(30)]
    ok, _ = Verifier(corpus).check("October 23, 2026", "1",
                                   max_occurrences=10)
    assert not ok


def test_absent_from_gold_still_rejected():
    """Presence in the gold document remains mandatory."""
    corpus = [doc("1", "No dates appear in this body text."),
              doc("2", "Comments due by October 23, 2026.")]
    ok, _ = Verifier(corpus).check("October 23, 2026", "1")
    assert not ok


def test_answer_absent_from_its_own_document_rejected():
    """Metadata can state a date the prose never mentions."""
    corpus = [doc("1", "This rule contains no dates in its body text.")]
    ok, found = Verifier(corpus).check("October 23, 2026", "1")
    assert not ok and found == []


def test_verification_is_case_insensitive():
    corpus = [doc("1", "docket number uscg-2026-0001 refers.")]
    ok, _ = Verifier(corpus).check("USCG-2026-0001", "1")
    assert ok


# --- phrasing --------------------------------------------------------------

def test_subject_uses_the_distinguishing_part_of_the_title():
    d = doc("1", "t", title="Safety Zone; Lake Ontario, Olcott, NY")
    assert subject_of(d) == "Safety Zone for Lake Ontario, Olcott, NY"


def test_subject_handles_title_without_semicolon():
    d = doc("1", "t", title="Air Plan Approval")
    assert subject_of(d) == "Air Plan Approval"


def test_subject_strips_trailing_punctuation():
    d = doc("1", "t", title="Safety Zone; Ohio River, Aurora, IN;")
    assert not subject_of(d).endswith(";")


def test_question_names_agency_and_subject_but_not_document_number():
    d = doc("2026-15300", "t", title="Safety Zone; Lake Ontario, Olcott, NY")
    q = phrase(d, "comments_close_on")
    assert "Coast Guard" in q and "Olcott" in q
    assert "2026-15300" not in q     # otherwise it is a trivial string match


# --- builders --------------------------------------------------------------

def test_fact_questions_only_use_verified_answers():
    corpus = [
        doc("1", "Comments must be received by October 23, 2026.",
            comments_close_on="2026-10-23"),
        doc("2", "This document never states its own effective date.",
            effective_on="2026-11-01"),        # absent from text -> rejected
    ]
    out = build_fact_questions(corpus, Verifier(corpus), 10, random.Random(0))
    assert len(out) == 1
    assert out[0]["gold_doc"] == "1"
    assert out[0]["corpus_occurrences"] == 1


def test_fact_questions_respect_the_limit():
    corpus = [doc(str(i), f"Comments must be received by October {i}, 2026.",
                  comments_close_on=f"2026-10-{i:02d}") for i in range(1, 9)]
    out = build_fact_questions(corpus, Verifier(corpus), 3, random.Random(0))
    assert len(out) == 3


def test_fact_questions_one_per_document():
    corpus = [doc("1", "Comments by October 23, 2026. Effective "
                       "November 1, 2026. Docket USCG-2026-0001.",
                  comments_close_on="2026-10-23", effective_on="2026-11-01",
                  docket_ids=["USCG-2026-0001"])]
    out = build_fact_questions(corpus, Verifier(corpus), 10, random.Random(0))
    assert len(out) == 1


def test_discriminator_records_lookalike_siblings():
    corpus = [
        doc("1", "Safety zone established. Comments by October 3, 2026.",
            comments_close_on="2026-10-03"),
        doc("2", "Safety zone established. Comments by October 4, 2026.",
            comments_close_on="2026-10-04"),
        doc("3", "Safety zone established. Comments by October 5, 2026.",
            comments_close_on="2026-10-05"),
    ]
    out = build_discriminator_questions(
        corpus, [[0, 1, 2]], Verifier(corpus), 5, random.Random(0))
    assert len(out) == 1
    q = out[0]
    assert q["type"] == "discriminator"
    assert len(q["distractors"]) == 2
    assert q["gold_doc"] not in q["distractors"]


def test_one_discriminator_per_family():
    """No single family should dominate the evaluation set."""
    corpus = [doc(str(i), f"Zone rule. Comments by October {i}, 2026.",
                  comments_close_on=f"2026-10-{i:02d}") for i in range(1, 7)]
    clusters = [[0, 1, 2], [3, 4, 5]]
    out = build_discriminator_questions(
        corpus, clusters, Verifier(corpus), 10, random.Random(0))
    assert len(out) == 2


def test_unanswerable_have_no_gold_document():
    corpus = [doc(str(i), f"body {i} " * 30) for i in range(4)]
    out = build_unanswerable_questions(corpus, Verifier(corpus), 3,
                                       random.Random(0))
    assert out and all(q["answer"] is None for q in out)
    assert all(q["gold_doc"] is None for q in out)
    assert all(q["type"] == "unanswerable" for q in out)


def test_fabricated_subject_included_when_truly_absent():
    corpus = [doc(str(i), f"ordinary rule text {i} " * 30) for i in range(3)]
    out = build_unanswerable_questions(corpus, Verifier(corpus), 2,
                                       random.Random(0))
    assert any("drone parcel delivery" in q["question"] for q in out)


# --- ordering: discriminators must not be crowded out ----------------------

def test_fact_questions_can_exclude_documents():
    corpus = [doc(str(i), f"Effective October {i}, 2026 for this rule.",
                  effective_on=f"2026-10-{i:02d}") for i in range(1, 5)]
    out = build_fact_questions(corpus, Verifier(corpus), 10, random.Random(0),
                               exclude={"2", "3"})
    assert {q["gold_doc"] for q in out} == {"1", "4"}


def test_discriminator_survives_dedupe_against_a_fact_question():
    """Both builders phrase questions identically for the same document.

    If facts are generated first, the discriminator is dropped as a duplicate
    and the most informative question in the set disappears silently.
    """
    corpus = [
        doc("1", "Zone rule. Effective October 1, 2026.",
            effective_on="2026-10-01"),
        doc("2", "Zone rule. Effective October 2, 2026.",
            effective_on="2026-10-02"),
    ]
    verifier = Verifier(corpus)
    rng = random.Random(0)

    disc = build_discriminator_questions(corpus, [[0, 1]], verifier, 5, rng)
    assert disc, "precondition: a discriminator is available"
    used = {q["gold_doc"] for q in disc}
    facts = build_fact_questions(corpus, verifier, 5, rng, exclude=used)

    phrasings = {q["question"] for q in disc}
    assert not (phrasings & {q["question"] for q in facts})
    assert used & {q["gold_doc"] for q in facts} == set()


# --- field balance ---------------------------------------------------------

def test_field_cap_spreads_questions_across_fields():
    """Docket numbers are easy; they must not swamp the set."""
    corpus = []
    for i in range(1, 11):
        corpus.append(doc(str(i),
                          f"Effective October {i}, 2026. "
                          f"Docket No. USCG-2026-{i:04d}.",
                          effective_on=f"2026-10-{i:02d}",
                          docket_ids=[f"USCG-2026-{i:04d}"]))
    out = build_fact_questions(corpus, Verifier(corpus), 10, random.Random(0),
                               field_cap=0.4)
    from collections import Counter
    fields = Counter(q["answer_field"] for q in out)
    assert len(fields) >= 2, fields
    assert max(fields.values()) <= 6


def test_second_pass_fills_when_only_one_field_exists():
    """The cap is a preference for variety, not a reason to under-deliver."""
    corpus = [doc(str(i), f"Comments must be received by October {i}, 2026.",
                  comments_close_on=f"2026-10-{i:02d}") for i in range(1, 9)]
    out = build_fact_questions(corpus, Verifier(corpus), 3, random.Random(0),
                               field_cap=0.4)
    assert len(out) == 3
    assert {q["answer_field"] for q in out} == {"comments_close_on"}
