import datetime as dt
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fetch_fr import (  # noqa: E402
    agency_names, build_record, cfr_refs, clean_raw_text, daterange,
    is_correction, already_have,
)

WRAPPED = """<html>
<head>
<title>Federal Register, Volume 91 Issue 162 (Monday, August 24, 2026)</title>
</head>
<body><pre>
[Federal Register Volume 91, Number 162 (Monday, August 24, 2026)]
[Notices]
[Pages 54001-54003]
[FR Doc No: 2026-17250]

DEPARTMENT OF HEALTH AND HUMAN SERVICES

SUMMARY: The Department seeks public comment.

DATES: Comments must be received by October 23, 2026.
</pre></body>
</html>"""


# --- text extraction -------------------------------------------------------

def test_extracts_body_from_pre_block():
    text = clean_raw_text(WRAPPED)
    assert text.startswith("[Federal Register Volume 91")
    assert "Comments must be received by October 23, 2026." in text
    assert "<pre>" not in text and "<html>" not in text
    assert "Federal Register, Volume 91 Issue 162" not in text   # title dropped


def test_plain_text_passes_through():
    plain = "SUMMARY: A notice.\n\nDATES: Effective September 1, 2026."
    assert clean_raw_text(plain) == plain


def test_entities_decoded():
    assert clean_raw_text("<pre>A &amp; B &quot;x&quot;</pre>") == 'A & B "x"'


def test_blank_line_runs_collapsed():
    assert clean_raw_text("<pre>a\n\n\n\n\nb</pre>") == "a\n\nb"


# --- dates -----------------------------------------------------------------

def test_daterange_skips_weekends():
    days = list(daterange(dt.date(2026, 8, 21), dt.date(2026, 8, 24)))
    assert days == [dt.date(2026, 8, 21), dt.date(2026, 8, 24)]


def test_daterange_weekend_only_is_empty():
    assert list(daterange(dt.date(2026, 8, 22), dt.date(2026, 8, 23))) == []


# --- corrections -----------------------------------------------------------

def test_correction_detected_by_prefix():
    assert is_correction({"document_number": "C1-2026-16331"})
    assert is_correction({"document_number": "C2-2026-04516"})


def test_correction_detected_by_field():
    assert is_correction({"document_number": "2026-17250",
                          "correction_of": "2026-16000"})


def test_ordinary_document_is_not_a_correction():
    assert not is_correction({"document_number": "2026-17250"})
    assert not is_correction({"document_number": "2026-17250",
                              "correction_of": None})


# --- metadata derivation ---------------------------------------------------

def test_agency_names_prefers_name_and_dedupes():
    detail = {"agencies": [
        {"raw_name": "DEPARTMENT OF THE TREASURY", "name": "Treasury Department"},
        {"raw_name": "Internal Revenue Service", "name": "Internal Revenue Service"},
        {"raw_name": "Treasury Department", "name": "Treasury Department"},
    ]}
    assert agency_names(detail) == ["Treasury Department",
                                    "Internal Revenue Service"]


def test_agency_names_falls_back_to_raw_name():
    assert agency_names({"agencies": [{"raw_name": "Office of the Secretary"}]}) \
        == ["Office of the Secretary"]


def test_agency_names_empty():
    assert agency_names({}) == []


def test_cfr_refs_flattened():
    detail = {"cfr_references": [{"title": 40, "part": 52},
                                 {"title": 2, "part": 3474},
                                 {"title": 40, "part": 52}]}
    assert cfr_refs(detail) == ["40 CFR 52", "2 CFR 3474"]


def test_cfr_refs_without_part():
    assert cfr_refs({"cfr_references": [{"title": 21}]}) == ["21 CFR"]


# --- record building -------------------------------------------------------

def test_build_record_drops_empty_fields_and_counts_words():
    detail = {
        "document_number": "2026-17250",
        "title": "A Notice",
        "type": "Notice",
        "abstract": None,                 # dropped
        "docket_ids": [],                 # dropped
        "comments_close_on": "2026-10-23",
        "agencies": [{"name": "Health and Human Services Department"}],
        "cfr_references": [{"title": 42, "part": 100}],
    }
    rec = build_record(detail, "one two three four five")
    assert rec["document_number"] == "2026-17250"
    assert rec["comments_close_on"] == "2026-10-23"
    assert "abstract" not in rec and "docket_ids" not in rec
    assert rec["agency_names"] == ["Health and Human Services Department"]
    assert rec["cfr_references"] == ["42 CFR 100"]
    assert rec["word_count"] == 5
    assert rec["char_count"] == len("one two three four five")


# --- resume ----------------------------------------------------------------

def test_already_have_reads_document_numbers(tmp_path):
    p = tmp_path / "documents.jsonl"
    p.write_text("\n".join(json.dumps({"document_number": n})
                           for n in ("2026-1", "2026-2")) + "\n",
                 encoding="utf-8")
    assert already_have(p) == {"2026-1", "2026-2"}


def test_already_have_tolerates_a_truncated_final_line(tmp_path):
    """An interrupted run can leave a half-written line."""
    p = tmp_path / "documents.jsonl"
    p.write_text('{"document_number": "2026-1"}\n{"document_num',
                 encoding="utf-8")
    assert already_have(p) == {"2026-1"}


def test_already_have_missing_file(tmp_path):
    assert already_have(tmp_path / "nope.jsonl") == set()
