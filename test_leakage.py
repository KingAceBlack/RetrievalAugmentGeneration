import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from leakage import ablate, analyse, document_frequency  # noqa: E402


def doc(num, text, title=""):
    return {"document_number": num, "title": title, "text": text}


CORPUS = [
    doc("A", "Safety zone on the navigable waters at Olcott during fireworks.",
        "Safety Zone; Lake Ontario, Olcott, NY"),
    doc("B", "Safety zone on the navigable waters at Aurora during fireworks.",
        "Safety Zone; Ohio River, Aurora, IN"),
    doc("C", "A rule about fisheries management and quota allocation.",
        "Fisheries Rule"),
]


def test_document_frequency_counts_documents_not_occurrences():
    df = document_frequency(CORPUS)
    assert df["safety"] == {"A", "B"}
    assert df["olcott"] == {"A"}
    assert df["fisheries"] == {"C"}


def test_title_terms_are_included():
    df = document_frequency(CORPUS)
    assert "ontario" in df and df["ontario"] == {"A"}


def test_fingerprint_term_detected():
    df = document_frequency(CORPUS)
    q = {"id": "q1", "type": "fact", "gold_doc": "A",
         "question": "What is the docket number for the Safety Zone for "
                     "Lake Ontario, Olcott, NY?"}
    row = analyse(q, df, max_df=3)
    assert "olcott" in row["fingerprint_terms"]
    assert row["rarest_df"] == 1


def test_shared_terms_are_not_fingerprints():
    df = document_frequency(CORPUS)
    q = {"id": "q2", "type": "fact", "gold_doc": "A",
         "question": "Which safety zone applies during fireworks?"}
    row = analyse(q, df, max_df=1)
    assert row["fingerprint_terms"] == []


def test_stopwords_ignored():
    df = document_frequency(CORPUS)
    q = {"id": "q3", "type": "fact", "gold_doc": "C",
         "question": "What is the effective date of the rule?"}
    row = analyse(q, df, max_df=3)
    assert "date" not in row["rare_terms"]
    assert "rule" not in row["rare_terms"]


def test_digits_ignored():
    df = document_frequency([doc("A", "Effective September 5 2026 for zone.")])
    q = {"id": "q4", "type": "fact", "gold_doc": "A",
         "question": "Is it effective in 2026?"}
    assert "2026" not in analyse(q, df, max_df=3)["rare_terms"]


def test_unanswerable_questions_have_no_gold_document():
    df = document_frequency(CORPUS)
    q = {"id": "q5", "type": "unanswerable", "gold_doc": None,
         "question": "What penalty applies?"}
    row = analyse(q, df, max_df=3)
    assert row["fingerprint_terms"] == [] and row["rarest_df"] is None


# --- ablation --------------------------------------------------------------

def test_ablate_removes_the_fingerprint():
    q = {"question": "What is the docket number for the Safety Zone for "
                     "Lake Ontario, Olcott, NY?"}
    out = ablate(q, ["olcott"])
    assert "Olcott" not in out and "olcott" not in out
    assert "Safety Zone" in out


def test_ablate_is_case_insensitive_and_whole_word():
    q = {"question": "The Aurora rule and the aurora borealis reference."}
    out = ablate(q, ["aurora"])
    assert "Aurora" not in out and "aurora" not in out
    assert "borealis" in out


def test_ablate_tidies_leftover_punctuation():
    q = {"question": "Rule for Lake Ontario, Olcott, NY?"}
    out = ablate(q, ["olcott"])
    assert ", ," not in out
    assert "  " not in out


def test_ablate_without_terms_is_a_noop():
    q = {"question": "What is the effective date?"}
    assert ablate(q, []) == "What is the effective date?"


# --- degeneracy guard ------------------------------------------------------

def test_stub_question_flagged_degenerate():
    from leakage import is_degenerate
    assert is_degenerate("On what date does the Department rule on Rule "
                         "take effect?")


def test_meaningful_question_not_degenerate():
    from leakage import is_degenerate
    assert not is_degenerate("What is the docket number for the Coast Guard "
                             "safety zone on the navigable waters of Lake "
                             "Ontario during a fireworks display?")


def test_degeneracy_ignores_stopwords_and_digits():
    from leakage import is_degenerate
    # only "zone" survives as content
    assert is_degenerate("What is the effective date of the rule on zone 2026?")
