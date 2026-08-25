import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from inspect_corpus import (  # noqa: E402
    UnionFind, build_dev, exact_key, jaccard, load,
    near_duplicate_clusters, norm_text, sketch,
)


def doc(num, text, words=None, agency="Agency A", typ="Rule"):
    return {
        "document_number": num,
        "title": f"Title {num}",
        "type": typ,
        "agency_names": [agency],
        "text": text,
        "word_count": words if words is not None else len(text.split()),
    }


def airworthiness(model, docket):
    """FAA directives differ only in a few tokens - the hard retrieval case."""
    return (
        "AGENCY: Federal Aviation Administration, DOT. ACTION: Final rule. "
        "SUMMARY: We are adopting a new airworthiness directive for certain "
        f"{model} airplanes. This AD was prompted by reports of cracking in "
        "the fuselage skin. We are issuing this AD to address the unsafe "
        f"condition on these products. DATES: Docket No. {docket}. "
        "Comments must be received by the closing date. "
        "SUPPLEMENTARY INFORMATION: The FAA proposed to amend part 39 of "
        "Title 14 of the Code of Federal Regulations by adding the directive."
    )


# --- normalisation ---------------------------------------------------------

def test_norm_text_collapses_whitespace_and_case():
    assert norm_text("  A  B\n\nC  ") == "a b c"


def test_exact_key_ignores_whitespace_and_case():
    assert exact_key("Hello   World") == exact_key("hello world")


def test_exact_key_differs_on_content():
    assert exact_key("hello world") != exact_key("hello worlds")


# --- sketching and similarity ----------------------------------------------

def test_identical_text_has_jaccard_one():
    t = airworthiness("Boeing 737", "FAA-2026-1234")
    assert jaccard(sketch(t), sketch(t)) == 1.0


def test_unrelated_text_has_low_jaccard():
    a = sketch(airworthiness("Boeing 737", "FAA-2026-1234"))
    b = sketch("The Secretary of Education proposes to amend the general "
               "administrative regulations governing federal student aid "
               "programs and cost principles for grant awards." * 5)
    assert jaccard(a, b) < 0.1


def test_near_duplicates_score_high_but_not_one():
    a = sketch(airworthiness("Boeing 737", "FAA-2026-1234"))
    b = sketch(airworthiness("Airbus A320", "FAA-2026-9999"))
    score = jaccard(a, b)
    assert 0.6 < score < 1.0, score


def test_sketch_handles_text_shorter_than_shingle():
    assert len(sketch("two words")) >= 1


def test_sketch_is_deterministic():
    t = airworthiness("Boeing 737", "FAA-2026-1234")
    assert sketch(t) == sketch(t)


# --- union-find ------------------------------------------------------------

def test_union_find_groups_transitively():
    uf = UnionFind(4)
    uf.union(0, 1)
    uf.union(1, 2)
    assert uf.find(0) == uf.find(2)
    assert uf.find(3) != uf.find(0)


# --- clustering ------------------------------------------------------------

def test_clusters_group_near_duplicates_only():
    rows = [
        doc("1", airworthiness("Boeing 737", "FAA-2026-0001")),
        doc("2", airworthiness("Airbus A320", "FAA-2026-0002")),
        doc("3", airworthiness("Cessna 172", "FAA-2026-0003")),
        doc("4", "An entirely separate rule about Medicare hospital "
                 "inpatient prospective payment system rates and wage "
                 "index adjustments for the fiscal year." * 6),
    ]
    clusters = near_duplicate_clusters(rows, threshold=0.6)
    assert len(clusters) == 1
    assert clusters[0] == [0, 1, 2]


def test_no_clusters_when_all_distinct():
    rows = [doc(str(i), f"unique subject {i} " * 60) for i in range(4)]
    assert near_duplicate_clusters(rows, threshold=0.8) == []


def test_clusters_sorted_largest_first():
    rows = [doc(str(i), airworthiness(f"Model {i}", f"D-{i}"))
            for i in range(3)]
    rows += [doc("x", "alpha beta gamma delta epsilon " * 40),
             doc("y", "alpha beta gamma delta epsilon " * 40)]
    clusters = near_duplicate_clusters(rows, threshold=0.6)
    assert [len(c) for c in clusters] == sorted(
        [len(c) for c in clusters], reverse=True)


# --- dev slice -------------------------------------------------------------

def test_dev_slice_excludes_oversized_documents():
    rows = [doc("big", "x " * 10, words=900_000)]
    rows += [doc(str(i), f"body {i} " * 50, agency=f"A{i}") for i in range(5)]
    picked = build_dev(rows, [], size=10, max_words=50_000)
    assert "big" not in {r["document_number"] for r in picked}
    assert len(picked) == 5


def test_dev_slice_seeds_with_a_near_duplicate_cluster():
    rows = [doc(str(i), airworthiness(f"Model {i}", f"D-{i}"), agency="FAA")
            for i in range(4)]
    rows += [doc(f"o{i}", f"other subject {i} " * 50, agency=f"A{i}")
             for i in range(6)]
    clusters = near_duplicate_clusters(rows, threshold=0.6)
    picked = build_dev(rows, clusters, size=8, max_words=50_000)
    nums = {r["document_number"] for r in picked}
    assert len({"0", "1", "2", "3"} & nums) >= 3   # the hard case is present


def test_dev_slice_spreads_across_agencies():
    rows = []
    for agency, n in (("Big", 30), ("Mid", 5), ("Small", 2)):
        for i in range(n):
            rows.append(doc(f"{agency}-{i}", f"text {agency} {i} " * 50,
                            agency=agency))
    picked = build_dev(rows, [], size=9, max_words=50_000)
    seen = {r["agency_names"][0] for r in picked}
    assert seen == {"Big", "Mid", "Small"}


def test_dev_slice_stops_when_pool_exhausted():
    rows = [doc(str(i), f"text {i} " * 50) for i in range(3)]
    assert len(build_dev(rows, [], size=50, max_words=50_000)) == 3


def test_dev_slice_has_no_repeats():
    rows = [doc(str(i), f"text {i} " * 50, agency=f"A{i % 3}")
            for i in range(20)]
    picked = build_dev(rows, [], size=15, max_words=50_000)
    nums = [r["document_number"] for r in picked]
    assert len(nums) == len(set(nums))


# --- loading ---------------------------------------------------------------

def test_load_reads_jsonl(tmp_path):
    p = tmp_path / "documents.jsonl"
    p.write_text(json.dumps(doc("1", "hello world")) + "\n\n", encoding="utf-8")
    rows = load(p)
    assert len(rows) == 1 and rows[0]["document_number"] == "1"


def test_load_missing_file_exits(tmp_path):
    import pytest
    with pytest.raises(SystemExit):
        load(tmp_path / "nope.jsonl")


# --- similarity distribution -----------------------------------------------

def test_similarity_pass_returns_nearest_scores():
    from inspect_corpus import similarity_pass
    rows = [
        doc("1", airworthiness("Boeing 737", "D-1")),
        doc("2", airworthiness("Airbus A320", "D-2")),
        doc("3", "wholly unrelated subject matter about grant cost "
                 "principles and audit requirements " * 30),
    ]
    clusters, nearest = similarity_pass(rows, threshold=0.6)
    assert len(nearest) == 3
    assert nearest[0] > 0.6 and nearest[1] > 0.6      # the pair find each other
    assert nearest[2] < 0.2                            # the loner does not
    assert clusters == [[0, 1]]


def test_similarity_pass_single_document():
    from inspect_corpus import similarity_pass
    clusters, nearest = similarity_pass([doc("1", "text " * 50)], 0.6)
    assert clusters == [] and nearest == [0.0]


def test_dev_slice_seeds_from_a_pair():
    """A cluster of exactly two should still seed the slice."""
    rows = [doc("a", airworthiness("Boeing 737", "D-1"), agency="FAA"),
            doc("b", airworthiness("Airbus A320", "D-2"), agency="FAA")]
    rows += [doc(f"o{i}", f"other {i} " * 50, agency=f"A{i}") for i in range(5)]
    clusters = near_duplicate_clusters(rows, threshold=0.6)
    picked = build_dev(rows, clusters, size=5, max_words=50_000)
    assert {"a", "b"} <= {r["document_number"] for r in picked}


def test_dev_slice_seeds_from_several_clusters():
    """Seeding from one family only would leave the slice easy elsewhere."""
    templates = {
        "zone": "SUMMARY: The Coast Guard is establishing a temporary safety "
                "zone on navigable waters near {} during a fireworks display. "
                "Entry is prohibited unless authorized by the Captain. ",
        "pipe": "SUMMARY: PHMSA is incorporating by reference the updated "
                "ASTM standard {} governing steel pipe used in gas pipeline "
                "systems and revising the referenced edition. ",
        "space": "SUMMARY: This action establishes Class E airspace extending "
                 "upward from 700 feet above the surface at {} airport to "
                 "support instrument flight rule operations. ",
    }
    rows = []
    for fam, body in templates.items():
        for i in range(4):
            rows.append(doc(f"{fam}-{i}", body.format(f"location {i}") * 20,
                            agency=f"{fam.title()} Dept"))
    rows += [doc(f"o{i}", f"entirely unrelated subject {i} " * 60,
                 agency=f"O{i}") for i in range(6)]

    clusters = near_duplicate_clusters(rows, threshold=0.5)
    assert len(clusters) >= 2, "test corpus should hold distinct families"

    picked = build_dev(rows, clusters, size=18, max_words=50_000,
                       per_cluster=3)
    families = {r["document_number"].split("-")[0] for r in picked
                if "-" in r["document_number"]}
    assert len(families & set(templates)) >= 2


def test_per_cluster_limits_seeding():
    """The seed is capped; round-robin may still add more from that agency."""
    rows = [doc(str(i), airworthiness(f"Model {i}", f"D-{i}"), agency="FAA")
            for i in range(8)]
    rows += [doc(f"o{i}", f"other {i} " * 60, agency=f"A{i}") for i in range(6)]
    clusters = near_duplicate_clusters(rows, threshold=0.6)
    assert clusters and len(clusters[0]) == 8

    picked = build_dev(rows, clusters, size=3, max_words=50_000,
                       per_cluster=2)
    # the seed sits at the head of the slice
    assert [r["document_number"] for r in picked[:2]] == ["0", "1"]
    assert len(picked) == 3
