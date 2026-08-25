import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rag import BM25, Chunk  # noqa: E402
from rerank import Reranker  # noqa: E402


def chunk(cid, doc, text, title="T"):
    return Chunk(cid, doc, title, "Agency", text, 0)


class FakeReranker(Reranker):
    """Scores by keyword overlap - deterministic, no model download."""

    def _score(self, query, texts):
        terms = set(query.lower().split())
        return [len(terms & set(t.lower().split())) / (len(terms) or 1)
                for t in texts]


CHUNKS = [
    chunk("a#0", "a", "general background about maritime regulation matters"),
    chunk("b#0", "b", "petitions for judicial review must be filed by "
                      "September 28 2026"),
    chunk("c#0", "c", "the safety zone is enforced during fireworks"),
]


def build(chunks=CHUNKS):
    return BM25().index(chunks)


# --- reordering ------------------------------------------------------------

def test_reranker_can_promote_a_lower_ranked_passage():
    index = FakeReranker(build(), depth=10)
    hits = index.search("petitions for judicial review", k=1)
    assert hits[0][0].doc_id == "b"


def test_results_are_ordered_by_reranker_score():
    hits = FakeReranker(build(), depth=10).search("safety zone fireworks", k=3)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_k_limits_results():
    assert len(FakeReranker(build(), depth=10).search("regulation", k=2)) <= 2


def test_empty_candidate_list_returns_nothing():
    empty = FakeReranker(BM25().index([]), depth=10)
    assert empty.search("anything", k=5) == []


def test_query_matching_nothing_returns_nothing():
    assert FakeReranker(build(), depth=10).search("zzzz qqqq", k=5) == []


# --- candidate pool --------------------------------------------------------

def test_depth_controls_how_many_candidates_are_scored():
    seen = {}

    class Counting(FakeReranker):
        def _score(self, query, texts):
            seen["n"] = len(texts)
            return super()._score(query, texts)

    chunks = [chunk(f"d{i}#0", f"d{i}", f"regulation passage {i}")
              for i in range(20)]
    Counting(build(chunks), depth=5).search("regulation", k=3)
    assert seen["n"] == 5


def test_per_doc_cap_not_applied_before_reranking():
    """Capping first can discard the passage that would have won."""
    seen = {}

    class Counting(FakeReranker):
        def _score(self, query, texts):
            seen["n"] = len(texts)
            return super()._score(query, texts)

    chunks = [chunk(f"big#{i}", "big", f"regulation passage number {i}")
              for i in range(6)]
    Counting(build(chunks), depth=10).search("regulation passage", k=2,
                                             per_doc=1)
    assert seen["n"] == 6          # all six scored, despite per_doc=1


def test_per_doc_cap_applied_after_reranking():
    chunks = [chunk(f"big#{i}", "big", f"regulation passage {i}")
              for i in range(6)]
    chunks.append(chunk("other#0", "other", "regulation passage elsewhere"))
    hits = FakeReranker(build(chunks), depth=10).search("regulation passage",
                                                        k=3, per_doc=1)
    assert [c.doc_id for c, _ in hits].count("big") == 1


# --- wrapping --------------------------------------------------------------

def test_reranker_wraps_any_retriever_interface():
    class Stub:
        def search(self, query, k, per_doc=2):
            return [(CHUNKS[1], 1.0), (CHUNKS[0], 0.5)]

    hits = FakeReranker(Stub(), depth=10).search("judicial review", k=1)
    assert hits[0][0].doc_id == "b"


def test_reranker_cannot_recover_a_missing_document():
    """Recall is set by the first pass; reranking only reorders."""
    index = FakeReranker(build(CHUNKS[:1]), depth=10)
    hits = index.search("petitions for judicial review", k=3)
    assert all(c.doc_id != "b" for c, _ in hits)
