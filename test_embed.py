import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from embed import DenseIndex, HybridIndex, content_hash  # noqa: E402
from rag import BM25, Chunk  # noqa: E402


def chunk(cid, doc, text, title="T"):
    return Chunk(cid, doc, title, "Agency", text, 0)


CHUNKS = [
    chunk("a#0", "a", "The Coast Guard establishes a safety zone at Olcott."),
    chunk("b#0", "b", "PHMSA incorporates ASTM standard A53 for steel pipe."),
    chunk("c#0", "c", "Petitions for judicial review are due September 28."),
]


class FakeDense(DenseIndex):
    """Deterministic stand-in vectors, so tests need no model download."""

    def _embed(self, texts, batch=64):
        rows = []
        for t in texts:
            h = abs(hash(t.lower().strip())) % 997
            v = np.array([h % 7, h % 11, h % 13], dtype=np.float32) + 1.0
            rows.append(v / np.linalg.norm(v))
        return np.asarray(rows, dtype=np.float32)


# --- cache keying ----------------------------------------------------------

def test_hash_changes_when_chunk_text_changes():
    other = list(CHUNKS[:2]) + [chunk("c#0", "c", "different text entirely")]
    assert content_hash(CHUNKS) != content_hash(other)


def test_hash_changes_when_chunk_ids_change():
    other = [chunk("z#9", "a", CHUNKS[0].text)] + CHUNKS[1:]
    assert content_hash(CHUNKS) != content_hash(other)


def test_hash_is_stable_for_identical_input():
    assert content_hash(CHUNKS) == content_hash(list(CHUNKS))


def test_cache_is_reused_when_nothing_changed(tmp_path):
    cache = tmp_path / "e.npz"
    first = FakeDense().build(CHUNKS, cache=cache)
    assert cache.exists()
    second = FakeDense().build(CHUNKS, cache=cache)
    assert np.allclose(first.vectors, second.vectors)


def test_cache_invalidates_when_chunks_change(tmp_path):
    cache = tmp_path / "e.npz"
    FakeDense().build(CHUNKS, cache=cache)
    changed = CHUNKS[:2] + [chunk("c#0", "c", "completely new content here")]
    rebuilt = FakeDense().build(changed, cache=cache)
    assert rebuilt.vectors.shape[0] == 3


def test_cache_invalidates_when_model_changes(tmp_path):
    cache = tmp_path / "e.npz"
    FakeDense("model-one").build(CHUNKS, cache=cache)
    other = FakeDense("model-two").build(CHUNKS, cache=cache)
    assert str(np.load(cache)["model"]) == "model-two"
    assert other.vectors is not None


# --- dense search ----------------------------------------------------------

def test_vectors_are_normalised(tmp_path):
    index = FakeDense().build(CHUNKS, cache=tmp_path / "e.npz")
    norms = np.linalg.norm(index.vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_identical_text_scores_highest(tmp_path):
    index = FakeDense().build(CHUNKS, cache=tmp_path / "e.npz")
    hits = index.search(CHUNKS[1].text, k=1)
    assert hits[0][0].doc_id == "b"


def test_per_doc_cap_applies(tmp_path):
    chunks = [chunk(f"big#{i}", "big", f"passage {i} about safety zones")
              for i in range(6)]
    chunks.append(chunk("other#0", "other", "a different subject entirely"))
    index = FakeDense().build(chunks, cache=tmp_path / "e.npz")
    hits = index.search("safety zones", k=4, per_doc=2)
    assert [c.doc_id for c, _ in hits].count("big") <= 2


def test_search_before_build_raises():
    import pytest
    with pytest.raises(RuntimeError):
        DenseIndex().search("anything")


# --- hybrid fusion ---------------------------------------------------------

def build_hybrid(tmp_path, chunks=CHUNKS):
    bm25 = BM25().index(chunks)
    dense = FakeDense().build(chunks, cache=tmp_path / "e.npz")
    return HybridIndex(bm25, dense)


def test_hybrid_returns_k_results(tmp_path):
    hits = build_hybrid(tmp_path).search("safety zone Olcott", k=2)
    assert len(hits) == 2


def test_hybrid_surfaces_a_keyword_only_match(tmp_path):
    """An exact identifier must survive fusion even if dense misses it."""
    hits = build_hybrid(tmp_path).search("ASTM A53 steel pipe", k=3)
    assert "b" in [c.doc_id for c, _ in hits]


def test_rrf_rewards_agreement_between_retrievers(tmp_path):
    hybrid = build_hybrid(tmp_path)
    hits = hybrid.search(CHUNKS[0].text, k=3)
    assert hits[0][0].doc_id == "a"


def test_rrf_scores_are_small_and_descending(tmp_path):
    hits = build_hybrid(tmp_path).search("safety zone", k=3)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(0 < s < 1 for s in scores)


def test_hybrid_ignores_raw_score_magnitudes(tmp_path):
    """BM25 scores are unbounded; cosine is bounded. Only ranks are used."""
    hybrid = build_hybrid(tmp_path)
    hits = hybrid.search("judicial review petitions", k=3)
    assert all(s < 0.1 for _, s in hits)      # RRF range, not BM25 range


def test_hybrid_per_doc_cap(tmp_path):
    chunks = [chunk(f"big#{i}", "big", f"safety zone passage {i}")
              for i in range(6)]
    chunks.append(chunk("other#0", "other", "safety zone somewhere else"))
    hybrid = build_hybrid(tmp_path, chunks)
    hits = hybrid.search("safety zone", k=4, per_doc=1)
    assert [c.doc_id for c, _ in hits].count("big") <= 1
