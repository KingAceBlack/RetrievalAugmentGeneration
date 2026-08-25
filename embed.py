#!/usr/bin/env python3
"""Dense and hybrid retrieval, to be compared against the BM25 baseline.

Three retrievers with the same interface, so the evaluation harness can swap
between them and change nothing else:

    bm25    keyword matching. Exact, literal, deaf to meaning.
    dense   embeddings. Matches meaning, fuzzy about identifiers.
    hybrid  both, fused. Each covers the other's blind spot.

Fusion uses Reciprocal Rank Fusion rather than adding the scores together.
BM25 scores are unbounded and corpus-dependent; cosine similarity sits between
-1 and 1. Adding them means the weighting drifts with the corpus and any
"tuning" is really tuning the scale mismatch. RRF discards the magnitudes and
uses only the ranks, so there is nothing to tune and nothing to drift.

Embeddings are cached to disk, keyed on the model name and a hash of the chunk
text. Change the chunker and the cache invalidates itself - otherwise you
would silently score new chunks against old vectors.

Usage:
    python embed.py --build                     # build the cache
    python embed.py --ask "when is the deadline" --retriever hybrid
    python embed.py --compare "petition for judicial review"
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys
from collections import defaultdict

import numpy as np

from rag import BM25, Chunk, build_index, load_corpus

CACHE = pathlib.Path("corpus/embeddings.npz")
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# bge v1.5 retrieval improved without instructions, but short queries against
# long passages can still benefit. Tested rather than assumed - see --compare.
QUERY_PREFIX = ""


def content_hash(chunks: list[Chunk]) -> str:
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(chunk.chunk_id.encode())
        h.update(chunk.text.encode())
    return h.hexdigest()[:16]


class DenseIndex:
    """Cosine similarity over embedded chunks."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None
        self._model = None

    def _embed(self, texts: list[str], batch: int = 64) -> np.ndarray:
        if self._model is None:
            from fastembed import TextEmbedding
            print(f"loading {self.model_name} ...", file=sys.stderr)
            self._model = TextEmbedding(model_name=self.model_name)

        out, done = [], 0
        for vec in self._model.embed(texts, batch_size=batch):
            out.append(vec)
            done += 1
            if done % 2000 == 0:
                print(f"  embedded {done}/{len(texts)}", file=sys.stderr,
                      flush=True)
        arr = np.asarray(out, dtype=np.float32)
        # normalise once so cosine similarity is a plain dot product
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-9, None)

    def build(self, chunks: list[Chunk], cache: pathlib.Path = CACHE,
              rebuild: bool = False) -> "DenseIndex":
        self.chunks = chunks
        digest = content_hash(chunks)

        if cache.exists() and not rebuild:
            data = np.load(cache, allow_pickle=False)
            if (str(data["model"]) == self.model_name
                    and str(data["digest"]) == digest
                    and data["vectors"].shape[0] == len(chunks)):
                self.vectors = data["vectors"]
                print(f"loaded {len(chunks)} cached vectors", file=sys.stderr)
                return self
            print("cache is stale (chunking or model changed) - rebuilding",
                  file=sys.stderr)

        print(f"embedding {len(chunks)} chunks - this takes a few minutes",
              file=sys.stderr)
        self.vectors = self._embed([c.text for c in chunks])
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, vectors=self.vectors,
                            model=np.array(self.model_name),
                            digest=np.array(digest))
        print(f"cached -> {cache}", file=sys.stderr)
        return self

    def search(self, query: str, k: int = 5,
               per_doc: int = 2) -> list[tuple[Chunk, float]]:
        if self.vectors is None:
            raise RuntimeError("build() first")
        q = self._embed([QUERY_PREFIX + query])[0]
        scores = self.vectors @ q

        out: list[tuple[Chunk, float]] = []
        seen: dict[str, int] = defaultdict(int)
        for i in np.argsort(-scores):
            chunk = self.chunks[i]
            if per_doc and seen[chunk.doc_id] >= per_doc:
                continue
            seen[chunk.doc_id] += 1
            out.append((chunk, float(scores[i])))
            if len(out) >= k:
                break
        return out


class HybridIndex:
    """Reciprocal Rank Fusion of BM25 and dense results."""

    def __init__(self, bm25: BM25, dense: DenseIndex, rrf_k: int = 60,
                 depth: int = 50):
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k
        self.depth = depth

    def search(self, query: str, k: int = 5,
               per_doc: int = 2) -> list[tuple[Chunk, float]]:
        # Fuse over a deeper pool than k, or a result ranked 6th by both
        # retrievers could never surface despite broad agreement.
        lists = [self.bm25.search(query, self.depth, per_doc=0),
                 self.dense.search(query, self.depth, per_doc=0)]

        fused: dict[str, float] = defaultdict(float)
        chunks: dict[str, Chunk] = {}
        for hits in lists:
            for rank, (chunk, _) in enumerate(hits, 1):
                fused[chunk.chunk_id] += 1.0 / (self.rrf_k + rank)
                chunks[chunk.chunk_id] = chunk

        ordered = sorted(fused.items(), key=lambda kv: -kv[1])
        out: list[tuple[Chunk, float]] = []
        seen: dict[str, int] = defaultdict(int)
        for chunk_id, score in ordered:
            chunk = chunks[chunk_id]
            if per_doc and seen[chunk.doc_id] >= per_doc:
                continue
            seen[chunk.doc_id] += 1
            out.append((chunk, score))
            if len(out) >= k:
                break
        return out


def make_retriever(kind: str, corpus: pathlib.Path,
                   model: str = DEFAULT_MODEL, rebuild: bool = False,
                   rerank_model: str | None = None,
                   rerank_depth: int = 50):
    """Build whichever retriever was asked for. One interface for all of them.

    Reranking wraps the choice rather than replacing it, so any first pass can
    be reordered by a cross-encoder.
    """
    bm25 = build_index(corpus)
    if kind == "bm25":
        index = bm25
    else:
        dense = DenseIndex(model).build(bm25.chunks, rebuild=rebuild)
        if kind == "dense":
            index = dense
        elif kind == "hybrid":
            index = HybridIndex(bm25, dense)
        else:
            raise SystemExit(f"unknown retriever: {kind}")

    if rerank_model:
        from rerank import Reranker
        index = Reranker(index, rerank_model, rerank_depth)
    return index


def compare(query: str, corpus: pathlib.Path, model: str, k: int = 5) -> int:
    """Show the same query through all three retrievers, side by side."""
    bm25 = build_index(corpus)
    dense = DenseIndex(model).build(bm25.chunks)
    hybrid = HybridIndex(bm25, dense)

    print(f"\nquery: {query}\n")
    for name, index in (("bm25", bm25), ("dense", dense), ("hybrid", hybrid)):
        print(f"{name}:")
        for n, (chunk, score) in enumerate(index.search(query, k), 1):
            print(f"  {n}. {score:7.3f}  {chunk.doc_id}  "
                  f"{chunk.title[:52]}")
        print()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=pathlib.Path,
                    default=pathlib.Path("corpus/documents.jsonl"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--retriever", default="hybrid",
                    choices=["bm25", "dense", "hybrid"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--ask")
    ap.add_argument("--compare", metavar="QUERY")
    args = ap.parse_args()

    if args.compare:
        return compare(args.compare, args.corpus, args.model, args.k)

    if args.build or args.rebuild:
        bm25 = build_index(args.corpus)
        DenseIndex(args.model).build(bm25.chunks, rebuild=args.rebuild)
        print("\nready - now run:\n"
              "  python evaluate.py --k 5 --retriever hybrid "
              "--questions eval/questions_all.jsonl --by-type")
        return 0

    if not args.ask:
        ap.print_help()
        return 1

    index = make_retriever(args.retriever, args.corpus, args.model)
    for n, (chunk, score) in enumerate(index.search(args.ask, args.k), 1):
        print(f"  {n}. {score:7.3f}  {chunk.doc_id}  {chunk.title[:52]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
