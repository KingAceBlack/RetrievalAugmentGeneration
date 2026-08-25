#!/usr/bin/env python3
"""Rerank retrieved passages with a cross-encoder.

The difference from embeddings, which is the whole reason this helps:

  An embedding model reads the question and the passage *separately*, turns
  each into numbers, and compares them. It never sees them together, so it
  cannot notice that this passage answers this question - only that they are
  about similar things. That is why it confused A53 with A372.

  A cross-encoder reads the question and the passage *as one input* and scores
  the pair directly. Far more accurate, and far slower - too slow to run over
  30,000 chunks, but fine over the 50 a first-pass retriever hands it.

So this is a second stage, never a replacement: retrieve broadly and cheaply,
then reorder precisely. It attacks ranking rather than recall - if the right
document is not in the candidates, no reranker can rescue it.

There is no cache. Scores depend on the question, so every query pays the
cost. Budget for that in production; it is the main reason to keep the
candidate pool at 50 rather than 500.

Usage:
    python rerank.py --ask "what is the deadline to file for judicial review"
    python rerank.py --compare "when can someone challenge this in court"
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import defaultdict

from rag import Chunk, build_index

DEFAULT_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
DEFAULT_DEPTH = 50


class Reranker:
    """Wraps any retriever with the same search() interface."""

    def __init__(self, base, model_name: str = DEFAULT_MODEL,
                 depth: int = DEFAULT_DEPTH):
        self.base = base
        self.model_name = model_name
        self.depth = depth
        self._model = None

    def _score(self, query: str, texts: list[str]) -> list[float]:
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            print(f"loading reranker {self.model_name} ...", file=sys.stderr)
            self._model = TextCrossEncoder(model_name=self.model_name)
        return list(self._model.rerank(query, texts))

    def search(self, query: str, k: int = 5,
               per_doc: int = 2) -> list[tuple[Chunk, float]]:
        # Pull candidates without the per-document cap: capping before
        # reranking can discard the passage that would have won, because the
        # first pass ranked a weaker passage from the same document higher.
        candidates = self.base.search(query, self.depth, per_doc=0)
        if not candidates:
            return []

        scores = self._score(query, [c.text for c, _ in candidates])
        ranked = sorted(zip((c for c, _ in candidates), scores),
                        key=lambda pair: -pair[1])

        out: list[tuple[Chunk, float]] = []
        seen: dict[str, int] = defaultdict(int)
        for chunk, score in ranked:
            if per_doc and seen[chunk.doc_id] >= per_doc:
                continue
            seen[chunk.doc_id] += 1
            out.append((chunk, float(score)))
            if len(out) >= k:
                break
        return out


def compare(query: str, corpus: pathlib.Path, model: str, depth: int,
            k: int = 5) -> int:
    """Show the same query before and after reranking."""
    base = build_index(corpus)
    reranked = Reranker(base, model, depth)

    print(f"\nquery: {query}\n")
    for name, index in (("bm25", base), ("bm25 + rerank", reranked)):
        print(f"{name}:")
        for n, (chunk, score) in enumerate(index.search(query, k), 1):
            print(f"  {n}. {score:8.3f}  {chunk.doc_id}  {chunk.title[:50]}")
        print()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=pathlib.Path,
                    default=pathlib.Path("corpus/documents.jsonl"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ask")
    ap.add_argument("--compare", metavar="QUERY")
    args = ap.parse_args()

    if args.compare:
        return compare(args.compare, args.corpus, args.model, args.depth,
                       args.k)
    if not args.ask:
        ap.print_help()
        return 1

    index = Reranker(build_index(args.corpus), args.model, args.depth)
    for n, (chunk, score) in enumerate(index.search(args.ask, args.k), 1):
        print(f"  {n}. {score:8.3f}  {chunk.doc_id}  {chunk.title[:50]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
