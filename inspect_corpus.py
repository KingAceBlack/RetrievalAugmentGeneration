#!/usr/bin/env python3
"""Inspect the corpus and carve a small working slice from it.

Answers three questions before any retrieval code is written:

  1. How is length distributed? A single enormous document distorts an index
     and makes document-level ground truth meaningless.
  2. Are any documents byte-identical? Two "correct" answers to one question
     silently corrupts a benchmark.
  3. Which documents are near-identical? Those are not noise to remove - they
     are the hard case worth testing against, and they need to be known.

Then writes a development slice: small enough to iterate on in seconds, but
deliberately including near-duplicates so the easy path is not the only one
being exercised.

Usage:
    python inspect_corpus.py                    # report only
    python inspect_corpus.py --dev              # also write corpus/dev.jsonl
    python inspect_corpus.py --dev --dev-size 60 --max-words 30000
    python inspect_corpus.py --threshold 0.7    # looser near-duplicate cutoff

No third-party dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import statistics
import zlib
from collections import Counter, defaultdict

CORPUS = pathlib.Path("corpus/documents.jsonl")
DEV = pathlib.Path("corpus/dev.jsonl")

SKETCH_SIZE = 256      # bottom-k sketch; larger = more accurate, slower
SHINGLE = 5            # words per shingle
MAX_TOKENS = 20_000    # cap per document, so one huge file cannot dominate


# ---------------------------------------------------------------------------
# Loading and normalising
# ---------------------------------------------------------------------------

def load(path: pathlib.Path = CORPUS) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} not found - run fetch_fr.py first")
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


_WS = re.compile(r"\s+")


def norm_text(text: str) -> str:
    return _WS.sub(" ", text.lower()).strip()


def exact_key(text: str) -> str:
    return hashlib.sha256(norm_text(text).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Near-duplicate detection (bottom-k sketch, Jaccard estimate)
# ---------------------------------------------------------------------------

def sketch(text: str, k: int = SKETCH_SIZE, shingle: int = SHINGLE,
           max_tokens: int = MAX_TOKENS) -> tuple[int, ...]:
    """The k smallest shingle hashes - a compact stand-in for the full set.

    Hashing every shingle once and keeping the smallest k gives a usable
    Jaccard estimate at a fraction of the cost of MinHash with k permutations.
    crc32 is used for speed; occasional collisions do not matter at this scale.
    """
    words = norm_text(text).split()[:max_tokens]
    if len(words) < shingle:
        return tuple(sorted({zlib.crc32(" ".join(words).encode())}))
    seen: set[int] = set()
    for i in range(len(words) - shingle + 1):
        seen.add(zlib.crc32(" ".join(words[i:i + shingle]).encode()))
    return tuple(sorted(seen)[:k])


def jaccard(a: tuple[int, ...], b: tuple[int, ...],
            k: int = SKETCH_SIZE) -> float:
    """Estimate Jaccard similarity from two bottom-k sketches."""
    if not a or not b:
        return 0.0
    union = sorted(set(a) | set(b))[:k]
    if not union:
        return 0.0
    both = set(a) & set(b)
    return sum(1 for h in union if h in both) / len(union)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[rj] = ri


def similarity_pass(rows: list[dict],
                    threshold: float) -> tuple[list[list[int]], list[float]]:
    """One O(n^2) pass returning both clusters and each document's nearest match.

    The nearest-neighbour scores are what let you choose a threshold from
    evidence instead of guessing: a corpus of boilerplate-heavy documents
    clusters at a very different cutoff from a corpus of distinct ones.
    """
    sketches = [sketch(r.get("text", "")) for r in rows]
    uf = UnionFind(len(rows))
    nearest = [0.0] * len(rows)

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            score = jaccard(sketches[i], sketches[j])
            if score > nearest[i]:
                nearest[i] = score
            if score > nearest[j]:
                nearest[j] = score
            if score >= threshold:
                uf.union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(rows)):
        groups[uf.find(i)].append(i)
    clusters = sorted((sorted(v) for v in groups.values() if len(v) > 1),
                      key=len, reverse=True)
    return clusters, nearest


def near_duplicate_clusters(rows: list[dict],
                            threshold: float = 0.8) -> list[list[int]]:
    """Group documents whose estimated Jaccard similarity meets the threshold."""
    return similarity_pass(rows, threshold)[0]


def report_similarity_distribution(nearest: list[float]) -> None:
    """Show how close each document is to its most similar neighbour."""
    ordered = sorted(nearest, reverse=True)
    print("\nnearest-neighbour similarity (each document vs its closest match):")
    for label, idx in (("max", 0),
                       ("p90", int(0.10 * len(ordered))),
                       ("p75", int(0.25 * len(ordered))),
                       ("median", len(ordered) // 2),
                       ("p25", int(0.75 * len(ordered))),
                       ("min", len(ordered) - 1)):
        print(f"  {label:8s} {ordered[min(idx, len(ordered) - 1)]:.3f}")

    buckets = [(0.95, "1.00"), (0.85, "0.95"), (0.75, "0.85"),
               (0.60, "0.75"), (0.40, "0.60"), (0.00, "0.40")]
    print("\n  documents by closest-match score:")
    for lo, hi in buckets:
        n = sum(1 for v in nearest if lo <= v < float(hi) or
                (hi == "1.00" and v >= lo))
        if n:
            bar = "#" * min(40, max(1, n * 40 // max(len(nearest), 1)))
            print(f"    {lo:.2f}-{hi}  {n:5d}  {bar}")
    print("\n  pick --threshold just below the cluster you want to catch")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_lengths(rows: list[dict]) -> list[dict]:
    """Print the length distribution. Returns documents flagged as outliers."""
    counts = sorted(r.get("word_count", 0) for r in rows)
    total = sum(counts)
    print(f"{len(rows)} documents, {total:,} words\n")
    print("word count distribution:")
    for label, value in (
        ("min", counts[0]),
        ("p25", counts[len(counts) // 4]),
        ("median", int(statistics.median(counts))),
        ("p75", counts[3 * len(counts) // 4]),
        ("p95", counts[int(0.95 * len(counts)) - 1]),
        ("max", counts[-1]),
        ("mean", total // len(counts)),
    ):
        print(f"  {label:8s} {value:>10,}")

    # A document is an outlier when it holds a disproportionate share.
    outliers = [r for r in rows if r.get("word_count", 0) > 0.05 * total]
    if outliers:
        print("\ndocuments holding over 5% of the corpus on their own:")
        for r in sorted(outliers, key=lambda x: -x["word_count"]):
            share = r["word_count"] / total
            print(f"  {r['word_count']:>9,} words ({share:.0%})  "
                  f"{r['document_number']}  {r['title'][:48]}")
    return outliers


def report_exact(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[exact_key(r.get("text", ""))].append(r)
    dupes = {k: v for k, v in groups.items() if len(v) > 1}
    if not dupes:
        print("\nexact duplicates: none")
        return {}
    print(f"\nexact duplicates: {len(dupes)} group(s) - "
          f"these WILL corrupt ground truth")
    for members in dupes.values():
        nums = ", ".join(m["document_number"] for m in members)
        print(f"  {nums}  {members[0]['title'][:48]}")
    return dupes


def report_near(rows: list[dict], clusters: list[list[int]],
                threshold: float) -> None:
    if not clusters:
        print(f"\nnear-duplicate clusters (>={threshold:.0%}): none")
        return
    involved = sum(len(c) for c in clusters)
    print(f"\nnear-duplicate clusters (>={threshold:.0%}): "
          f"{len(clusters)} groups covering {involved} documents")
    print("  these are the hard case - keep them, build questions against them")
    for c in clusters[:5]:
        agencies = Counter(a for i in c
                           for a in (rows[i].get("agency_names") or []))
        top = agencies.most_common(1)[0][0] if agencies else "?"
        print(f"\n  cluster of {len(c)} - mostly {top[:40]}")
        for i in c[:3]:
            print(f"    {rows[i]['document_number']}  "
                  f"{rows[i]['title'][:56]}")
        if len(c) > 3:
            print(f"    ... and {len(c) - 3} more")


# ---------------------------------------------------------------------------
# Development slice
# ---------------------------------------------------------------------------

def build_dev(rows: list[dict], clusters: list[list[int]], size: int,
              max_words: int, per_cluster: int = 4) -> list[dict]:
    """A small slice: mostly varied, deliberately including near-duplicates.

    A dev set of only distinct documents would make every retriever look good.
    Including one near-duplicate cluster keeps the hard case in view while
    iterating.
    """
    eligible = {i for i, r in enumerate(rows)
                if r.get("word_count", 0) <= max_words}

    picked: list[int] = []

    # Seed from the largest few clusters, so more than one document family is
    # present. Seeding from a single cluster leaves the slice easy in every
    # other respect, which is the failure this slice exists to avoid.
    seeded = 0
    for cluster in clusters:
        if seeded >= max(3, size // 10):
            break
        members = [i for i in cluster if i in eligible and i not in picked]
        if len(members) >= 2:
            picked.extend(members[:per_cluster])
            seeded += 1

    # Fill the rest round-robin across agencies, for variety.
    by_agency: dict[str, list[int]] = defaultdict(list)
    for i in sorted(eligible):
        if i in picked:
            continue
        names = rows[i].get("agency_names") or ["(none)"]
        by_agency[names[0]].append(i)

    order = sorted(by_agency, key=lambda a: -len(by_agency[a]))
    cursor = {a: 0 for a in order}
    while len(picked) < size:
        added = False
        for agency in order:
            if len(picked) >= size:
                break
            idx = cursor[agency]
            if idx < len(by_agency[agency]):
                picked.append(by_agency[agency][idx])
                cursor[agency] += 1
                added = True
        if not added:
            break

    return [rows[i] for i in picked]


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=pathlib.Path, default=CORPUS)
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="Jaccard cutoff for near-duplicate clustering")
    ap.add_argument("--dev", action="store_true",
                    help="write corpus/dev.jsonl")
    ap.add_argument("--dev-size", type=int, default=40)
    ap.add_argument("--per-cluster", type=int, default=4,
                    help="documents to seed from each near-duplicate family")
    ap.add_argument("--max-words", type=int, default=50_000,
                    help="exclude documents longer than this from the slice")
    args = ap.parse_args()

    rows = load(args.corpus)
    report_lengths(rows)
    report_exact(rows)

    print("\ncomparing every document against every other ...")
    clusters, nearest = similarity_pass(rows, args.threshold)
    report_similarity_distribution(nearest)
    report_near(rows, clusters, args.threshold)

    if not args.dev:
        print("\nrun with --dev to write a working slice")
        return 0

    slice_rows = build_dev(rows, clusters, args.dev_size, args.max_words,
                           args.per_cluster)
    DEV.parent.mkdir(parents=True, exist_ok=True)
    with DEV.open("w", encoding="utf-8") as fh:
        for r in slice_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    words = sum(r.get("word_count", 0) for r in slice_rows)
    agencies = Counter(a for r in slice_rows
                       for a in (r.get("agency_names") or []))
    print(f"\n{len(slice_rows)} documents -> {DEV}")
    print(f"  {words:,} words, {len(agencies)} agencies")
    print(f"  excluded documents over {args.max_words:,} words")
    types = Counter(r.get("type") for r in slice_rows)
    print("  " + ", ".join(f"{t}: {n}" for t, n in types.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
