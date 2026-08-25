#!/usr/bin/env python3
"""Download a large, recent, English-only corpus from the US Federal Register.

Why this source, for a RAG test corpus:

  * Recency is the only reliable guarantee that text is not in a model's
    training data. The Federal Register publishes every federal business day,
    so anything after a chosen date is genuinely unseen.
  * Public domain (US government work), so it can be redistributed and shown
    to prospects without a licensing conversation.
  * Structurally similar to enterprise documents: dry, procedural, sectioned,
    and dense with checkable specifics - docket numbers, CFR citations,
    comment deadlines, named contacts. Those make good evaluation questions
    precisely because no model can guess them.
  * No API key.

How it works: the listing endpoint ignores `fields[]` and returns a fixed set
of ten keys, none of which is the full-text link. So the listing is used only
to enumerate document numbers for a date, and each document is then read from
the single-document endpoint, which carries the metadata worth keeping -
comment deadlines, effective dates, CFR references, docket IDs.

Two requests per document. A 30-day window is roughly 4,800 documents, so
expect ~9,600 requests and a run of 20-40 minutes. Use --resume if it breaks.

Usage:
    python fetch_fr.py --probe                  # inspect the API's field names
    python fetch_fr.py                          # last 30 days
    python fetch_fr.py --start 2026-06-01 --end 2026-08-01
    python fetch_fr.py --types RULE PRORULE     # denser documents
    python fetch_fr.py --resume                 # continue an interrupted run
    python fetch_fr.py --stats

Output:
    corpus/documents.jsonl   one JSON record per document, full text included
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

API = "https://www.federalregister.gov/api/v1"
UA = "enrag-corpus-builder/0.3 (research; contact: you@example.com)"

OUT = pathlib.Path("corpus/documents.jsonl")

# Kept from the single-document endpoint. Anything absent is simply skipped,
# so adding a field here is safe.
KEEP = [
    "document_number", "title", "type", "subtype", "abstract", "action",
    "publication_date", "effective_on", "comments_close_on", "signing_date",
    "citation", "volume", "start_page", "end_page", "page_length",
    "docket_ids", "regulation_id_numbers", "topics", "significant",
    "html_url", "regulations_dot_gov_url", "correction_of",
    "executive_order_number", "presidential_document_number",
]

_local = threading.local()
_write_lock = threading.Lock()


def session() -> requests.Session:
    """One session per thread; requests.Session is not thread-safe."""
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": UA})
        _local.s = s
    return s


def get(url: str, params: dict | None = None, tries: int = 4,
        as_json: bool = True):
    for attempt in range(tries):
        try:
            r = session().get(url, params=params, timeout=90)
        except requests.RequestException:
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json() if as_json else r.text
    raise RuntimeError(f"gave up on {url}")


# ---------------------------------------------------------------------------
# Listing: enumerate document numbers for a day
# ---------------------------------------------------------------------------

def list_day(day: dt.date, types: list[str] | None) -> list[dict]:
    params = {
        "conditions[publication_date][is]": day.isoformat(),
        "per_page": 1000,
    }
    if types:
        params["conditions[type][]"] = types

    out: list[dict] = []
    data = get(f"{API}/documents.json", params)
    while data:
        out.extend(data.get("results") or [])
        nxt = data.get("next_page_url")
        if not nxt:
            break
        data = get(nxt)
    return out


def daterange(start: dt.date, end: dt.date):
    day = start
    while day <= end:
        if day.weekday() < 5:          # the FR publishes on business days
            yield day
        day += dt.timedelta(days=1)


def is_correction(doc: dict) -> bool:
    """Corrections restate a few lines of an earlier document."""
    if doc.get("correction_of"):
        return True
    return bool(re.match(r"^C\d+-", doc.get("document_number") or ""))


# ---------------------------------------------------------------------------
# Detail + full text
# ---------------------------------------------------------------------------

_PRE = re.compile(r"<pre>(.*?)</pre>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")


def clean_raw_text(raw: str) -> str:
    """The raw_text endpoint wraps the document in a minimal HTML shell."""
    m = _PRE.search(raw)
    text = m.group(1) if m else raw
    text = _TAG.sub("", text)
    text = (text.replace("&amp;", "&").replace("&lt;", "<")
                .replace("&gt;", ">").replace("&quot;", '"')
                .replace("&#39;", "'").replace("&nbsp;", " "))
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def agency_names(detail: dict) -> list[str]:
    """The API returns agency objects; `agency_names` does not exist."""
    names = []
    for a in detail.get("agencies") or []:
        name = a.get("name") or a.get("raw_name")
        if name and name not in names:
            names.append(name)
    return names


def cfr_refs(detail: dict) -> list[str]:
    """Flatten [{'title': 40, 'part': 52}] into ['40 CFR 52']."""
    out = []
    for ref in detail.get("cfr_references") or []:
        title, part = ref.get("title"), ref.get("part")
        if title is None:
            continue
        label = f"{title} CFR {part}" if part is not None else f"{title} CFR"
        if label not in out:
            out.append(label)
    return out


def build_record(detail: dict, text: str) -> dict:
    rec = {k: detail[k] for k in KEEP if detail.get(k) not in (None, [], "")}
    rec["agency_names"] = agency_names(detail)
    refs = cfr_refs(detail)
    if refs:
        rec["cfr_references"] = refs
    rec["text"] = text
    rec["char_count"] = len(text)
    rec["word_count"] = len(text.split())
    return rec


def fetch_document(number: str) -> dict | None:
    detail = get(f"{API}/documents/{number}.json")
    if not detail:
        return None
    url = detail.get("raw_text_url")
    if not url:
        return None
    raw = get(url, as_json=False)
    if not raw:
        return None
    text = clean_raw_text(raw)
    if len(text) < 200:               # placeholder or failed extraction
        return None
    return build_record(detail, text)


# ---------------------------------------------------------------------------

def probe() -> int:
    """Print the field names the API actually returns, so nothing is guessed."""
    data = get(f"{API}/documents.json", {"per_page": 1})
    results = (data or {}).get("results") or []
    if not results:
        print("no results returned")
        return 1
    print("listing result keys (fields[] is ignored here):")
    for k in sorted(results[0]):
        print(f"  {k}")

    num = results[0]["document_number"]
    detail = get(f"{API}/documents/{num}.json")
    print(f"\nsingle-document keys ({num}):")
    for k in sorted(detail or {}):
        print(f"  {k}")

    missing = [f for f in KEEP if f not in (detail or {})]
    print(f"\nof the fields this script keeps, {len(KEEP) - len(missing)}"
          f"/{len(KEEP)} present on this document")
    if missing:
        print(f"  absent here (may exist on other types): {missing}")
    print(f"\ncorrection document: {is_correction(detail or {})}")
    return 0


def already_have(path: pathlib.Path) -> set[str]:
    if not path.exists():
        return set()
    seen = set()
    for line in path.open(encoding="utf-8"):
        try:
            seen.add(json.loads(line)["document_number"])
        except Exception:
            continue
    return seen


def stats(path: pathlib.Path = OUT) -> int:
    if not path.exists():
        print(f"{path} not found - run the fetcher first")
        return 1
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    if not rows:
        print("corpus is empty")
        return 1
    words = sum(r.get("word_count", 0) for r in rows)
    print(f"{len(rows)} documents, {words:,} words "
          f"({words // len(rows):,} avg)")

    print("\nby type:")
    for t, n in Counter(r.get("type") or "?" for r in rows).most_common():
        print(f"  {t:24s} {n:5d}")

    print("\ntop agencies:")
    ag = Counter(a for r in rows for a in (r.get("agency_names") or []))
    for a, n in ag.most_common(10):
        print(f"  {a[:44]:46s} {n:5d}")

    dates = sorted(r.get("publication_date", "") for r in rows)
    print(f"\ndate range: {dates[0]} to {dates[-1]}")

    print("\nmetadata usable as ground truth:")
    for field in ("comments_close_on", "effective_on", "docket_ids",
                  "cfr_references", "regulation_id_numbers", "citation"):
        n = sum(1 for r in rows if r.get(field))
        print(f"  {field:24s} {n:5d}  ({n / len(rows):.0%})")

    longest = max(rows, key=lambda r: r.get("word_count", 0))
    print(f"\nlongest: {longest['word_count']:,} words - "
          f"{longest['title'][:60]}")
    return 0


def main() -> int:
    today = dt.date.today()
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=dt.date.fromisoformat,
                    default=today - dt.timedelta(days=30))
    ap.add_argument("--end", type=dt.date.fromisoformat, default=today)
    ap.add_argument("--types", nargs="+",
                    choices=["RULE", "PRORULE", "NOTICE", "PRESDOCU"],
                    help="default: all types")
    ap.add_argument("--min-words", type=int, default=300)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--keep-corrections", action="store_true",
                    help="include C1-/C2- correction documents")
    ap.add_argument("--resume", action="store_true",
                    help="append to an existing corpus, skipping what's there")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    if args.probe:
        return probe()
    if args.stats:
        return stats()
    if args.start > args.end:
        print("--start is after --end", file=sys.stderr)
        return 1

    days = list(daterange(args.start, args.end))
    print(f"listing {len(days)} business days ({args.start} to {args.end})")

    index: list[dict] = []
    for day in days:
        try:
            found = list_day(day, args.types)
        except Exception as exc:
            print(f"  {day}: failed ({exc})", file=sys.stderr)
            continue
        index.extend(found)
        print(f"  {day}: {len(found):4d} documents  (total {len(index)})")
        time.sleep(0.3)

    if not index:
        print("nothing found for that range")
        return 1

    corrections = 0
    if not args.keep_corrections:
        before = len(index)
        index = [d for d in index if not is_correction(d)]
        corrections = before - len(index)

    seen = already_have(OUT) if args.resume else set()
    numbers = [d["document_number"] for d in index
               if d["document_number"] not in seen]

    if not numbers:
        print("\nnothing new to fetch")
        return 0

    print(f"\n{len(numbers)} documents to fetch"
          + (f" ({corrections} corrections skipped)" if corrections else "")
          + (f", {len(seen)} already present" if seen else ""))
    print(f"two requests each, {args.workers} workers (Ctrl-C is safe; rerun with --resume)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if (args.resume and OUT.exists()) else "w"

    kept = short = failed = 0
    with OUT.open(mode, encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_document, n): n for n in numbers}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
            except Exception:
                rec = None
            if rec is None:
                failed += 1
            elif rec["word_count"] < args.min_words:
                short += 1
            else:
                with _write_lock:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
            if i % 25 == 0 or i == len(numbers):
                fh.flush()
                print(f"  {i}/{len(numbers)}  kept={kept} "
                      f"short={short} failed={failed}")

    print(f"\n{kept} documents -> {OUT}")
    print(f"  {short} skipped (under {args.min_words} words)")
    print(f"  {failed} failed to fetch")
    print("\nrun  python fetch_fr.py --stats  to summarise the corpus")
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
