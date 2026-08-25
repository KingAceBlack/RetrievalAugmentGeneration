#!/usr/bin/env python3
"""Chunk, index, retrieve, answer.

Deliberately small. BM25 only, no embeddings, no vector database - a keyword
baseline that runs anywhere with no model download and no API key. Measure it
first: if a hybrid or semantic retriever cannot beat this on your evaluation
set, the extra machinery is not earning its place.

Generation is pluggable and optional:

    ANTHROPIC_API_KEY set   -> Claude via the HTTP API
    OLLAMA_HOST set         -> a local model (e.g. http://localhost:11434)
    neither                 -> retrieval only, no answers

Usage:
    python rag.py --ask "What is the docket number for the Olcott safety zone?"
    python rag.py --ask "..." --k 8 --show-chunks
    python rag.py --corpus corpus/dev.jsonl --ask "..."
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field

CORPUS = pathlib.Path("corpus/documents.jsonl")

CHUNK_CHARS = 1200
CHUNK_OVERLAP = 150

REFUSAL = "Not found in the available documents."

SYSTEM = (
    "You answer questions using only the numbered passages provided. "
    "Every statement must be supported by a passage, and you cite the "
    "passage numbers you used in square brackets. "
    f'If the passages do not contain the answer, reply exactly: "{REFUSAL}" '
    "Do not use prior knowledge. Do not guess. Be brief - one or two "
    "sentences is usually enough."
)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    agency: str
    text: str
    position: int


BOILERPLATE = [
    re.compile(r"^\[Federal Register Volume.*?\]\s*$", re.M),
    re.compile(r"^\[(?:Rules and Regulations|Proposed Rules|Notices)\]\s*$", re.M),
    re.compile(r"^\[Pages? [\d\-]+\]\s*$", re.M),
    re.compile(r"^From the Federal Register Online.*?$", re.M),
    re.compile(r"^\[FR Doc No: [^\]]+\]\s*$", re.M),
    re.compile(r"\[\[Page \d+\]\]"),
    re.compile(r"^=+\s*$", re.M),
    re.compile(r"^-{5,}\s*$", re.M),
]


def strip_boilerplate(text: str) -> str:
    """Remove the masthead every Federal Register document carries.

    It is identical across all 415 documents, so it contributes no signal but
    does inflate the length of the first chunk - which is where the answer
    usually lives, in the DATES and ADDRESSES blocks.
    """
    for pattern in BOILERPLATE:
        text = pattern.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def split_text(text: str, size: int = CHUNK_CHARS,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Accumulate paragraphs up to `size`, carrying a little context forward.

    Splitting on blank lines keeps sentences and list items intact, which
    matters here because the answer is often a single line - "DATES: Comments
    must be received by October 23, 2026" - that a fixed character window
    would cut in half.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        return []

    chunks: list[str] = []
    buf = ""
    for para in paras:
        while len(para) > size:               # a single huge paragraph
            head, para = para[:size], para[size - overlap:]
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.append(head)
        if buf and len(buf) + len(para) + 2 > size:
            chunks.append(buf)
            buf = (buf[-overlap:] + "\n" + para) if overlap else para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def chunk_corpus(docs: list[dict]) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        agency = (doc.get("agency_names") or ["?"])[0]
        cleaned = strip_boilerplate(doc.get("text", ""))
        for i, piece in enumerate(split_text(cleaned)):
            out.append(Chunk(
                chunk_id=f"{doc['document_number']}#{i}",
                doc_id=doc["document_number"],
                title=doc.get("title", ""),
                agency=agency,
                text=piece,
                position=i,
            ))
    return out


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+(?:[-/][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Keep hyphens and slashes together so USCG-2026-0123 and A53/A53M survive."""
    return _TOKEN.findall(text.lower())


@dataclass
class BM25:
    k1: float = 1.5
    b: float = 0.75
    chunks: list[Chunk] = field(default_factory=list)
    postings: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    lengths: list[int] = field(default_factory=list)
    avg_len: float = 0.0

    def index(self, chunks: list[Chunk]) -> "BM25":
        self.chunks = chunks
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths = []
        for i, chunk in enumerate(chunks):
            counts = Counter(tokenize(chunk.text))
            self.lengths.append(sum(counts.values()))
            for term, n in counts.items():
                postings[term].append((i, n))
        self.postings = dict(postings)
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        return self

    def search(self, query: str, k: int = 5,
               per_doc: int = 2) -> list[tuple[Chunk, float]]:
        n = len(self.chunks)
        if not n:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = math.log(1 + (n - len(posting) + 0.5) / (len(posting) + 0.5))
            for i, freq in posting:
                norm = 1 - self.b + self.b * (self.lengths[i] / (self.avg_len or 1))
                scores[i] += idf * (freq * (self.k1 + 1)) / (freq + self.k1 * norm)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])

        # Without a per-document cap, several chunks of one long document fill
        # every slot and the model never sees an alternative source.
        out: list[tuple[Chunk, float]] = []
        seen: Counter = Counter()
        for i, score in ranked:
            chunk = self.chunks[i]
            if per_doc and seen[chunk.doc_id] >= per_doc:
                continue
            seen[chunk.doc_id] += 1
            out.append((chunk, score))
            if len(out) >= k:
                break
        return out


# ---------------------------------------------------------------------------
# Generation backends
# ---------------------------------------------------------------------------

def build_prompt(question: str, hits: list[tuple[Chunk, float]]) -> str:
    parts = []
    for n, (chunk, _) in enumerate(hits, 1):
        parts.append(f"[{n}] ({chunk.doc_id} - {chunk.title})\n{chunk.text}")
    return ("Passages:\n\n" + "\n\n".join(parts)
            + f"\n\nQuestion: {question}\nAnswer:")


# Any provider speaking the OpenAI chat-completions format works here.
# name -> (env var holding the key, base URL, default model)
PROVIDERS = {
    "groq": ("GROQ_API_KEY", "https://api.groq.com/openai/v1",
             "openai/gpt-oss-120b"),
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com/v1",
                 "deepseek-chat"),
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1",
                   "meta-llama/llama-3.3-70b-instruct:free"),
    "together": ("TOGETHER_API_KEY", "https://api.together.xyz/v1",
                 "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    "openai": ("OPENAI_API_KEY", "https://api.openai.com/v1", "gpt-4o-mini"),
    "nvidia": ("NVIDIA_API_KEY", "https://integrate.api.nvidia.com/v1",
               "deepseek-ai/deepseek-v4-flash-0731"),
}

# Provider-specific request-body additions. These correspond to what the
# OpenAI SDK calls `extra_body` - it merges them into the JSON body, so over
# raw HTTP they go at the top level.
#
# Thinking is switched OFF by default. For grounded extraction the answer is
# in the passages already, so reasoning adds latency and burns output tokens
# without improving accuracy - and on a reasoning model those tokens count
# against max_tokens, which can leave no budget for the answer itself.
PROVIDER_EXTRA = {
    "nvidia": {"chat_template_kwargs": {"thinking": False}},
}


def reasoning_payload(backend: str, model: str, max_tokens: int,
                      thinking: bool) -> dict:
    """Per-model adjustments for reasoning models.

    Two families need different handling and both get it wrong by default:

      gpt-oss on Groq takes `max_completion_tokens`, not `max_tokens`, and a
      `reasoning_effort` setting. Send `max_tokens` and the budget is ignored;
      leave effort unset and it thinks at length before answering.

      DeepSeek on NVIDIA takes `chat_template_kwargs`.

    In both cases reasoning tokens are drawn from the same budget as the
    answer, so a model that deliberates for 500 tokens with a 500-token budget
    returns nothing at all. Effort is low by default: the answer is already in
    the passages, and extraction does not improve with deliberation.
    """
    extra = dict(PROVIDER_EXTRA.get(backend, {}))

    if thinking and "chat_template_kwargs" in extra:
        extra["chat_template_kwargs"] = {"thinking": True,
                                         "reasoning_effort": "low"}

    if "gpt-oss" in model:
        extra["reasoning_effort"] = "medium" if thinking else "low"
        # reasoning shares the budget, so give it room to reach an answer
        extra["max_completion_tokens"] = max(max_tokens, 2048)
        extra["_drop_max_tokens"] = True

    return extra

# Order matters only for auto-detection: whichever key is present wins.
_ORDER = ["anthropic", "nvidia", "groq", "deepseek", "openrouter",
          "together", "openai", "ollama"]


def backend_name(preferred: str | None = None) -> str | None:
    if preferred:
        return preferred
    for name in _ORDER:
        if name == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"
        if name == "ollama" and os.environ.get("OLLAMA_HOST"):
            return "ollama"
        env = PROVIDERS.get(name, (None,))[0]
        if env and os.environ.get(env):
            return name
    return None


class RateLimiter:
    """Spread requests out to stay inside a free tier's per-minute budget.

    Groq's free tier allows 30 requests a minute but only 6,000 tokens a
    minute, and a RAG request carrying five passages is around 1,800 tokens.
    The token budget binds long before the request count does, so pacing by
    requests per minute is the practical control.
    """

    def __init__(self, rpm: float = 0.0):
        self.interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._last = 0.0

    def wait(self) -> None:
        if not self.interval:
            return
        import time
        gap = self.interval - (time.monotonic() - self._last)
        if gap > 0:
            time.sleep(gap)
        self._last = time.monotonic()


_limiter = RateLimiter()


def set_rate_limit(rpm: float) -> None:
    global _limiter
    _limiter = RateLimiter(rpm)


def _post(url: str, headers: dict, payload: dict, tries: int = 5,
          timeout: float = 120.0):
    """POST with backoff.

    Retries on rate limits, server errors AND network timeouts. A timeout is
    the most common failure against a free endpoint under load, and treating
    it as fatal means one slow request ends a 75-question run.
    """
    import time

    import requests

    last = None
    for attempt in range(tries):
        _limiter.wait()
        try:
            resp = requests.post(url, headers=headers, json=payload,
                                 timeout=timeout)
        except requests.RequestException as exc:
            last = exc
            delay = min(30.0, 5.0 * (attempt + 1))
            print(f"    {type(exc).__name__}, retrying in {delay:.0f}s "
                  f"({attempt + 1}/{tries})", file=sys.stderr, flush=True)
            time.sleep(delay)
            continue

        if resp.status_code == 429 or resp.status_code >= 500:
            retry_after = resp.headers.get("retry-after")
            try:
                delay = float(retry_after) if retry_after else 0.0
            except ValueError:
                delay = 0.0
            delay = delay or min(60.0, 5.0 * (attempt + 1))
            print(f"    HTTP {resp.status_code}, waiting {delay:.0f}s",
                  file=sys.stderr, flush=True)
            time.sleep(delay)
            continue

        resp.raise_for_status()
        return resp.json()

    raise RuntimeError(f"gave up after {tries} attempts: {url} ({last})")


def generate(question: str, hits: list[tuple[Chunk, float]],
             model: str | None = None,
             backend: str | None = None,
             max_tokens: int = 500,
             thinking: bool = False,
             timeout: float = 120.0) -> str | None:
    backend = backend_name(backend)
    if backend is None:
        return None

    prompt = build_prompt(question, hits)

    if backend == "anthropic":
        data = _post(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": os.environ["ANTHROPIC_API_KEY"],
             "anthropic-version": "2023-06-01",
             "content-type": "application/json"},
            {"model": model or "claude-sonnet-4-6", "max_tokens": max_tokens,
             "system": SYSTEM,
             "messages": [{"role": "user", "content": prompt}]},
            timeout=timeout,
        )
        return "".join(b.get("text", "") for b in data.get("content", [])
                       if b.get("type") == "text").strip()

    if backend == "ollama":
        host = os.environ["OLLAMA_HOST"].rstrip("/")
        data = _post(f"{host}/api/chat", {},
                     {"model": model or "llama3.1", "stream": False,
                      "messages": [{"role": "system", "content": SYSTEM},
                                   {"role": "user", "content": prompt}]},
                     timeout=timeout)
        return (data.get("message") or {}).get("content", "").strip()

    if backend not in PROVIDERS:
        raise SystemExit(f"unknown backend: {backend}")

    env, base, default_model = PROVIDERS[backend]
    key = os.environ.get(env)
    if not key:
        raise SystemExit(f"{env} is not set")
    base = os.environ.get("OPENAI_BASE_URL", base).rstrip("/")

    model = model or default_model
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        # Zero, not the vendor sample's 1. A benchmark that changes its answers
        # between runs cannot tell you whether a change helped.
        "temperature": 0,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": prompt}],
    }
    extra = reasoning_payload(backend, model, max_tokens, thinking)
    if extra.pop("_drop_max_tokens", False):
        payload.pop("max_tokens")
    payload.update(extra)

    data = _post(
        f"{base}/chat/completions",
        {"Authorization": f"Bearer {key}", "content-type": "application/json"},
        payload,
        timeout=timeout,
    )
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = (message.get("content") or "").strip()

    # A reasoning model returns its scratchpad separately. That is never the
    # answer - but if the whole token budget went to thinking, content comes
    # back empty and the run would silently score zero.
    if not content and (message.get("reasoning_content")
                        or message.get("reasoning")):
        print("    warning: model returned only reasoning, no answer - "
              "raise --max-tokens or disable thinking", file=sys.stderr)
    return content


# ---------------------------------------------------------------------------

def load_corpus(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} not found - run fetch_fr.py first")
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def build_index(path: pathlib.Path = CORPUS, quiet: bool = False) -> BM25:
    docs = load_corpus(path)
    chunks = chunk_corpus(docs)
    if not quiet:
        print(f"{len(docs)} documents -> {len(chunks)} chunks", file=sys.stderr)
    return BM25().index(chunks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=pathlib.Path, default=CORPUS)
    ap.add_argument("--ask", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--per-doc", type=int, default=2,
                    help="max chunks from any one document")
    ap.add_argument("--model")
    ap.add_argument("--backend", choices=["anthropic", "ollama",
                                          *PROVIDERS])
    ap.add_argument("--rpm", type=float, default=0,
                    help="throttle requests per minute (Groq free tier: 3)")
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--thinking", action="store_true",
                    help="enable reasoning on models that support it")
    ap.add_argument("--show-chunks", action="store_true")
    args = ap.parse_args()

    index = build_index(args.corpus)
    hits = index.search(args.ask, args.k, args.per_doc)
    if not hits:
        print("no passages matched")
        return 1

    print(f"\ntop {len(hits)} passages:")
    for n, (chunk, score) in enumerate(hits, 1):
        print(f"  [{n}] {score:6.2f}  {chunk.doc_id}  {chunk.title[:52]}")
        if args.show_chunks:
            preview = " ".join(chunk.text.split())[:220]
            print(f"        {preview}...")

    set_rate_limit(args.rpm)
    answer = generate(args.ask, hits, args.model, args.backend,
                      args.max_tokens, args.thinking, args.timeout)
    if answer is None:
        print("\n(no generation backend - set one of: "
              + ", ".join([e for e, _, _ in PROVIDERS.values()])
              + ", ANTHROPIC_API_KEY, OLLAMA_HOST)")
    else:
        print(f"\nanswer: {answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
