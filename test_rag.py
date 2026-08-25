import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rag import BM25, Chunk, chunk_corpus, split_text, tokenize  # noqa: E402
from evaluate import contains_answer, is_refusal, normalise  # noqa: E402


def doc(num, text, title=None, agency="Coast Guard"):
    return {"document_number": num, "title": title or f"Rule {num}",
            "agency_names": [agency], "text": text}


# --- chunking --------------------------------------------------------------

def test_splits_on_paragraph_boundaries():
    text = "First para.\n\nSecond para.\n\nThird para."
    assert split_text(text, size=1000) == [text.replace("\n\n", "\n\n")]


def test_long_text_produces_several_chunks():
    text = "\n\n".join(f"Paragraph number {i} with some filler words." * 3
                       for i in range(40))
    chunks = split_text(text, size=500, overlap=50)
    assert len(chunks) > 3
    assert all(len(c) <= 700 for c in chunks)


def test_single_oversized_paragraph_is_broken_up():
    chunks = split_text("x" * 5000, size=1000, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)


def test_dates_line_survives_intact():
    """The answer is often one line; it must not be cut in half."""
    line = "DATES: Comments must be received by October 23, 2026."
    text = "\n\n".join(["Filler paragraph." * 20, line, "More filler." * 20])
    chunks = split_text(text, size=600, overlap=80)
    assert any(line in c for c in chunks)


def test_empty_text_yields_no_chunks():
    assert split_text("") == []
    assert split_text("   \n\n  ") == []


def test_chunk_corpus_carries_metadata():
    chunks = chunk_corpus([doc("2026-1", "A para.\n\nB para.",
                               title="Safety Zone; Olcott")])
    assert chunks[0].doc_id == "2026-1"
    assert chunks[0].title == "Safety Zone; Olcott"
    assert chunks[0].agency == "Coast Guard"
    assert chunks[0].chunk_id == "2026-1#0"


# --- tokenisation ----------------------------------------------------------

def test_docket_numbers_stay_whole():
    assert "uscg-2026-0123" in tokenize("Docket Number USCG-2026-0123.")


def test_astm_standards_stay_whole():
    assert "a53/a53m" in tokenize("ASTM standard A53/A53M applies.")


def test_punctuation_dropped():
    assert tokenize("Effective: October 23, 2026.") == [
        "effective", "october", "23", "2026"]


# --- BM25 ------------------------------------------------------------------

def build(texts):
    chunks = [Chunk(f"d{i}#0", f"d{i}", f"Title {i}", "A", t, 0)
              for i, t in enumerate(texts)]
    return BM25().index(chunks)


def test_finds_the_relevant_chunk():
    index = build([
        "The Coast Guard establishes a safety zone on Lake Ontario at Olcott.",
        "PHMSA incorporates ASTM standard A53 for steel pipe in pipelines.",
        "The Secretary of Education amends grant administrative regulations.",
    ])
    top = index.search("safety zone Olcott Lake Ontario", 1)
    assert top[0][0].doc_id == "d0"


def test_rare_identifier_beats_common_words():
    index = build([
        "This rule concerns a safety zone. Docket Number USCG-2026-0007.",
        "This rule concerns a safety zone. Docket Number USCG-2026-0008.",
        "This rule concerns a safety zone in another location entirely.",
    ])
    top = index.search("USCG-2026-0008", 1)
    assert top[0][0].doc_id == "d1"


def test_empty_index_returns_nothing():
    assert BM25().index([]).search("anything", 5) == []


def test_query_with_no_matching_terms():
    assert build(["alpha beta gamma"]).search("zzzzz qqqqq", 5) == []


def test_results_are_ordered_by_score():
    index = build(["safety zone " * 10, "safety " * 10, "unrelated text"])
    hits = index.search("safety zone", 3)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_k_limits_results():
    index = build([f"safety zone number {i}" for i in range(10)])
    assert len(index.search("safety zone", 3)) == 3


# --- scoring ---------------------------------------------------------------

def test_answer_match_ignores_punctuation_and_case():
    assert contains_answer("The deadline is October 23, 2026.",
                           "october 23 2026")


def test_answer_match_finds_docket_in_prose():
    assert contains_answer("The docket is USCG-2026-0123 [2].",
                           "USCG-2026-0123")


def test_wrong_answer_not_matched():
    assert not contains_answer("The deadline is October 24, 2026.",
                               "October 23, 2026")


def test_refusal_detected():
    assert is_refusal("Not found in the available documents.")
    assert is_refusal("not found in the available documents")


def test_normal_answer_is_not_a_refusal():
    assert not is_refusal("The docket number is USCG-2026-0123 [1].")


def test_normalise_collapses_separators():
    assert normalise("USCG-2026/0123") == "uscg 2026 0123"


# --- prompt construction ---------------------------------------------------

def test_prompt_numbers_passages_and_names_source_documents():
    from rag import build_prompt
    hits = [(Chunk("a#0", "2026-15300", "Safety Zone; Olcott", "CG",
                   "Effective September 5, 2026.", 0), 3.2),
            (Chunk("b#0", "2026-15301", "Safety Zone; Aurora", "CG",
                   "Effective September 6, 2026.", 0), 2.1)]
    prompt = build_prompt("When does the Olcott zone take effect?", hits)
    assert "[1] (2026-15300 - Safety Zone; Olcott)" in prompt
    assert "[2] (2026-15301 - Safety Zone; Aurora)" in prompt
    assert prompt.rstrip().endswith("Answer:")
    assert "When does the Olcott zone take effect?" in prompt


def test_system_prompt_requires_grounding_and_refusal():
    from rag import SYSTEM, REFUSAL
    assert REFUSAL in SYSTEM
    assert "only the numbered passages" in SYSTEM
    assert "cite" in SYSTEM.lower()


def test_no_backend_returns_none(monkeypatch):
    import rag
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert rag.backend_name() is None
    assert rag.generate("q", []) is None


# --- boilerplate stripping -------------------------------------------------

FR_HEADER = """[Federal Register Volume 91, Number 144 (Wednesday, July 29, 2026)]
[Rules and Regulations]
[Pages 47791-47792]
From the Federal Register Online via the Government Publishing Office [www.gpo.gov]
[FR Doc No: 2026-15300]
=======================================================================

AGENCY: Coast Guard, DHS.

DATES: This rule is effective September 5, 2026.

ADDRESSES: Docket Number USCG-2026-0100.

Text continues [[Page 47792]] across a page break."""


def test_masthead_removed():
    from rag import strip_boilerplate
    out = strip_boilerplate(FR_HEADER)
    assert "Federal Register Volume 91" not in out
    assert "Government Publishing Office" not in out
    assert "[FR Doc No:" not in out
    assert "[[Page 47792]]" not in out
    assert "=====" not in out


def test_substance_survives_stripping():
    from rag import strip_boilerplate
    out = strip_boilerplate(FR_HEADER)
    assert "DATES: This rule is effective September 5, 2026." in out
    assert "Docket Number USCG-2026-0100." in out
    assert "AGENCY: Coast Guard, DHS." in out


def test_stripping_is_idempotent():
    from rag import strip_boilerplate
    once = strip_boilerplate(FR_HEADER)
    assert strip_boilerplate(once) == once


def test_stripping_leaves_clean_text_alone():
    from rag import strip_boilerplate
    text = "AGENCY: Coast Guard.\n\nDATES: Effective September 5, 2026."
    assert strip_boilerplate(text) == text


# --- per-document diversity ------------------------------------------------

def test_one_document_cannot_fill_every_slot():
    chunks = [Chunk(f"big#{i}", "big", "Long Rule", "A",
                    f"safety zone paragraph {i} about fireworks", i)
              for i in range(10)]
    chunks.append(Chunk("other#0", "other", "Other Rule", "A",
                        "safety zone fireworks in another location", 0))
    index = BM25().index(chunks)
    hits = index.search("safety zone fireworks", k=5, per_doc=2)
    docs = [c.doc_id for c, _ in hits]
    assert docs.count("big") <= 2
    assert "other" in docs


def test_per_doc_zero_disables_the_cap():
    chunks = [Chunk(f"big#{i}", "big", "T", "A", f"safety zone {i}", i)
              for i in range(6)]
    index = BM25().index(chunks)
    hits = index.search("safety zone", k=4, per_doc=0)
    assert len(hits) == 4
    assert all(c.doc_id == "big" for c, _ in hits)


# --- backend selection -----------------------------------------------------

def test_auto_detects_groq_from_env(monkeypatch):
    import rag
    for env, _, _ in rag.PROVIDERS.values():
        monkeypatch.delenv(env, raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    assert rag.backend_name() == "groq"


def test_explicit_backend_overrides_detection(monkeypatch):
    import rag
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    assert rag.backend_name("deepseek") == "deepseek"


def test_anthropic_wins_auto_detection_when_both_present(monkeypatch):
    import rag
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    assert rag.backend_name() == "anthropic"


def test_missing_key_for_named_backend_exits(monkeypatch):
    import rag
    import pytest
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        rag.generate("q", [], backend="groq")


def test_openai_compatible_payload_shape(monkeypatch):
    """Groq, DeepSeek and OpenRouter all take the same request body."""
    import rag
    captured = {}

    def fake_post(url, headers, payload, tries=5, timeout=120.0):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "USCG-2026-0100 [1]"}}]}

    monkeypatch.setattr(rag, "_post", fake_post)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    hits = [(Chunk("a#0", "d1", "T", "A", "Docket USCG-2026-0100.", 0), 1.0)]

    out = rag.generate("What is the docket?", hits, backend="groq")
    assert out == "USCG-2026-0100 [1]"
    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"]["Authorization"] == "Bearer gsk_test"
    assert captured["payload"]["messages"][0]["role"] == "system"
    assert captured["payload"]["temperature"] == 0


def test_empty_choices_returns_empty_string(monkeypatch):
    import rag
    monkeypatch.setattr(rag, "_post", lambda *a, **k: {"choices": []})
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    assert rag.generate("q", [], backend="groq") == ""


# --- rate limiting ---------------------------------------------------------

def test_rate_limiter_spaces_calls():
    import time
    from rag import RateLimiter
    limiter = RateLimiter(rpm=120)          # 0.5s apart
    limiter.wait()
    start = time.monotonic()
    limiter.wait()
    assert time.monotonic() - start >= 0.4


def test_rate_limiter_disabled_by_default():
    import time
    from rag import RateLimiter
    limiter = RateLimiter(rpm=0)
    start = time.monotonic()
    for _ in range(5):
        limiter.wait()
    assert time.monotonic() - start < 0.05


# --- NVIDIA NIM / reasoning models -----------------------------------------

def test_nvidia_detected_and_uses_nim_endpoint(monkeypatch):
    import rag
    captured = {}

    def fake_post(url, headers, payload, tries=5, timeout=120.0):
        captured.update(url=url, headers=headers, payload=payload)
        return {"choices": [{"message": {"content": "USCG-2026-0100 [1]"}}]}

    monkeypatch.setattr(rag, "_post", fake_post)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")

    out = rag.generate("What is the docket?", [], backend="nvidia")
    assert out == "USCG-2026-0100 [1]"
    assert captured["url"] == \
        "https://integrate.api.nvidia.com/v1/chat/completions"
    assert captured["payload"]["model"].startswith("deepseek-ai/")


def test_thinking_disabled_by_default(monkeypatch):
    """Reasoning tokens count against max_tokens and add nothing here."""
    import rag
    captured = {}
    monkeypatch.setattr(rag, "_post", lambda u, h, p, tries=5, timeout=120.0: (
        captured.update(p) or {"choices": [{"message": {"content": "x"}}]}))
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    rag.generate("q", [], backend="nvidia")
    assert captured["chat_template_kwargs"] == {"thinking": False}


def test_thinking_can_be_enabled(monkeypatch):
    import rag
    captured = {}
    monkeypatch.setattr(rag, "_post", lambda u, h, p, tries=5, timeout=120.0: (
        captured.update(p) or {"choices": [{"message": {"content": "x"}}]}))
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    rag.generate("q", [], backend="nvidia", thinking=True)
    assert captured["chat_template_kwargs"]["thinking"] is True


def test_reasoning_field_is_never_treated_as_the_answer(monkeypatch):
    """The scratchpad must not leak into the scored answer."""
    import rag
    monkeypatch.setattr(rag, "_post", lambda *a, **k: {"choices": [{"message": {
        "reasoning_content": "Let me look at passage 1 and think...",
        "content": "The docket is USCG-2026-0100 [1]."}}]})
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    out = rag.generate("q", [], backend="nvidia")
    assert out == "The docket is USCG-2026-0100 [1]."
    assert "Let me look" not in out


def test_all_budget_spent_on_reasoning_warns(monkeypatch, capsys):
    import rag
    monkeypatch.setattr(rag, "_post", lambda *a, **k: {"choices": [{"message": {
        "reasoning_content": "thinking forever", "content": ""}}]})
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    assert rag.generate("q", [], backend="nvidia") == ""
    assert "only reasoning" in capsys.readouterr().err


def test_max_tokens_is_passed_through(monkeypatch):
    import rag
    captured = {}
    monkeypatch.setattr(rag, "_post", lambda u, h, p, tries=5, timeout=120.0: (
        captured.update(p) or {"choices": [{"message": {"content": "x"}}]}))
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    rag.generate("q", [], backend="nvidia", max_tokens=3000)
    assert captured["max_tokens"] == 3000


def test_non_reasoning_provider_gets_no_extra_body(monkeypatch):
    import rag
    captured = {}
    monkeypatch.setattr(rag, "_post", lambda u, h, p, tries=5, timeout=120.0: (
        captured.update(p) or {"choices": [{"message": {"content": "x"}}]}))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    rag.generate("q", [], backend="groq")
    assert "chat_template_kwargs" not in captured


# --- network resilience ----------------------------------------------------

def test_post_retries_on_timeout(monkeypatch):
    """A free endpoint under load times out; that must not be fatal."""
    import rag
    import requests

    calls = {"n": 0}

    class Resp:
        status_code = 200
        headers = {}

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

        def raise_for_status(self):
            pass

    def flaky(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.exceptions.ReadTimeout("read timed out")
        return Resp()

    monkeypatch.setattr(rag, "_limiter", rag.RateLimiter(0))
    monkeypatch.setattr("requests.post", flaky)
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert rag._post("http://x", {}, {})["choices"][0]["message"]["content"] == "ok"
    assert calls["n"] == 3


def test_post_gives_up_after_repeated_timeouts(monkeypatch):
    import rag
    import requests
    import pytest

    def always_timeout(url, headers=None, json=None, timeout=None):
        raise requests.exceptions.ReadTimeout("read timed out")

    monkeypatch.setattr(rag, "_limiter", rag.RateLimiter(0))
    monkeypatch.setattr("requests.post", always_timeout)
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="gave up"):
        rag._post("http://x", {}, {}, tries=2)


def test_timeout_is_passed_to_requests(monkeypatch):
    import rag
    captured = {}

    class Resp:
        status_code = 200
        headers = {}

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

        def raise_for_status(self):
            pass

    def capture(url, headers=None, json=None, timeout=None):
        captured["timeout"] = timeout
        return Resp()

    monkeypatch.setattr(rag, "_limiter", rag.RateLimiter(0))
    monkeypatch.setattr("requests.post", capture)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    rag.generate("q", [], backend="nvidia", timeout=45.0)
    assert captured["timeout"] == 45.0


# --- Groq gpt-oss reasoning models -----------------------------------------

def test_gpt_oss_uses_max_completion_tokens(monkeypatch):
    """gpt-oss ignores max_tokens; sending it means no budget is applied."""
    import rag
    captured = {}
    monkeypatch.setattr(rag, "_post",
                        lambda u, h, p, tries=5, timeout=120.0: (
                            captured.update(p)
                            or {"choices": [{"message": {"content": "x"}}]}))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    rag.generate("q", [], model="openai/gpt-oss-120b", backend="groq")
    assert "max_tokens" not in captured
    assert captured["max_completion_tokens"] >= 2048


def test_gpt_oss_sets_low_reasoning_effort_by_default():
    import rag
    extra = rag.reasoning_payload("groq", "openai/gpt-oss-120b", 500, False)
    assert extra["reasoning_effort"] == "low"


def test_thinking_flag_raises_reasoning_effort():
    import rag
    extra = rag.reasoning_payload("groq", "openai/gpt-oss-120b", 500, True)
    assert extra["reasoning_effort"] == "medium"


def test_non_reasoning_groq_model_unchanged(monkeypatch):
    import rag
    captured = {}
    monkeypatch.setattr(rag, "_post",
                        lambda u, h, p, tries=5, timeout=120.0: (
                            captured.update(p)
                            or {"choices": [{"message": {"content": "x"}}]}))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    rag.generate("q", [], model="llama-3.3-70b-versatile", backend="groq")
    assert captured["max_tokens"] == 500
    assert "reasoning_effort" not in captured


def test_temperature_is_zero_for_reproducibility(monkeypatch):
    """The vendor sample uses 1; a benchmark needs deterministic output."""
    import rag
    captured = {}
    monkeypatch.setattr(rag, "_post",
                        lambda u, h, p, tries=5, timeout=120.0: (
                            captured.update(p)
                            or {"choices": [{"message": {"content": "x"}}]}))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    rag.generate("q", [], backend="groq")
    assert captured["temperature"] == 0


def test_internal_flag_never_sent_to_the_api(monkeypatch):
    import rag
    captured = {}
    monkeypatch.setattr(rag, "_post",
                        lambda u, h, p, tries=5, timeout=120.0: (
                            captured.update(p)
                            or {"choices": [{"message": {"content": "x"}}]}))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    rag.generate("q", [], model="openai/gpt-oss-120b", backend="groq")
    assert "_drop_max_tokens" not in captured


# --- answer matching across typographic variants ---------------------------

def test_non_breaking_hyphen_still_matches():
    """Models emit U+2011 inside identifiers; the source uses a plain hyphen."""
    assert contains_answer("The docket is USCG\u20112026\u20110861 [1].",
                           "USCG-2026-0861")


def test_en_dash_in_a_date_matches():
    assert contains_answer("Effective September 5\u20132026.",
                           "September 5, 2026")


def test_curly_apostrophe_matches():
    assert contains_answer("The Secretary\u2019s rule applies.",
                           "Secretary's")


def test_non_breaking_space_matches():
    assert contains_answer("A total of 139,175\u00a0acres.", "139,175 acres")


def test_genuinely_wrong_answer_still_fails():
    assert not contains_answer("The docket is USCG\u20112026\u20110862.",
                               "USCG-2026-0861")


# --- answer matching allows natural phrasing -------------------------------

def test_label_prefix_may_be_rephrased():
    """Expected comes from metadata with its label; models write a sentence."""
    assert contains_answer("The docket number is USCG\u20112026\u20111030.",
                           "Docket Number USCG-2026-1030")


def test_docket_no_abbreviation_matches():
    assert contains_answer("The docket number is PHMSA-2025-0092 (HM-268D).",
                           "Docket No. PHMSA-2025-0092 (HM-268D)")


def test_wrong_digit_still_fails():
    assert not contains_answer("The docket number is FAA-2026-8783.",
                               "Docket No. FAA-2026-8781")


def test_missing_word_fails():
    """Every expected token must appear somewhere, not just the digits."""
    assert not contains_answer("The number is 2026-0092.",
                               "Docket No. PHMSA-2025-0092")


def test_plain_date_unaffected():
    assert contains_answer("Comments are due by September 25, 2026.",
                           "September 25, 2026")


def test_wrong_date_fails():
    assert not contains_answer("Comments are due by September 24, 2026.",
                               "September 25, 2026")


def test_answer_without_digits_needs_exact_containment():
    assert contains_answer("The rule requires a unanimous vote.",
                           "Unanimous vote")
    assert not contains_answer("The vote was not unanimous in any sense.",
                               "Unanimous decision")


def test_identifier_prefix_kept_with_the_core():
    """'uscg 2026 1030' must stay whole, so a different agency code fails."""
    assert not contains_answer("The docket number is FAA-2026-1030.",
                               "Docket Number USCG-2026-1030")
