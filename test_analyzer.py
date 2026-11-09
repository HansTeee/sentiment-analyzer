"""
test_analyzer.py - tests that use REAL sentences and print the real output.

Run either way:
    python test_analyzer.py        (plain script, prints a report)
    python -m pytest -s            (-s shows the prints; pytest hides them otherwise)

Two kinds of checks:
  * Clear-cut cases (obviously positive/negative, empty, too long) get ASSERTS.
    If these fail, something is broken.
  * Ambiguous cases (neutral, sarcasm, mixed, Tagalog) are PRINT-ONLY. There's
    no "correct" label to assert. The point is to see what the model actually
    does so the README can describe it honestly.
"""

from analyzer import MAX_TOKENS, _default_models, analyze

# The 10 test sentences. The numbers are referenced in the README.
CASES = {
    1: ("clearly positive", "I absolutely love this phone, the camera is stunning and the battery lasts all day."),
    2: ("clearly negative", "This was the worst customer service I have ever experienced. I am furious."),
    3: ("neutral", "The package arrived on Tuesday."),
    4: ("sarcastic", "Oh great, another meeting."),
    5: ("mixed", "The food was delicious but the service was painfully slow."),
    6: ("negation", "The movie was not bad at all."),
    7: ("Taglish", "Ang ganda ng service, sobrang bilis!"),
    8: ("pure Tagalog (negative)", "Ang pangit ng serbisyo, sobrang bagal at bastos pa ang staff."),
    # 9 (long) and 10 (empty) are built in their own tests below.
}

# Long input: about 1,050 tokens of praise, THEN a negative sentence. The model
# only sees the first 512 tokens, so it never sees the complaint at the end.
# This shows the real cost of truncation, not just that the flag gets set.
LONG_TEXT = ("The hotel was wonderful and the staff were friendly. " * 100
             + "Actually, everything was terrible and I want a refund.")


def show(num, kind, text, result):
    """Print one result in a readable block."""
    preview = text if len(text) <= 80 else text[:77] + "..."
    print(f"\n[{num}] {kind}: {preview!r}")
    if result["error"]:
        print(f"    ERROR: {result['error']}")
        return
    s = result["sentiment"]
    print(f"    sentiment: {s['label']} ({s['score']:.4f})")
    emo = ", ".join(f"{e['label']} {e['score']:.3f}" for e in result["emotions"])
    print(f"    emotions:  {emo}")
    print(f"    tokens: {result['token_count']}/{result['max_tokens']}  truncated: {result['truncated']}")


def run_case(num):
    kind, text = CASES[num]
    result = analyze(text)
    show(num, kind, text, result)
    return result


# --- Clear-cut cases: assert ------------------------------------------------
def test_clearly_positive():
    r = run_case(1)
    assert r["sentiment"]["label"] == "POSITIVE"
    assert r["sentiment"]["score"] > 0.9


def test_clearly_negative():
    r = run_case(2)
    assert r["sentiment"]["label"] == "NEGATIVE"
    assert r["sentiment"]["score"] > 0.9


def test_long_input_is_truncated():
    r = analyze(LONG_TEXT)
    show(9, "long input (>512 tokens)", LONG_TEXT, r)
    assert r["error"] is None          # didn't crash
    assert r["truncated"] is True      # the UI will be told to warn
    assert r["token_count"] > MAX_TOKENS


def test_empty_string():
    for empty in ["", "   ", "\n\t"]:
        r = analyze(empty)
        show(10, "empty/whitespace", empty, r)
        assert r["error"] is not None
        assert "sentiment" not in r     # the models weren't called


def test_short_input_not_flagged():
    r = analyze("Hello there.")
    assert r["truncated"] is False


def test_emotions_are_complete_and_sorted():
    r = analyze(CASES[1][1])
    labels = {e["label"] for e in r["emotions"]}
    # All 7 labels found in Phase 1 come back, not just the top one.
    assert labels == {"anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"}
    scores = [e["score"] for e in r["emotions"]]
    assert scores == sorted(scores, reverse=True)
    assert abs(sum(scores) - 1.0) < 1e-3   # softmax: probabilities sum to 1


def test_model_labels_match_phase1():
    """If someone changes the model, this fails loudly instead of the UI breaking quietly."""
    sentiment_pipe, emotion_pipe = _default_models()
    assert set(sentiment_pipe.model.config.id2label.values()) == {"NEGATIVE", "POSITIVE"}
    assert len(emotion_pipe.model.config.id2label) == 7


# --- Ambiguous cases: print only, no label assertions -----------------------
def test_neutral():
    run_case(3)


def test_sarcastic():
    run_case(4)


def test_mixed():
    run_case(5)


def test_negation():
    run_case(6)


def test_taglish():
    run_case(7)


def test_pure_tagalog():
    run_case(8)


# --- Extra probes: print-only, quoted in the README limitations section -----
# Short output (one line each) for checking a specific behaviour.
PROBES = [
    # More neutral facts. How confident is a model with no NEUTRAL class?
    "The meeting is at 3pm.",
    "The report has twelve pages.",
    "Water boils at 100 degrees Celsius.",
    "I took the bus to work today.",
    # Tagalog without English words. Does the model understand, or is it luck?
    "Ang ganda ng serbisyo, sobrang bilis!",   # "The service is great, so fast!"
    "Ang pangit ng serbisyo, sobrang bagal!",  # "The service is awful, so slow!"
    "Masaya ako ngayon.",                      # "I am happy today."
    "Malungkot ako ngayon.",                   # "I am sad today."
]


def test_probes():
    print("\n--- extra probes ---")
    for text in PROBES:
        r = analyze(text)
        s, top = r["sentiment"], r["emotions"][0]
        print(f"    {s['label']:8} {s['score']:.3f} | top emotion: {top['label']} {top['score']:.3f} | {text}")


if __name__ == "__main__":
    # Plain-script mode: run each test function in sentence order.
    tests = [test_clearly_positive, test_clearly_negative, test_neutral, test_sarcastic,
             test_mixed, test_negation, test_taglish, test_pure_tagalog,
             test_long_input_is_truncated, test_empty_string, test_short_input_not_flagged,
             test_emotions_are_complete_and_sorted, test_model_labels_match_phase1, test_probes]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"    FAILED: {t.__name__} {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
