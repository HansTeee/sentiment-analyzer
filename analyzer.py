"""
analyzer.py - all model logic for the Sentiment + Emotion Analyzer.

This file knows nothing about Streamlit. Keeping the ML code separate from
the UI means we can test it from the command line (test_analyzer.py) and
swap the UI later without touching the models.
"""

from functools import lru_cache

from transformers import pipeline

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
# Both IDs were checked against the Hugging Face Hub (Phase 1).
#
# SENTIMENT: the short ID "distilbert-base-uncased-finetuned-sst-2-english"
# still works, but only because the Hub redirects it. The model now lives under
# the "distilbert" organisation, so we use the canonical name and avoid
# depending on a redirect that could disappear.
SENTIMENT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# WHY pin a revision (a git commit hash on the Hub)?
# If the model owner pushes new weights later, "latest" would silently change.
# Then the outputs quoted in the README would no longer match the live app.
# Pinning a commit means everyone gets exactly the model we tested.
SENTIMENT_REVISION = "714eb0fa89d2f80546fda750413ed43d93601a13"

# The emotion model's main branch (latest commit 0e1cd91, by the author) only
# has pytorch_model.bin. transformers then ALSO downloads a model.safetensors
# from an open Hugging Face bot PR (#3017, by SFconvertbot), so every cold
# start downloaded the same 328MB of weights twice.
# We pin directly to that bot commit (7e9d7fe), which sits on top of 0e1cd91
# and only adds model.safetensors. Phase 1 compared both files: all 105 weight
# tensors are bit-identical, so it's the same model in a safer format
# (safetensors can't run hidden code when loaded, which pickle-based .bin can).
EMOTION_REVISION = "7e9d7fe536a4d2bb472f9e883cc6cbbe81f68961"

# Both models are built on transformer architectures that can only look at
# 512 tokens at a time. Anything longer has to be cut off ("truncated").
MAX_TOKENS = 512


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_models():
    """
    Load and return (sentiment_pipeline, emotion_pipeline).

    It takes no arguments and doesn't print or write any files of its own.
    This matters because in Phase 2 Streamlit wraps it in @st.cache_resource,
    which caches the result based on the arguments. With no arguments, the
    function runs once and every later call reuses the same loaded models.

    (Hugging Face does save the weights to its disk cache on the first call.
    That's the library's own download cache, not state that we manage.)
    """
    try:
        sentiment_pipe = pipeline(
            "text-classification",
            model=SENTIMENT_MODEL,
            revision=SENTIMENT_REVISION,
        )
        emotion_pipe = pipeline(
            "text-classification",
            model=EMOTION_MODEL,
            revision=EMOTION_REVISION,
        )
    except Exception as e:
        # Re-raise with a message that tells the reader what to CHECK,
        # rather than just a stack trace from deep inside transformers.
        raise RuntimeError(
            "Could not load the Hugging Face models.\n"
            f"  Sentiment model: {SENTIMENT_MODEL} @ {SENTIMENT_REVISION[:10]}\n"
            f"  Emotion model:   {EMOTION_MODEL} @ {EMOTION_REVISION[:10]}\n"
            "Things to check:\n"
            "  1. Internet connection (the first run downloads ~600MB of weights).\n"
            "  2. The model IDs/revisions still exist at https://huggingface.co/<model-id>.\n"
            "  3. Free disk space for the Hugging Face cache (~/.cache/huggingface).\n"
            "  4. torch and transformers are installed: pip install -r requirements.txt\n"
            f"Original error: {type(e).__name__}: {e}"
        ) from e

    return sentiment_pipe, emotion_pipe


@lru_cache(maxsize=1)
def _default_models():
    """
    Fallback cache for when analyze() is called without models
    (for example from a script or test). lru_cache keeps the result in memory
    so the models load once, not on every call. Streamlit doesn't use this.
    It caches load_models() itself with @st.cache_resource.
    """
    return load_models()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _validate(text):
    """Reject empty input before it reaches a model."""
    # A model given "" will still return a label with a confident-looking
    # score. That's a made-up answer, so we refuse the input instead.
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text is empty. Please enter some text to analyze.")


def count_tokens(text, pipe):
    """
    Count how many tokens this pipeline's tokenizer turns `text` into,
    including special tokens like [CLS]/[SEP] (BERT) or <s>/</s> (RoBERTa).

    WHY count tokens, not characters or words? The 512 limit is in tokens.
    One word can be several tokens ("unbelievably" -> "un", "##bel", ...),
    so a character count can't tell us whether truncation happened.
    """
    # verbose=False silences the "sequence length is longer than..." warning.
    # Here a long sequence is expected: counting it is how we detect truncation.
    return len(pipe.tokenizer(text, truncation=False, verbose=False)["input_ids"])


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze_sentiment(text, pipe):
    """
    Return {"label": "POSITIVE" | "NEGATIVE", "score": float}.

    NOTE: this model has only two classes. There is no NEUTRAL, so a neutral
    sentence still gets forced into POSITIVE or NEGATIVE. The score is the
    model's probability for the label it picked, between 0.5 and 1.0.
    """
    _validate(text)
    # truncation=True: cut input to 512 tokens instead of crashing on long text.
    result = pipe(text, truncation=True)[0]
    return {"label": result["label"], "score": float(result["score"])}


def analyze_emotion(text, pipe):
    """
    Return ALL emotion scores, sorted from most to least likely:
        [{"label": "joy", "score": 0.79}, {"label": "surprise", "score": 0.14}, ...]

    WHY all scores and not just the top one? "joy 0.40, surprise 0.38" and
    "joy 0.99" are very different answers, and showing only "joy" hides that.
    """
    _validate(text)
    # top_k=None returns every label. The old option, return_all_scores=True,
    # is deprecated.
    results = pipe(text, top_k=None, truncation=True)
    # The pipeline already sorts, but we sort here anyway so our function's
    # promise doesn't depend on a library detail that could change.
    return sorted(
        ({"label": r["label"], "score": float(r["score"])} for r in results),
        key=lambda r: r["score"],
        reverse=True,
    )


def analyze(text, models=None):
    """
    Run both models and combine the results into one dict.

    `models` is the (sentiment_pipe, emotion_pipe) tuple from load_models().
    The Streamlit app passes its cached copy. Scripts can leave it out.

    On success returns:
        {
          "error": None,
          "sentiment": {"label": ..., "score": ...},
          "emotions": [ {"label": ..., "score": ...}, ... ],   # sorted desc
          "truncated": bool,     # True if EITHER model had to cut the text
          "token_count": int,    # longest token count across both tokenizers
          "max_tokens": 512,
        }
    On empty input returns {"error": "<message>"} and doesn't call the models.
    """
    # Check for empty input FIRST, before loading or running anything.
    try:
        _validate(text)
    except ValueError as e:
        return {"error": str(e)}

    sentiment_pipe, emotion_pipe = models if models is not None else _default_models()

    # The two models use DIFFERENT tokenizers (BERT WordPiece vs RoBERTa BPE),
    # so the same text can be e.g. 510 tokens for one and 515 for the other.
    # We check both and flag truncation if either one went over the limit.
    token_count = max(count_tokens(text, sentiment_pipe), count_tokens(text, emotion_pipe))

    return {
        "error": None,
        "sentiment": analyze_sentiment(text, sentiment_pipe),
        "emotions": analyze_emotion(text, emotion_pipe),
        "truncated": token_count > MAX_TOKENS,
        "token_count": token_count,
        "max_tokens": MAX_TOKENS,
    }
