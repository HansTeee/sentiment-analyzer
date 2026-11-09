"""
app.py - Streamlit UI for the Sentiment + Emotion Analyzer.

All model logic lives in analyzer.py. This file only handles layout and display.

How Streamlit works: on EVERY interaction (button click, editing the text box)
it re-runs this whole file from top to bottom. Two things make that workable:
  * @st.cache_resource, so the models load once instead of on every rerun
  * st.session_state, so values (like the text box contents) survive reruns
"""

import pandas as pd
import streamlit as st

from analyzer import MAX_TOKENS, analyze, count_tokens, load_models

# Below this sentiment score we show a "low confidence" caveat.
# The score is between 0.5 and 1.0 (two classes), so 0.7 is fairly unsure.
LOW_CONFIDENCE = 0.7

# One-click examples: (button label, text)
EXAMPLES = [
    ("😀 Positive", "I absolutely love this phone, the camera is stunning and the battery lasts all day."),
    ("😠 Negative", "This was the worst customer service I have ever experienced. I am furious."),
    ("😐 Neutral", "The package arrived on Tuesday."),
    ("🙄 Sarcastic", "Oh great, another meeting."),
    ("🤔 Mixed", "The food was delicious but the service was painfully slow."),
    ("🇵🇭 Taglish", "Ang ganda ng service, sobrang bilis!"),
]

# set_page_config must be the first Streamlit command in the script.
st.set_page_config(page_title="Sentiment & Emotion Analyzer", page_icon="🎭", layout="centered")


def pct(score):
    """Format 0.9998 as '>99.9%', not '100%'. A model is never 100% sure."""
    return ">99.9%" if score >= 0.9995 else f"{score:.1%}"


@st.cache_resource(show_spinner=False)
def get_models():
    # cache_resource (not cache_data): models are big objects we can't
    # serialize, and they should be shared by all users, not copied per user.
    return load_models()


# --- Header + expectations ---------------------------------------------------
st.title("🎭 Sentiment & Emotion Analyzer")
st.write("Paste English text to see whether it reads as positive or negative, and which emotions it expresses.")

# Up near the top, so people read it before they type.
with st.expander("ℹ️ What this can and can't do"):
    st.markdown(
        """
**Can:**
- Classify English text as **POSITIVE** or **NEGATIVE**
- Score 7 emotions: anger, disgust, fear, joy, neutral, sadness, surprise

**Can't:**
- **Say "neutral" for sentiment.** The sentiment model only has two classes, so a plain fact
  ("The package arrived on Tuesday.") still gets labelled positive or negative, often with
  *high* confidence. Check the emotion model's *neutral* score as well.
- **Detect sarcasm.** "Oh great, another meeting." scores as very positive.
- **Understand Tagalog or other non-English text.** Results look confident but are unreliable.
- **Read very long text.** Only the first 512 tokens (roughly 350 to 400 words) are analyzed.
        """
    )

# --- Load models (cached after the first run) -------------------------------
try:
    with st.spinner("Loading models... the first load can take a minute."):
        models = get_models()
except RuntimeError as e:
    st.error(str(e))
    st.stop()  # Nothing else can work without the models.

# --- Input ------------------------------------------------------------------
# The text box gets its value from st.session_state["text"] (via key="text").
# Setting the default here once means example buttons can overwrite it later.
if "text" not in st.session_state:
    st.session_state["text"] = EXAMPLES[0][1]


def use_example(example_text):
    # Runs as a button callback, BEFORE the rerun, so the text box
    # shows the new value when the script runs again.
    st.session_state["text"] = example_text


st.caption("Try an example:")
cols = st.columns(3)
for i, (label, example_text) in enumerate(EXAMPLES):
    cols[i % 3].button(label, on_click=use_example, args=(example_text,), width="stretch")

text = st.text_area("Text to analyze", key="text", height=150)

# Character + token count. Tokens are what the 512 limit counts, so show both.
if text.strip():
    sentiment_pipe, emotion_pipe = models
    n_tokens = max(count_tokens(text, sentiment_pipe), count_tokens(text, emotion_pipe))
    over = " ⚠️ over the limit, the end will be ignored" if n_tokens > MAX_TOKENS else ""
    st.caption(f"{len(text)} characters · {n_tokens} / {MAX_TOKENS} tokens{over}")
else:
    st.caption(f"0 characters · 0 / {MAX_TOKENS} tokens")

# --- Analyze ----------------------------------------------------------------
if st.button("Analyze", type="primary"):
    with st.spinner("Analyzing..."):
        result = analyze(text, models)

    if result["error"]:
        # Empty input: a friendly message, not a crash.
        st.info("✏️ Type or paste some text first, or click one of the examples above.")
        st.stop()

    if result["truncated"]:
        st.warning(
            f"Your text is {result['token_count']} tokens long. Only the first {MAX_TOKENS} were "
            "analyzed, so anything after that point had no effect on these results."
        )

    # --- Sentiment ---
    st.subheader("Sentiment")
    sentiment = result["sentiment"]
    emotions = result["emotions"]
    icon = "😀" if sentiment["label"] == "POSITIVE" else "😠"
    st.markdown(f"### {icon} {sentiment['label']} · {pct(sentiment['score'])} confidence")
    st.progress(sentiment["score"])

    # The caveat: two triggers, because confidence alone isn't enough.
    #   1. Low score: the model itself is unsure.
    #   2. The emotion model's top label is "neutral". In testing, the sentiment
    #      model gave neutral facts scores as high as 98%, so a high score does
    #      NOT mean the text has real sentiment.
    low_confidence = sentiment["score"] < LOW_CONFIDENCE
    looks_neutral = emotions[0]["label"] == "neutral"
    if low_confidence or looks_neutral:
        reasons = []
        if low_confidence:
            reasons.append(f"the confidence is only {sentiment['score']:.0%}")
        if looks_neutral:
            reasons.append(f"the emotion model rates this text as mostly **neutral** ({emotions[0]['score']:.0%})")
        st.warning(
            "**Treat this label with caution:** " + " and ".join(reasons) + ". "
            "This sentiment model has no NEUTRAL class, so neutral text still gets labelled "
            "positive or negative. The text may not be positive or negative at all."
        )

    # --- Emotion ---
    st.subheader("Emotion")
    top = emotions[0]
    st.markdown(f"Top emotion: **{top['label']}** ({pct(top['score'])})")

    # Show the whole distribution, not just the winner. A dataframe with a
    # progress column keeps the sorted order (st.bar_chart re-sorts labels A to Z).
    df = pd.DataFrame(emotions)
    df["score"] = df["score"] * 100
    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        column_config={
            "label": st.column_config.TextColumn("Emotion"),
            "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f%%"),
        },
    )
