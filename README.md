# 🎭 Sentiment & Emotion Analyzer

Classifies English text as positive or negative and scores it across 7 emotions, and is upfront about where the models get it wrong.

### ▶️ [Try the live demo](https://huggingface.co/spaces/HansTee89411/sentiment-emotion-analyzer)

![Screenshot of the app analyzing a sentence](docs/screenshot.png)

<!-- TODO: add docs/screenshot.png -->

> Hosted on a free Hugging Face Space. If nobody has used it recently, the first load takes longer while the Space wakes up.

---

## What it does

| Task | Model output | Labels |
|---|---|---|
| **Sentiment** | 2 classes, with a confidence score | `POSITIVE`, `NEGATIVE` (no neutral) |
| **Emotion** | Probabilities for all 7 classes, summing to 1 | `anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise` |

The labels above were read from each model's `config.id2label` at runtime, not copied from the model cards.

The app shows the **full** emotion distribution rather than just the top label. *"joy 41%, surprise 38%"* and *"joy 99%"* are very different answers, and showing only "joy" would hide that.

## Models

| | Model | Why this one |
|---|---|---|
| Sentiment | [`distilbert/distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) | DistilBERT is ~40% smaller than BERT and runs fast on the free 2-vCPU Spaces tier. Fine-tuned on SST-2 (movie-review sentences). It's the default sentiment model in 🤗 Transformers, so its behaviour is widely documented. |
| Emotion | [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) | Also a distilled (small) model, trained on 6 English emotion datasets. It includes a `neutral` class, which the sentiment model lacks. The app uses that to flag unreliable sentiment labels (see below). |

Both models are **pinned to exact commits** in [`analyzer.py`](analyzer.py), so the outputs quoted in this README can be reproduced even if the model owners push new weights later.

<details>
<summary>Why the emotion model is pinned to a bot commit</summary>

The emotion model's latest author commit (`0e1cd91`) only ships `pytorch_model.bin`. When loading it, `transformers` 5.x also downloaded a `model.safetensors` from an open, auto-generated pull request (#3017, by Hugging Face's `SFconvertbot`). That meant every cold start downloaded the same 328 MB of weights twice. The app pins directly to that bot commit (`7e9d7fe`), which only adds the safetensors file. I compared every tensor: all 105 weight tensors are bit-identical to the author's `.bin`.
</details>

## Run locally

Requires Python 3.11+ (tested on 3.13).

```bash
git clone https://github.com/HansTeee/sentiment-analyzer.git
cd sentiment-analyzer
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads ~570 MB of model weights into the Hugging Face cache. After that, startup takes a few seconds.

Run the tests (they print every model output, so you can see the behaviour for yourself):

```bash
python test_analyzer.py
```

`requirements.txt` installs the **CPU-only** build of PyTorch (via `--extra-index-url`). The default build pulls in ~2 GB of CUDA libraries that a free CPU Space never uses.

## Architecture

```
app.py  (Streamlit UI: layout and display only)
  │  get_models()  ← @st.cache_resource, runs once per server process
  │  analyze(text, models)
  ▼
analyzer.py  (all model logic, no Streamlit imports)
  ├─ load_models()        → two transformers pipelines, pinned revisions
  ├─ analyze_sentiment()  → {"label", "score"}
  ├─ analyze_emotion()    → all 7 scores, sorted
  └─ analyze()            → validates input, counts tokens, combines both
```

- **Separation:** `analyzer.py` knows nothing about the UI, so it can be tested from the command line and the UI could be replaced without touching model code.
- **Streamlit's rerun model:** Streamlit re-executes `app.py` from top to bottom on *every* interaction. Without caching, both models would reload from disk each time a button is clicked.
- **`@st.cache_resource`** keeps one copy of the loaded pipelines in memory, shared across all users. (`@st.cache_data` would be wrong here: it's meant for serializable data, and it copies values for each caller.) `load_models()` takes no arguments, so there's exactly one cache entry.
- **`st.session_state`** holds the text box contents so the example buttons can fill it in. The buttons use an `on_click` callback, which runs *before* the rerun, so the new text appears immediately.
- **Input checks happen before the models run:** empty or whitespace-only input returns an error without calling a model. Input is counted in *tokens* with each model's own tokenizer (they differ), because the 512 limit is measured in tokens, not characters.
- **Build-time preloading:** the Space config uses `preload_from_hub` to bake the weights into the Space image, so a sleeping Space doesn't have to re-download them when it wakes.

## Limitations

These are real outputs from [`test_analyzer.py`](test_analyzer.py), using the pinned model revisions.

### 1. There is no NEUTRAL sentiment, and the confidence score won't warn you

The sentiment model can only answer POSITIVE or NEGATIVE, so neutral text gets forced into one of them. I expected neutral sentences to land near 50% confidence. **They don't:**

| Input | Sentiment | Emotion model's top label |
|---|---|---|
| *"The package arrived on Tuesday."* | **POSITIVE 0.983** | neutral 0.589 |
| *"The meeting is at 3pm."* | POSITIVE 0.935 | neutral 0.748 |
| *"Water boils at 100 degrees Celsius."* | **NEGATIVE 0.945** | neutral 0.568 |
| *"The report has twelve pages."* | POSITIVE 0.570 | neutral 0.839 |
| *"I took the bus to work today."* | NEGATIVE 0.576 | fear 0.426 |

A plain fact about boiling water is "94.5% negative". A low-confidence check alone (say, below 70%) would catch only 2 of these 5.

**What the app does about it:** it shows a caveat when confidence is below 70% *or* when the emotion model's top label is `neutral`. The neutral rule catches 4 of the 5 sentences above, and the low-confidence rule catches the bus sentence. It's still a heuristic, not a fix: it relies on a second model being right, and a neutral sentence that is confidently labelled *and* read as an emotion by the emotion model would get through.

### 2. Sarcasm is read literally

| Input | Sentiment | Top emotions |
|---|---|---|
| *"Oh great, another meeting."* | **POSITIVE 0.9996** | joy 0.901, neutral 0.052, surprise 0.028 |

Both models agree it's happy, with near-total confidence. No caveat fires, because neither model is unsure. The "can't do" box in the app mentions sarcasm before the user types anything, because nothing in the output can flag it.

### 3. The models read word by word, not the whole sentence

| Input | Sentiment | Top emotion |
|---|---|---|
| *"The food was delicious but the service was painfully slow."* | NEGATIVE 0.9966 | disgust 0.639 |
| *"The movie was not bad at all."* | POSITIVE 0.998 ✅ | **disgust 0.908** ❌ |

The sentiment model handles the negation in *"not bad at all"* correctly. The emotion model sees "bad" and reports 91% disgust. The mixed review is reduced to a single label with no sign of the positive half.

### 4. English only: Tagalog results are confident and meaningless

The test sentence happens to come out right:

| Input | Sentiment | Top emotion |
|---|---|---|
| *"Ang ganda ng service, sobrang bilis!"* ("The service is great, so fast!") | POSITIVE 0.956 | neutral 0.658 |

But swapping one word shows it's luck, not understanding:

| Input | Meaning | Sentiment |
|---|---|---|
| *"Ang ganda ng serbisyo, sobrang bilis!"* | The service is great, so fast! | POSITIVE 0.923 |
| *"Ang pangit ng serbisyo, sobrang bagal!"* | The service is awful, so slow! | **POSITIVE 0.926** ❌ |
| *"Masaya ako ngayon."* | I am happy today. | **NEGATIVE 0.918** ❌ |
| *"Malungkot ako ngayon."* | I am sad today. | NEGATIVE 0.988 |

Opposite meanings get the same label with the same confidence. The emotion model rates every Tagalog sentence as mostly `neutral` (0.66 to 0.78), which is effectively "I can't read this". That's more honest than the sentiment model, but neither result is usable. **High confidence does not mean the model understood the input.**

### 5. Only the first 512 tokens are read

Both models have a 512-token window, and longer input is cut off. The test builds a ~1,000-token hotel review that is glowing for 100 sentences and then ends with *"Actually, everything was terrible and I want a refund."*

```
sentiment: POSITIVE (0.9999)
emotions:  joy 0.912, neutral 0.066, surprise 0.008, ...
tokens: 1014/512  truncated: True
```

The model never sees the complaint. The app warns when this happens and shows a live token count under the text box, but it can't recover the lost text. (512 tokens is roughly 350 to 400 English words; the exact number depends on the words.)

### 6. The emotion labels are fixed and English-centric

There are exactly 7 labels (Ekman's 6 basic emotions plus neutral). There's no label for frustration, sarcasm, gratitude, anxiety, or excitement, so every input is forced into those 7. They come from English-language training data. Emotion words and how emotions are expressed don't map neatly across languages and cultures, so even a translated input would be judged by English norms.

### 7. Cold starts on the free tier

Free Spaces go to sleep after a period of inactivity, and the next visitor waits while the Space restarts. Measured locally:

| Stage | Time |
|---|---|
| Python imports (torch, transformers) | ~5 s |
| Load both models from disk | ~2.5 s |
| First prediction | ~0.2 s |
| **Total to first prediction, weights already on disk** | **~7.5 s** |
| Downloading weights from scratch (home connection) | 3.5 to 8 min |

Because the Space preloads the weights at build time, waking up should cost roughly the first number, not the second. The free tier's 2 vCPUs are slower than my laptop, so expect more than 7.5 s. I haven't measured it on the live Space yet.

## What I learned

1. **Verify, don't assume.** The sentiment model ID from most tutorials only works through a Hub redirect, since the model now lives under the `distilbert/` org. And the emotion model was being downloaded twice, which only showed up when I measured the cache size.
2. **A confidence score isn't a correctness score.** I expected neutral text to get ~50% confidence. It got up to 98%. A softmax over two classes has to put the probability somewhere, so "how sure" and "how right" are separate questions.
3. **Test the failure cases and keep the results.** The clear positive and negative sentences were boring. Everything worth writing about came from the ambiguous ones, and from changing one word to check whether a "correct" answer was luck.
4. **Deployment details matter as much as the model.** Spaces defaults to Python 3.10, but the pinned PyTorch only ships wheels for 3.11+. CPU-only torch saves ~2 GB per build. Spaces no longer offers persistent storage, so weights downloaded at runtime aren't guaranteed to survive a restart. None of this shows up when running locally.

## What I'd do next

- **Fine-tune a Tagalog / Taglish sentiment model.** Filipino has tens of millions of speakers but far less NLP tooling than that would suggest, and code-switched Taglish is how a lot of Filipino online text is written. Concretely:
  1. Start from a multilingual or Filipino base model. [`jcblaise/roberta-tagalog-base`](https://huggingface.co/jcblaise/roberta-tagalog-base) was pre-trained on Tagalog text. [`FacebookAI/xlm-roberta-base`](https://huggingface.co/FacebookAI/xlm-roberta-base) covers Tagalog and English in one model, which suits Taglish.
  2. Fine-tune on labelled Filipino reviews or tweets. Existing public datasets are small, so part of the work would be hand-labelling a few hundred Taglish sentences, including the ones in this README.
  3. Evaluate on a held-out Taglish set, comparing against this English model's results above. That's the baseline to beat, and the one-word-swap pairs make a useful sanity test.
- **Add a real neutral class** by switching sentiment to a 3-class model (e.g. one trained on tweets with positive/neutral/negative labels) and re-running the same test table to compare.
- **Handle long text by chunking:** split into 512-token windows, score each, and show how sentiment changes through the document, instead of silently dropping the end.
- **Measure the live Space's cold start** and add the real number to this README.

## License

MIT, see [LICENSE](LICENSE). Each model has its own license on its Hugging Face model card.
