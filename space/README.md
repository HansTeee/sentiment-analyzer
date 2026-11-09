---
title: Sentiment And Emotion Analyzer
emoji: 🎭
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.67.0
python_version: "3.13"
app_file: app.py
pinned: false
license: mit
short_description: English sentiment + 7-emotion analysis, with honest caveats
models:
  - distilbert/distilbert-base-uncased-finetuned-sst-2-english
  - j-hartmann/emotion-english-distilroberta-base
preload_from_hub:
  - distilbert/distilbert-base-uncased-finetuned-sst-2-english config.json,model.safetensors,tokenizer_config.json,vocab.txt 714eb0fa89d2f80546fda750413ed43d93601a13
  - j-hartmann/emotion-english-distilroberta-base config.json,model.safetensors,merges.txt,special_tokens_map.json,tokenizer.json,tokenizer_config.json,vocab.json 7e9d7fe536a4d2bb472f9e883cc6cbbe81f68961
---

# 🎭 Sentiment & Emotion Analyzer

Paste English text to get:

- **Sentiment:** POSITIVE or NEGATIVE, with confidence (the model has no NEUTRAL class)
- **Emotion:** scores for all 7 labels: anger, disgust, fear, joy, neutral, sadness, surprise

**Read the "What this can and can't do" box in the app before trusting a result.** In testing, the
sentiment model labelled the plain fact *"The package arrived on Tuesday."* POSITIVE at 98.3%, read
sarcasm literally (*"Oh great, another meeting."* → POSITIVE >99.9%), and gave confident but
meaningless answers on Tagalog.

Models: [DistilBERT SST-2](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) ·
[Emotion English DistilRoBERTa](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)

Source code, full test results and limitations: see the GitHub repository.
