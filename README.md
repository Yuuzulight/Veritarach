# Veritarach

Veritarach is a fine-tuned DeBERTa-v3-base binary classifier that tells AI-generated text
apart from human-written text. It's built to run as a Telegraph Protocol "Miner" serving the
`AI_DETECTION` intent, for Telegraph Hackathon Season 1 (Miner Track).

**Status: early scaffold.** The repo structure, data pipeline, and service skeleton are in
place; model training and protocol registration haven't happened yet (see Roadmap below for
what's blocking each).

## Why a fresh model instead of an existing detector

There are already open detectors out there (`roberta-base-openai-detector` and similar), but
they're tuned on GPT-2-era text and generalize poorly to what current-generation models
produce. Since the hackathon's validator ground truth will most likely look like modern LLM
output, Veritarach fine-tunes `microsoft/deberta-v3-base` from scratch on a mix of:

- **HC3** — paired human/ChatGPT answers to the same questions, which isolates writing style
  rather than topic.
- **Wikipedia** — additional human-text diversity beyond Q&A format.
- **Self-generated samples** from Claude, GPT-4o, and Gemini — to close the gap between
  HC3's GPT-3.5-era text and what current models actually write.

DeBERTa was picked over RoBERTa for its disentangled attention, which tends to help on
classification tasks like this one.

## Repo layout

```
registration/            Telegraph Miner registration files (see registration/README.md)
src/veritarach/
  config.py               Central settings (reads .env)
  data_pipeline/           Dataset fetching, generation, and assembly
  service/                 FastAPI app that serves predictions
scripts/                  Setup scripts
docs/superpowers/         Design specs and implementation plans
tests/                    Mirrors src/ layout; LLM providers are mocked, never called for real
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
scripts/setup_env.ps1
```

This creates a `.venv`, installs the non-torch dependencies, and prints the PyTorch install
command you need to run by hand — the CUDA and CPU builds are different commands, so the
script won't guess for you. Copy the exact one from
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) based on whether
you're training on a GPU instance or CPU.

You'll also need a `.env` file with:

```
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GEMINI_API_KEY=
```

These are only used by the data pipeline's self-generation step, and only spend real money
when you actually run it — nothing in the test suite touches the live APIs.

## Running the tests

```bash
pytest
```

## Roadmap

1. **Data pipeline** (in progress) — fetch HC3 + Wikipedia, generate modern-LLM samples,
   assemble the training set.
2. **Model training** — blocked on setting up a GPU rental account (Vast.ai); not needed
   until the dataset is ready.
3. **Service** (skeleton in progress) — FastAPI app; `/predict` returns 501 until a trained
   model is wired in.
4. **Protocol registration** — blocked on a few open questions in the Telegraph Discord
   (which Intent the hackathon's curated list actually includes, the node secret, where to
   get testnet MACHINA). See `registration/README.md` for the current state of each.
