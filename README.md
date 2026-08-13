# Veritarach

Veritarach is a fine-tuned DeBERTa-v3-base binary classifier that tells AI-generated text
apart from human-written text, deployed as a live, publicly reachable inference service.

**Status: live.** 99.65% test F1 on a held-out split, deployed behind real HTTPS, and
actively serving predictions. See Roadmap below.

> Built as an entry for Telegraph Hackathon Season 1 — deployed as a Telegraph Protocol
> "Miner" serving the `AI_TEXT_DETECTION` intent. See `registration/` for that integration.

## Why a fresh model instead of an existing detector

There are already open detectors out there (`roberta-base-openai-detector` and similar), but
they're tuned on GPT-2-era text and generalize poorly to what current-generation models
produce. Since real-world ground truth is far more likely to look like modern LLM output,
Veritarach fine-tunes `microsoft/deberta-v3-base` from scratch on a mix of:

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

## Running the service

`/predict` lazily loads the checkpoint from `data/model/final/` on first request (that
directory is gitignored — train locally or drop a checkpoint in before running):

```bash
uv run uvicorn veritarach.service.app:app --reload
curl -X POST localhost:8000/predict -H "content-type: application/json" \
  -d '{"text": "some text to classify"}'
```

If no checkpoint is present, `/predict` returns 501 instead of crashing — that's the
expected state in CI, since the trained weights are never committed.

A `Dockerfile` is included for containerized deployment.

## Roadmap

1. **Data pipeline** — done. HC3 + Wikipedia + self-generated samples (Claude/GPT-4o/Gemini)
   assembled into a 96.6k-row training set.
2. **Model training** — done. Fine-tuned on a rented cloud GPU; 99.65% test F1.
3. **Service** — done. FastAPI app with a real `/predict`, backed by the trained checkpoint.
4. **Deployment** — done. Live on a DigitalOcean droplet behind Caddy, real HTTPS via Let's
   Encrypt.
5. **Protocol registration** — done. Registered as a Telegraph Miner for the
   `AI_TEXT_DETECTION` intent and confirmed active. See `registration/README.md` for the
   full flow.
