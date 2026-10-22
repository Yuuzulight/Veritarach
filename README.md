# Veritarach

Veritarach is a fine-tuned DeBERTa-v3-base binary classifier that tells AI-generated text
apart from human-written text. It was deployed as a public inference service for Telegraph
Hackathon Season 1; that deployment has been shut down.

**Status: retired / decommissioned (9 Sep 2026).** The DigitalOcean droplet that served
`/predict` was deleted after Hackathon 1 ended, so there is no live endpoint any more. The
code, the data pipeline, the training pipeline, the `Dockerfile` and the results all remain
in this repo. The Telegraph Miner registration is no longer active in practice: the on-chain
entry (registrationId 85) was left in place, but the endpoint it points at no longer exists,
so nothing is being served.

> **Read the F1 figure with care.** The 99.86% test F1 is measured on a held-out split of
> Veritarach's *own* training distribution only. It is not a general AI-text detection
> accuracy. On Veracia's independent holdout, recall on the `ai_generated` class was **0.042**:
> the model labels most unseen content as human-written, with 0.98+ confidence.

The first retrain on the 96.6k-row dataset silently produced a degenerate checkpoint —
confidence never left roughly a 0.5-0.7 band even on completely unambiguous input, on a run
whose metrics were never saved anywhere to catch it. Root cause: the GPU instance's driver
only supported CUDA 12.4 while the pinned `torch` build defaulted to a CUDA 13 wheel, so
`torch.cuda.is_available()` was silently `False` and training proceeded without ever actually
using the GPU it thought it had. Retrained on an instance with a compatible driver and it
converged normally (test_metrics.json and training_log.json now get written alongside every
checkpoint specifically so this doesn't go unnoticed again). One real limitation the retrain
didn't fix: the model discriminates its own training distribution (HC3-style Q&A, Wikipedia,
the self-generated samples) very well, but confidently misclassifies AI-generated text in
other registers — prose paragraphs, casual notes, anything that doesn't look like an HC3
answer — as human-written. That gap was never addressed (see the note above).

> Built as an entry for Telegraph Hackathon Season 1 — it ran as a Telegraph Protocol
> "Miner" serving the `AI_TEXT_DETECTION` intent until the deployment was retired. See
> `registration/` for that integration.

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

A `Dockerfile` is included for containerized deployment. (The hosted instance that used it
is gone; see Status.)

## Roadmap

1. **Data pipeline** — done. HC3 + Wikipedia + self-generated samples (Claude/GPT-4o/Gemini)
   assembled into a 96.6k-row training set.
2. **Model training** — done. Fine-tuned on a rented cloud GPU; 99.86% test F1 (in-distribution
   only, see Status), retrained after the first attempt's driver/CUDA mismatch went undetected.
3. **Service** — done. FastAPI app with a real `/predict`, backed by the trained checkpoint.
4. **Deployment** — done, then retired. Ran on a DigitalOcean droplet behind Caddy with HTTPS
   via Let's Encrypt until the droplet was deleted on 9 Sep 2026, after Hackathon 1 ended.
5. **Protocol registration** — done, no longer active. Registered as a Telegraph Miner for the
   `AI_TEXT_DETECTION` intent and confirmed active at the time; the endpoint has since been
   taken down. See `registration/README.md` for the full flow.
