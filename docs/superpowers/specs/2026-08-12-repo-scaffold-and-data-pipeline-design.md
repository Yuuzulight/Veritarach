# Veritarach: Repo Scaffold + Data Pipeline + Service Skeleton

Date: 2026-08-12

## Purpose

Veritarach is a fine-tuned DeBERTa-v3-base binary text classifier (ai_generated vs
human_written) being built and registered as a Telegraph Protocol "Miner" for the
AI_DETECTION intent, for Telegraph Hackathon Season 1 (Miner Track). No repo exists yet.
This spec covers the first two independent, unblocked tracks: the project scaffold plus
Phase 1 data pipeline, and a Phase 0 service/registration skeleton. Model training and
actual registration are out of scope here — training needs a Vast.ai account (not yet set
up) and registration needs Discord answers that haven't come back yet.

## Repo structure

```
Veritarach/
├── .env                        # API keys, gitignored
├── .gitignore
├── pyproject.toml
├── README.md
├── scripts/
│   └── setup_env.ps1           # ported from the laptop-migration bundle's env-setup script
├── registration/
│   ├── miner.yaml               # Phase 0 stub, blocked fields commented
│   └── README.md                 # plain-language summary of what's blocked and why
├── data/                        # created at runtime by pipeline scripts; fully gitignored
├── src/veritarach/
│   ├── config.py                 # pydantic-settings, single source of truth for env vars
│   ├── data_pipeline/
│   │   ├── fetch_hc3.py
│   │   ├── fetch_wikipedia.py
│   │   ├── providers/
│   │   │   ├── base.py           # shared generation interface
│   │   │   ├── claude.py
│   │   │   ├── openai_gpt.py
│   │   │   └── gemini.py
│   │   ├── generate_llm_samples.py  # orchestrates providers + strategies
│   │   └── build_dataset.py
│   └── service/
│       ├── app.py
│       └── schemas.py
├── Dockerfile
└── tests/
    ├── conftest.py                # mocks provider clients — zero real spend in test runs
    ├── data_pipeline/
    └── service/
```

Tooling: `uv` for env/deps. Core dependencies: `transformers`, `datasets`, `accelerate`,
`scikit-learn` (ML stack); `fastapi`, `uvicorn`, `pydantic`, `pydantic-settings` (service);
`anthropic`, `openai`, `google-generativeai` (generation); `pytest` (tests).

`data/raw` and `data/generated` are not pre-scaffolded as empty directories — git doesn't
track empty dirs, and pipeline scripts create them on demand via `mkdir`.

## Data pipeline

**Sources:**
- HC3 (`Hello-SimpleAI/HC3`) — paired human/ChatGPT Q&A, gives topic-matched pairs.
- Wikipedia — additional human-text diversity beyond Q&A style.
- Self-generated LLM text (Claude, GPT-4o, Gemini) — covers the modern-LLM gap HC3 misses,
  since HC3 is GPT-3.5/GPT-2-era.

**Generation budget** (~$5-10 total cap, weighted toward Gemini's free tier to stretch the
paid portion further):

| Provider | Samples | Cost |
|---|---|---|
| Gemini | 150 | free tier |
| Claude | 75 | paid |
| GPT-4o | 75 | paid |

**Generation strategy**, 70/30 split per provider:
- Paired (70%): answer the same HC3 question a human answered, for topic-matched pairs that
  isolate style rather than subject matter.
- Topic-diverse (30%): fixed list of formats (essay, product review, news blurb, technical
  explanation, short story) not tied to HC3, for generalization beyond Q&A style.

**Unified dataset schema** (`build_dataset.py` output, JSONL):
```
text, label (ai_generated | human_written), source (hc3 | wikipedia | claude | gpt4o | gemini),
generation_strategy (paired | topic | null), pair_id (nullable), split
```

**Class balance:** AI class = HC3's AI-answer half + all self-generated samples. Human class
= HC3's human-answer half + Wikipedia, downsampled to match the AI class count so Wikipedia
volume doesn't skew the set human-heavy.

**Split strategy:** 80/10/10 train/val/test, group-aware — a paired HC3 question or
topic-diverse prompt is assigned to exactly one split as a group, so a human/AI pair never
splits across train and test (which would leak topic information and inflate eval scores).

**Error handling:** each provider call is idempotent against a manifest file — a re-run
skips samples already generated instead of re-spending budget. Retries use exponential
backoff on rate limits; a sample that fails repeatedly is logged and skipped rather than
crashing the run. Cumulative cost is tracked against the budget cap and the pipeline
hard-stops before exceeding it.

## Service skeleton

`service/app.py` — FastAPI app with two endpoints:
- `GET /health` — returns 200 immediately. This matters beyond a formality: per project
  notes, the live protocol runs unpaid spot-checks roughly every 20s, and a free-tier cold
  start causing repeated failures triggers Routing_Revocation. A real health check surfaces
  that reliability risk in testing before it's a live problem.
- `POST /predict` — takes `{text: str}`, returns `{label, confidence}`, but returns **HTTP
  501** until the trained model is wired in (Phase 2). A stub that silently returned a fake
  prediction would let later integration testing pass against garbage without anyone
  noticing the model isn't there yet.

`service/schemas.py`:
```
PredictRequest:  { text: str }
PredictResponse: { label: "ai_generated" | "human_written", confidence: float }
```

`Dockerfile` (repo root) — stub for the Hugging Face Spaces deploy target; builds and runs
the FastAPI app, no model artifact baked in yet.

## Registration stub

`registration/miner.yaml` — every blocked field commented with which open Discord ask it's
waiting on:
```yaml
supported_intents:
  - AI_DETECTION        # not yet confirmed on the hackathon's curated Intent list
                         # fallback if excluded: TEXT_AUTHENTICITY_CHECK or CONTENT_VERIFICATION
                         # (no model changes needed either way — see registration/README.md)
node_address: null       # blocked — need to request in Discord (not yet asked)
x_internal_secret: null  # blocked — same ask as node_address
endpoint: null           # filled in once deployed
intent_version: null     # unconfirmed whether an explicit @v1.0 suffix is required
```

`registration/README.md` — plain-language restatement of the three blocking items above, so
the repo is self-explanatory to anyone opening it without needing outside context.

## Testing

- `tests/data_pipeline/`: dataset schema validity, class-balance math, group-aware split (no
  leakage), manifest resume/skip logic.
- `tests/service/`: `/health` returns 200; `/predict` returns 501 with a clear "model not
  loaded" message rather than a silent 200 with fake data.
- `conftest.py` mocks all three provider clients so test runs never hit real APIs or spend
  budget.

## Out of scope

- Model training (Phase 2) — blocked on setting up a Vast.ai account.
- Actual registration (Phase 4) — blocked on Discord answers (node secret, curated Intent
  list confirmation, testnet MACHINA source).
- RAID benchmark stretch dataset — not needed for an initial working pipeline.
