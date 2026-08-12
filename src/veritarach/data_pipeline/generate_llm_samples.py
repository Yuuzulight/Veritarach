import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from veritarach.data_pipeline.fetch_hc3 import HC3Pair
from veritarach.data_pipeline.providers.base import LLMProvider, ProviderError

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4  # ~1 token per 4 chars of English text (Anthropic's published rule of thumb)
ASSUMED_COMPLETION_TOKENS = 500  # conservative pre-call estimate for the budget gate -- the
# real completion length isn't known until after the call succeeds

# USD per token. Looked up 2026-08-12 from each provider's official pricing docs -- re-verify
# before relying on this for anything beyond a soft local budget cap, published rates change:
# - Claude Sonnet 5: https://platform.claude.com/docs/en/about-claude/pricing ($2/$10 per MTok in/out)
# - GPT-4o: https://developers.openai.com/api/docs/pricing ($2.50/$10 per MTok in/out)
# - Gemini 2.5 Flash: https://ai.google.dev/gemini-api/docs/pricing (free tier -- input/output
#   free of charge; Settings.gemini_sample_count is sized to stay within it)
PRICING = {
    "claude": {"input": 2.0 / 1_000_000, "output": 10.0 / 1_000_000},
    "gpt4o": {"input": 2.5 / 1_000_000, "output": 10.0 / 1_000_000},
    "gemini": {"input": 0.0, "output": 0.0},
}

TOPIC_PROMPTS = [
    ("essay", "Write a short essay about {subject}."),
    ("product_review", "Write a product review for {subject}."),
    ("news_blurb", "Write a short news item about {subject}."),
    ("technical_explanation", "Write a technical explanation of {subject}."),
    ("short_story", "Write a short story involving {subject}."),
]

TOPIC_SUBJECTS = [
    "a local bakery",
    "renewable energy",
    "a museum exhibit",
    "a new smartphone",
    "urban gardening",
    "a marathon race",
    "a neighborhood coffee shop",
    "space exploration",
    "a hiking trail",
    "a jazz concert",
]


class BudgetExceededError(Exception):
    """Raised when the next call would exceed the configured budget cap."""


@dataclass
class GeneratedSample:
    pair_id: str | None
    source: str
    generation_strategy: str
    text: str


def generate_llm_samples(
    provider: LLMProvider,
    hc3_pairs: list[HC3Pair],
    sample_count: int,
    paired_ratio: float,
    manifest_path: Path,
    output_path: Path,
    budget_usd: float,
) -> list[GeneratedSample]:
    """Generates sample_count samples from `provider`: round(sample_count * paired_ratio)
    using the paired strategy (answering an HC3 question), the rest using the topic
    strategy (cycling through TOPIC_PROMPTS). Skips any sample already recorded in
    manifest_path. Appends each successful sample to output_path (JSONL) and marks it
    done in the manifest immediately after -- a crash mid-run loses no completed work.
    Raises BudgetExceededError before making a call that would push cumulative estimated
    cost over budget_usd."""
    manifest = _load_manifest(manifest_path)
    paired_count = round(sample_count * paired_ratio)
    topic_count = sample_count - paired_count
    plan = _build_plan(provider.name, hc3_pairs, paired_count, topic_count)

    samples: list[GeneratedSample] = []
    cumulative_cost = 0.0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as out_file:
        for key, strategy, pair_id, prompt in plan:
            if manifest.get(key):
                continue

            estimated_cost = _estimate_cost(provider.name, prompt, ASSUMED_COMPLETION_TOKENS)
            if cumulative_cost + estimated_cost > budget_usd:
                raise BudgetExceededError(
                    f"{provider.name}: next call (~${estimated_cost:.6f}) would exceed the "
                    f"${budget_usd:.2f} budget (${cumulative_cost:.6f} spent so far)"
                )

            try:
                text = provider.generate(prompt)
            except ProviderError:
                logger.warning("generate_llm_samples: skipping %s after provider error", key, exc_info=True)
                continue

            cumulative_cost += _estimate_cost(provider.name, prompt, _estimate_tokens(text))

            sample = GeneratedSample(pair_id=pair_id, source=provider.name, generation_strategy=strategy, text=text)
            samples.append(sample)
            out_file.write(json.dumps(asdict(sample)) + "\n")
            out_file.flush()

            manifest[key] = True
            _save_manifest(manifest_path, manifest)

    return samples


def _build_plan(
    provider_name: str, hc3_pairs: list[HC3Pair], paired_count: int, topic_count: int
) -> list[tuple[str, str, str | None, str]]:
    if not hc3_pairs:
        # Can't answer HC3 questions with no HC3 data -- fall back to topic-only generation
        # instead of crashing (relevant for fast local dev runs with a tiny HC3 subset).
        topic_count += paired_count
        paired_count = 0

    plan: list[tuple[str, str, str | None, str]] = []
    for i in range(paired_count):
        hc3_pair = hc3_pairs[i % len(hc3_pairs)]
        plan.append((f"{provider_name}:paired:{i}", "paired", hc3_pair.pair_id, hc3_pair.question))
    for i in range(topic_count):
        _format_name, template = TOPIC_PROMPTS[i % len(TOPIC_PROMPTS)]
        subject = TOPIC_SUBJECTS[i % len(TOPIC_SUBJECTS)]
        plan.append((f"{provider_name}:topic:{i}", "topic", None, template.format(subject=subject)))
    return plan


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _estimate_cost(provider_name: str, prompt: str, completion_tokens: int) -> float:
    rates = PRICING[provider_name]
    return _estimate_tokens(prompt) * rates["input"] + completion_tokens * rates["output"]


def _load_manifest(manifest_path: Path) -> dict:
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {}


def _save_manifest(manifest_path: Path, manifest: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
