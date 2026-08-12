import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from datasets import load_dataset


@dataclass
class HC3Pair:
    pair_id: str
    question: str
    human_answer: str
    ai_answer: str


def fetch_hc3(output_dir: Path, limit: int | None = None) -> list[HC3Pair]:
    """Loads Hello-SimpleAI/HC3, writes output_dir/hc3_pairs.jsonl, and returns the pairs.
    If limit is set, only the first `limit` valid pairs are kept (for fast local dev —
    production runs pass limit=None).

    Uses revision="refs/convert/parquet" (Hugging Face's auto-maintained Parquet mirror)
    because the repo's native format is a legacy dataset loading script, which current
    `datasets` versions refuse to execute at all (verified 2026-08-12 — a real, unrelated
    breaking change upstream, not something wrong on our end). The parquet mirror only
    exposes a single "default" config rather than the original "all_no_ttf" — same columns
    (question/human_answers/chatgpt_answers), same row count as "all" (48,644 rows); the
    no_ttf split-specific filtering doesn't survive the conversion, but our own blank-answer
    filtering below covers the same underlying data-quality concern."""
    dataset = load_dataset("Hello-SimpleAI/HC3", "default", split="train", revision="refs/convert/parquet")

    pairs: list[HC3Pair] = []
    saw_any_row = False
    for row in dataset:
        saw_any_row = True
        question = (row.get("question") or "").strip()
        human_answer = _first_nonblank(row.get("human_answers") or [])
        ai_answer = _first_nonblank(row.get("chatgpt_answers") or [])
        if not question or not human_answer or not ai_answer:
            continue

        pair_id = hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
        pairs.append(
            HC3Pair(pair_id=pair_id, question=question, human_answer=human_answer, ai_answer=ai_answer)
        )
        if limit is not None and len(pairs) >= limit:
            break

    if saw_any_row and not pairs:
        raise RuntimeError(
            "fetch_hc3: extracted zero valid pairs from a non-empty HC3 dataset — "
            "the dataset's field names (question/human_answers/chatgpt_answers) may have changed"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "hc3_pairs.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(asdict(pair)) + "\n")

    return pairs


def _first_nonblank(answers: list[str]) -> str:
    for answer in answers:
        if answer and answer.strip():
            return answer.strip()
    return ""
