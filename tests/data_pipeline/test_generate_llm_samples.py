import json

import pytest

from tests.conftest import FakeProvider
from veritarach.data_pipeline.fetch_hc3 import HC3Pair
from veritarach.data_pipeline.generate_llm_samples import (
    BudgetExceededError,
    GeneratedSample,
    generate_llm_samples,
)


def _hc3_pairs(n):
    return [
        HC3Pair(
            pair_id=f"pair-{i}",
            question=f"Question {i}?",
            human_answer=f"Human answer {i}",
            ai_answer=f"AI answer {i}",
        )
        for i in range(n)
    ]


def test_splits_paired_and_topic_by_ratio(tmp_path):
    provider = FakeProvider(name="claude", responses=["a response"])

    samples = generate_llm_samples(
        provider=provider,
        hc3_pairs=_hc3_pairs(10),
        sample_count=10,
        paired_ratio=0.7,
        manifest_path=tmp_path / "manifest.json",
        output_path=tmp_path / "samples.jsonl",
        budget_usd=1000.0,
    )

    paired = [s for s in samples if s.generation_strategy == "paired"]
    topic = [s for s in samples if s.generation_strategy == "topic"]
    assert len(paired) == 7
    assert len(topic) == 3
    assert all(isinstance(s, GeneratedSample) for s in samples)
    assert all(s.pair_id is not None for s in paired)
    assert all(s.pair_id is None for s in topic)
    assert all(s.source == "claude" for s in samples)


def test_rerun_with_populated_manifest_skips_done_samples_and_makes_no_new_calls(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "samples.jsonl"
    first_provider = FakeProvider(name="claude", responses=["a response"])

    generate_llm_samples(
        provider=first_provider,
        hc3_pairs=_hc3_pairs(5),
        sample_count=5,
        paired_ratio=1.0,
        manifest_path=manifest_path,
        output_path=output_path,
        budget_usd=1000.0,
    )
    assert first_provider._call_count == 5

    second_provider = FakeProvider(name="claude", responses=["a response"])
    second_samples = generate_llm_samples(
        provider=second_provider,
        hc3_pairs=_hc3_pairs(5),
        sample_count=5,
        paired_ratio=1.0,
        manifest_path=manifest_path,
        output_path=output_path,
        budget_usd=1000.0,
    )

    assert second_provider._call_count == 0
    assert second_samples == []


def test_provider_error_is_skipped_without_aborting_batch(tmp_path):
    # fail_after=1: the 1st call succeeds, every call after that raises ProviderError.
    provider = FakeProvider(name="claude", responses=["ok"], fail_after=1)

    samples = generate_llm_samples(
        provider=provider,
        hc3_pairs=_hc3_pairs(5),
        sample_count=3,
        paired_ratio=1.0,
        manifest_path=tmp_path / "manifest.json",
        output_path=tmp_path / "samples.jsonl",
        budget_usd=1000.0,
    )

    assert len(samples) == 1


def test_failed_sample_is_not_marked_done_so_a_future_run_retries_it(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    provider = FakeProvider(name="claude", responses=["ok"], fail_after=0)  # every call fails

    generate_llm_samples(
        provider=provider,
        hc3_pairs=_hc3_pairs(1),
        sample_count=1,
        paired_ratio=1.0,
        manifest_path=manifest_path,
        output_path=tmp_path / "samples.jsonl",
        budget_usd=1000.0,
    )

    # _save_manifest only runs after a successful sample -- if every call failed, the
    # manifest file may never have been written at all, which is an equally valid way
    # of confirming nothing was marked done.
    assert not manifest_path.exists() or json.loads(manifest_path.read_text(encoding="utf-8")) == {}


def test_exceeding_budget_raises_before_the_overbudget_call_happens(tmp_path):
    provider = FakeProvider(name="claude", responses=["a response"])

    with pytest.raises(BudgetExceededError):
        generate_llm_samples(
            provider=provider,
            hc3_pairs=_hc3_pairs(5),
            sample_count=5,
            paired_ratio=1.0,
            manifest_path=tmp_path / "manifest.json",
            output_path=tmp_path / "samples.jsonl",
            budget_usd=0.000001,  # effectively zero -- the first call already exceeds it
        )

    assert provider._call_count == 0


def test_writes_valid_jsonl(tmp_path):
    provider = FakeProvider(name="claude", responses=["a response"])
    output_path = tmp_path / "samples.jsonl"

    generate_llm_samples(
        provider=provider,
        hc3_pairs=_hc3_pairs(3),
        sample_count=3,
        paired_ratio=1.0,
        manifest_path=tmp_path / "manifest.json",
        output_path=output_path,
        budget_usd=1000.0,
    )

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    row = json.loads(lines[0])
    assert set(row.keys()) == {"pair_id", "source", "generation_strategy", "text"}


def test_falls_back_to_topic_strategy_when_no_hc3_pairs_available(tmp_path):
    provider = FakeProvider(name="claude", responses=["a response"])

    samples = generate_llm_samples(
        provider=provider,
        hc3_pairs=[],
        sample_count=3,
        paired_ratio=0.7,
        manifest_path=tmp_path / "manifest.json",
        output_path=tmp_path / "samples.jsonl",
        budget_usd=1000.0,
    )

    assert len(samples) == 3
    assert all(s.generation_strategy == "topic" for s in samples)
