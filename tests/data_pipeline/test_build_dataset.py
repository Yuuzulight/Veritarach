import json

import pytest

from veritarach.data_pipeline.build_dataset import build_dataset
from veritarach.data_pipeline.fetch_hc3 import HC3Pair
from veritarach.data_pipeline.fetch_wikipedia import HumanTextSample
from veritarach.data_pipeline.generate_llm_samples import GeneratedSample


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


def _wikipedia_samples(n):
    return [HumanTextSample(source="wikipedia", text=f"Wikipedia paragraph {i}") for i in range(n)]


def _generated_samples(n):
    return [
        GeneratedSample(pair_id=None, source="claude", generation_strategy="topic", text=f"Generated {i}")
        for i in range(n)
    ]


def test_classes_end_up_equal_count(tmp_path):
    result = build_dataset(
        hc3_pairs=_hc3_pairs(20),
        wikipedia_samples=_wikipedia_samples(15),
        generated_samples=_generated_samples(15),
        output_dir=tmp_path,
        seed=1,
    )

    all_rows = result["train"] + result["val"] + result["test"]
    ai_count = sum(1 for r in all_rows if r.label == "ai_generated")
    human_count = sum(1 for r in all_rows if r.label == "human_written")
    assert ai_count == human_count
    # 20 HC3 pairs -> 20 AI + 20 human, plus 15 generated (AI) + 15 wikipedia (human)
    assert ai_count == 35
    assert human_count == 35


def test_pair_id_halves_always_land_in_the_same_split(tmp_path):
    result = build_dataset(
        hc3_pairs=_hc3_pairs(20),
        wikipedia_samples=_wikipedia_samples(10),
        generated_samples=_generated_samples(10),
        output_dir=tmp_path,
        seed=1,
    )

    split_by_pair_id = {}
    for split_name, rows in result.items():
        for row in rows:
            if row.pair_id is None:
                continue
            if row.pair_id in split_by_pair_id:
                assert split_by_pair_id[row.pair_id] == split_name, (
                    f"pair_id {row.pair_id} appears in both "
                    f"{split_by_pair_id[row.pair_id]} and {split_name}"
                )
            else:
                split_by_pair_id[row.pair_id] = split_name

    # every one of the 20 HC3 pairs should have been seen (2 rows each, same split)
    assert len(split_by_pair_id) == 20


def test_deterministic_given_same_seed(tmp_path):
    kwargs = dict(
        hc3_pairs=_hc3_pairs(20),
        wikipedia_samples=_wikipedia_samples(10),
        generated_samples=_generated_samples(10),
        seed=7,
    )
    first = build_dataset(output_dir=tmp_path / "run1", **kwargs)
    second = build_dataset(output_dir=tmp_path / "run2", **kwargs)

    first_texts = {split: sorted(r.text for r in rows) for split, rows in first.items()}
    second_texts = {split: sorted(r.text for r in rows) for split, rows in second.items()}
    assert first_texts == second_texts


def test_wikipedia_shortfall_raises_value_error_naming_the_shortfall(tmp_path):
    with pytest.raises(ValueError, match=r"need 15.*only 5"):
        build_dataset(
            hc3_pairs=_hc3_pairs(5),
            wikipedia_samples=_wikipedia_samples(5),
            generated_samples=_generated_samples(15),
            output_dir=tmp_path,
            seed=1,
        )


def test_writes_three_jsonl_files_summing_to_total_row_count(tmp_path):
    result = build_dataset(
        hc3_pairs=_hc3_pairs(20),
        wikipedia_samples=_wikipedia_samples(10),
        generated_samples=_generated_samples(10),
        output_dir=tmp_path,
        seed=1,
    )

    total_from_result = sum(len(rows) for rows in result.values())
    # 20 HC3 pairs -> 40 rows, + 10 generated + 10 wikipedia (exact match, no shortfall) = 60
    assert total_from_result == 60

    total_from_files = 0
    for split_name in ("train", "val", "test"):
        path = tmp_path / f"{split_name}.jsonl"
        assert path.exists()
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        total_from_files += len(lines)
        if lines:
            row = json.loads(lines[0])
            assert set(row.keys()) == {"text", "label", "source", "generation_strategy", "pair_id", "split"}

    assert total_from_files == 60
