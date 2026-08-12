import json
from unittest.mock import patch

import pytest

from veritarach.data_pipeline.fetch_hc3 import HC3Pair, fetch_hc3

FAKE_ROWS = [
    {
        "question": "What is the capital of France?",
        "human_answers": ["Paris is the capital of France."],
        "chatgpt_answers": ["The capital of France is Paris."],
    },
    {
        "question": "How does photosynthesis work?",
        "human_answers": ["Plants convert sunlight into energy."],
        "chatgpt_answers": ["Photosynthesis is the process by which plants make food."],
    },
    {
        # blank human answer -- should be skipped
        "question": "What is dark matter?",
        "human_answers": ["   "],
        "chatgpt_answers": ["Dark matter is a form of matter that doesn't emit light."],
    },
    {
        # no AI answer at all -- should be skipped
        "question": "Who wrote Hamlet?",
        "human_answers": ["William Shakespeare wrote Hamlet."],
        "chatgpt_answers": [],
    },
]


@patch("veritarach.data_pipeline.fetch_hc3.load_dataset")
def test_fetch_hc3_produces_one_pair_per_valid_row(mock_load_dataset, tmp_path):
    mock_load_dataset.return_value = list(FAKE_ROWS)

    pairs = fetch_hc3(tmp_path)

    assert len(pairs) == 2
    assert all(isinstance(p, HC3Pair) for p in pairs)
    assert pairs[0].question == "What is the capital of France?"
    assert pairs[0].human_answer == "Paris is the capital of France."
    assert pairs[0].ai_answer == "The capital of France is Paris."
    mock_load_dataset.assert_called_once_with(
        "Hello-SimpleAI/HC3", "default", split="train", revision="refs/convert/parquet"
    )


@patch("veritarach.data_pipeline.fetch_hc3.load_dataset")
def test_fetch_hc3_limit_truncates_to_first_n_valid_pairs(mock_load_dataset, tmp_path):
    mock_load_dataset.return_value = list(FAKE_ROWS)

    pairs = fetch_hc3(tmp_path, limit=1)

    assert len(pairs) == 1
    assert pairs[0].question == "What is the capital of France?"


@patch("veritarach.data_pipeline.fetch_hc3.load_dataset")
def test_fetch_hc3_skips_blank_or_missing_answers(mock_load_dataset, tmp_path):
    mock_load_dataset.return_value = list(FAKE_ROWS)

    pairs = fetch_hc3(tmp_path)

    questions = {p.question for p in pairs}
    assert "What is dark matter?" not in questions
    assert "Who wrote Hamlet?" not in questions


@patch("veritarach.data_pipeline.fetch_hc3.load_dataset")
def test_fetch_hc3_writes_valid_jsonl(mock_load_dataset, tmp_path):
    mock_load_dataset.return_value = list(FAKE_ROWS)

    fetch_hc3(tmp_path)

    output_path = tmp_path / "hc3_pairs.jsonl"
    assert output_path.exists()
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    row = json.loads(lines[0])
    assert set(row.keys()) == {"pair_id", "question", "human_answer", "ai_answer"}


@patch("veritarach.data_pipeline.fetch_hc3.load_dataset")
def test_fetch_hc3_pair_id_is_stable_across_runs(mock_load_dataset, tmp_path):
    mock_load_dataset.return_value = list(FAKE_ROWS)
    first_pairs = fetch_hc3(tmp_path)

    mock_load_dataset.return_value = list(FAKE_ROWS)
    second_pairs = fetch_hc3(tmp_path)

    assert first_pairs[0].pair_id == second_pairs[0].pair_id


@patch("veritarach.data_pipeline.fetch_hc3.load_dataset")
def test_fetch_hc3_raises_if_schema_no_longer_matches(mock_load_dataset, tmp_path):
    # A non-empty dataset with none of the expected fields simulates HC3's upstream
    # schema changing — this must fail loudly, not silently write an empty file.
    mock_load_dataset.return_value = [{"unexpected_field": "some value"}]

    with pytest.raises(RuntimeError, match="zero valid pairs"):
        fetch_hc3(tmp_path)
