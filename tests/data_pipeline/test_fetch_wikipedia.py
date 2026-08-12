import json
from unittest.mock import patch

from veritarach.data_pipeline.fetch_wikipedia import HumanTextSample, fetch_wikipedia

SHORT = "Too short."  # under 200 chars
LONG = "x" * 2001  # over 2000 chars
IN_RANGE_PARAGRAPHS = [f"Paragraph number {i}. " + ("lorem ipsum dolor sit amet. " * 8) for i in range(20)]


def _fake_rows():
    # Each "article" has multiple \n\n-separated paragraphs, mixing valid-length ones
    # with too-short and too-long ones to exercise the filter.
    return [
        {"text": f"{SHORT}\n\n{IN_RANGE_PARAGRAPHS[0]}\n\n{IN_RANGE_PARAGRAPHS[1]}"},
        {"text": f"{LONG}\n\n{IN_RANGE_PARAGRAPHS[2]}"},
        {"text": "\n\n".join(IN_RANGE_PARAGRAPHS[3:20])},
    ]


@patch("veritarach.data_pipeline.fetch_wikipedia.load_dataset")
def test_fetch_wikipedia_filters_by_paragraph_length(mock_load_dataset, tmp_path):
    mock_load_dataset.return_value = _fake_rows()

    samples = fetch_wikipedia(tmp_path, sample_count=20, seed=1)

    texts = {s.text for s in samples}
    assert SHORT not in texts
    assert LONG not in texts
    assert all(200 <= len(t) <= 2000 for t in texts)
    mock_load_dataset.assert_called_once_with(
        "wikimedia/wikipedia", "20231101.en", split="train", streaming=True
    )


@patch("veritarach.data_pipeline.fetch_wikipedia.load_dataset")
def test_fetch_wikipedia_sampling_is_deterministic_given_same_seed(mock_load_dataset, tmp_path):
    mock_load_dataset.return_value = _fake_rows()
    first = fetch_wikipedia(tmp_path, sample_count=5, seed=7)

    mock_load_dataset.return_value = _fake_rows()
    second = fetch_wikipedia(tmp_path, sample_count=5, seed=7)

    assert [s.text for s in first] == [s.text for s in second]


@patch("veritarach.data_pipeline.fetch_wikipedia.load_dataset")
def test_fetch_wikipedia_different_seed_can_change_selection(mock_load_dataset, tmp_path):
    mock_load_dataset.return_value = _fake_rows()
    first = fetch_wikipedia(tmp_path, sample_count=5, seed=1)

    mock_load_dataset.return_value = _fake_rows()
    second = fetch_wikipedia(tmp_path, sample_count=5, seed=2)

    assert [s.text for s in first] != [s.text for s in second]


@patch("veritarach.data_pipeline.fetch_wikipedia.load_dataset")
def test_fetch_wikipedia_writes_valid_jsonl(mock_load_dataset, tmp_path):
    mock_load_dataset.return_value = _fake_rows()

    fetch_wikipedia(tmp_path, sample_count=3, seed=1)

    output_path = tmp_path / "wikipedia_samples.jsonl"
    assert output_path.exists()
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    row = json.loads(lines[0])
    assert set(row.keys()) == {"source", "text"}
    assert row["source"] == "wikipedia"


@patch("veritarach.data_pipeline.fetch_wikipedia.load_dataset")
def test_fetch_wikipedia_returns_dataclass_instances(mock_load_dataset, tmp_path):
    mock_load_dataset.return_value = _fake_rows()

    samples = fetch_wikipedia(tmp_path, sample_count=3, seed=1)

    assert all(isinstance(s, HumanTextSample) for s in samples)


@patch("veritarach.data_pipeline.fetch_wikipedia.load_dataset")
def test_fetch_wikipedia_caps_at_available_candidates(mock_load_dataset, tmp_path):
    mock_load_dataset.return_value = _fake_rows()

    # More requested than the fake stream can actually supply after filtering.
    samples = fetch_wikipedia(tmp_path, sample_count=10_000, seed=1)

    assert len(samples) < 10_000
