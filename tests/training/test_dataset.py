import json

from veritarach.training.dataset import ID_TO_LABEL, LABEL_TO_ID, load_split, tokenize_dataset


class _FakeTokenizer:
    """Test double standing in for a real HF tokenizer -- returns fixed-length
    input_ids/attention_mask lists without needing an actual model download."""

    def __call__(self, texts, truncation=True, max_length=8, padding="max_length"):
        return {
            "input_ids": [[1] * max_length for _ in texts],
            "attention_mask": [[1] * max_length for _ in texts],
        }


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_label_mapping_is_a_clean_bijection():
    assert LABEL_TO_ID == {"human_written": 0, "ai_generated": 1}
    assert ID_TO_LABEL == {0: "human_written", 1: "ai_generated"}


def test_load_split_maps_labels_and_keeps_text(tmp_path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [
            {"text": "a human wrote this", "label": "human_written", "source": "hc3",
             "generation_strategy": None, "pair_id": "p1", "split": "train"},
            {"text": "an AI wrote this", "label": "ai_generated", "source": "gpt4o",
             "generation_strategy": "topic", "pair_id": None, "split": "train"},
        ],
    )

    dataset = load_split(path)

    assert dataset.column_names == ["text", "label"]
    assert dataset[0]["text"] == "a human wrote this"
    assert dataset[0]["label"] == 0
    assert dataset[1]["text"] == "an AI wrote this"
    assert dataset[1]["label"] == 1


def test_load_split_skips_blank_lines(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        '{"text": "x", "label": "human_written"}\n\n{"text": "y", "label": "ai_generated"}\n',
        encoding="utf-8",
    )

    dataset = load_split(path)

    assert len(dataset) == 2


def test_tokenize_dataset_replaces_text_with_tokenizer_output(tmp_path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [{"text": "hello", "label": "human_written"}, {"text": "world", "label": "ai_generated"}])
    dataset = load_split(path)

    tokenized = tokenize_dataset(dataset, _FakeTokenizer(), max_length=8)

    assert "text" not in tokenized.column_names
    assert set(tokenized.column_names) == {"label", "input_ids", "attention_mask"}
    assert len(tokenized[0]["input_ids"]) == 8
    assert tokenized[0]["label"] == 0
    assert tokenized[1]["label"] == 1
