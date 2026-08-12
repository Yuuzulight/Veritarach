from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from veritarach.training.dataset import ID_TO_LABEL, LABEL_TO_ID, load_split, tokenize_dataset
from veritarach.training.metrics import compute_metrics


def train_model(
    data_dir: Path,
    output_dir: Path,
    base_model: str,
    max_length: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
) -> dict:
    """Fine-tunes base_model for AI-vs-human text classification on data_dir's
    train/val/test.jsonl (as produced by build_dataset.py). Saves the best checkpoint
    (by validation F1) to output_dir/final. Returns the test-set metrics."""
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=2, id2label=ID_TO_LABEL, label2id=LABEL_TO_ID
    )

    train_ds = tokenize_dataset(load_split(data_dir / "train.jsonl"), tokenizer, max_length)
    val_ds = tokenize_dataset(load_split(data_dir / "val.jsonl"), tokenizer, max_length)
    test_ds = tokenize_dataset(load_split(data_dir / "test.jsonl"), tokenizer, max_length)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to=[],
        # bf16 was tried for speed but produced NaN eval_loss / zero F1 on a real GPU run
        # (2026-08-13) -- DeBERTa-v3's disentangled attention has documented numerical
        # stability issues under reduced precision. Reverted to fp32; the time budget here
        # comfortably absorbs the ~2x slower training, correctness matters more than speed.
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    return test_metrics
