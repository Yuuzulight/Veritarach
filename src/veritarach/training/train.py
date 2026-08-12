import math
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

    # On a real GPU run (2026-08-13): loss reads a normal ~0.34-0.36 at step 50, then
    # collapses to exactly 0 at every subsequent logged step (logits exploding to +/-inf --
    # cross-entropy saturates to a displayed 0 in that regime), and eval_loss comes back nan
    # at epoch end. Reproduced identically across bf16, fp32, and fp32+warmup -- ruling out
    # both precision and an unwarmed classification head as the sole cause. Training data
    # itself checked clean (no empty/malformed rows). The one untested variable common to
    # every failing attempt: dataloader_num_workers=4, added in the same original change as
    # bf16. Multi-worker loading with a fast tokenizer has known fork-safety issues on Linux
    # that can silently hand back corrupted batches rather than crash outright -- removed
    # here as the next thing to isolate, not yet a confirmed root cause.
    steps_per_epoch = math.ceil(len(train_ds) / batch_size)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = round(total_steps * 0.1)

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
        warmup_steps=warmup_steps,
        # max_grad_norm is already the Trainer default (1.0); set explicitly for clarity.
        max_grad_norm=1.0,
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
