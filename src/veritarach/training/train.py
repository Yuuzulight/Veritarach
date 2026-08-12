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

    # Confirmed on a real GPU run (2026-08-13): without warmup, loss collapsed to exactly 0
    # by step 100 (logits exploding to +/-inf -- cross-entropy saturates to a displayed 0 in
    # that regime) and eval_loss came back nan at epoch end. Reproduced identically in both
    # bf16 and fp32, ruling out precision as the cause -- it's the freshly-initialized
    # classification head getting hit with the full LR from step 1. Ramping the LR up over
    # the first 10% of steps fixes this. This transformers version (5.15.0, confirmed via
    # the installed signature) only accepts warmup_steps, not the more common warmup_ratio,
    # so it's computed here from the actual step count rather than passed as a fraction.
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
        dataloader_num_workers=4,
        warmup_steps=warmup_steps,
        # max_grad_norm is already the Trainer default (1.0); set explicitly since clipping
        # alone wasn't sufficient here without warmup too.
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
