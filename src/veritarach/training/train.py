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
    # On a real GPU run (2026-08-13): grad_norm was already nan at the very first logged
    # step (step 50), even though the forward-pass loss still showed a plausible number --
    # ruling out precision (bf16/fp32), warmup, and dataloader_num_workers, none of which
    # can make gradients NaN this immediately since they only affect update magnitude, not
    # whether the backward pass itself produces NaN. Training data confirmed clean. This
    # points to DeBERTa-v3's non-standard disentangled attention hitting a bug in whichever
    # optimized/fused attention kernel this transformers version auto-selects -- forcing the
    # original eager implementation is the standard escape hatch for exactly this failure
    # mode (custom attention architectures are the ones most likely to have kernel-specific
    # bugs, since optimized kernels are usually validated against standard QK^T attention).
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=2, id2label=ID_TO_LABEL, label2id=LABEL_TO_ID, attn_implementation="eager"
    )

    train_ds = tokenize_dataset(load_split(data_dir / "train.jsonl"), tokenizer, max_length)
    val_ds = tokenize_dataset(load_split(data_dir / "val.jsonl"), tokenizer, max_length)
    test_ds = tokenize_dataset(load_split(data_dir / "test.jsonl"), tokenizer, max_length)

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
