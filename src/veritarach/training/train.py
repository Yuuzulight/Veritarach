import math
from pathlib import Path

import torch
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
    # ROOT CAUSE, confirmed by direct inspection (2026-08-13): from_pretrained() was loading
    # this checkpoint's weights in float16 by default -- e.g. classifier.bias came back as
    # dtype=torch.float16 -- completely independent of TrainingArguments' bf16 flag, which
    # only controls autocast during compute, not the underlying weight storage dtype. This
    # explains why every precision-related attempt (bf16, "fp32" via bf16=False, warmup,
    # dataloader_num_workers, eager attention) failed identically: the actual stored weights
    # were fp16 in every single one of them. fp16's narrow dynamic range is a well-known
    # cause of exactly the symptom seen (grad_norm already nan by the first logged step) for
    # models like DeBERTa-v3, and none of those attempts used fp16's required loss-scaling
    # (TrainingArguments(fp16=True)), so nothing was protecting against it. dtype=torch.float32
    # forces genuine fp32 weight storage; verified directly after this fix that
    # classifier.bias.dtype reports torch.float32, not float16.
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        attn_implementation="eager",
        dtype=torch.float32,
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
