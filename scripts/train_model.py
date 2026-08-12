"""Fine-tunes DeBERTa-v3-base on the built dataset (data/generated/{train,val,test}.jsonl).
Meant to run on a GPU instance -- CPU training would take far too long for this dataset
size. Saves the final checkpoint to data/model/final and prints test-set metrics.

Usage: uv run python scripts/train_model.py
"""

from veritarach.config import get_settings
from veritarach.training.train import train_model


def main() -> None:
    settings = get_settings()
    metrics = train_model(
        data_dir=settings.data_dir / "generated",
        output_dir=settings.data_dir / "model",
        base_model=settings.base_model,
        max_length=settings.training_max_length,
        num_epochs=settings.training_num_epochs,
        batch_size=settings.training_batch_size,
        learning_rate=settings.training_learning_rate,
    )
    print("Test metrics:", metrics)


if __name__ == "__main__":
    main()
