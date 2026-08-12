import torch
from fastapi import FastAPI, HTTPException
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from veritarach.config import get_settings
from veritarach.training.dataset import ID_TO_LABEL

from .schemas import PredictRequest, PredictResponse

app = FastAPI()

_model = None
_tokenizer = None


def _checkpoint_dir():
    return get_settings().data_dir / "model" / "final"


def _load_model() -> None:
    """Lazily loads the trained checkpoint on first use. Leaves _model as None (rather
    than raising) when no checkpoint exists yet -- that's the expected state in CI, which
    never has the ~740MB trained weights (gitignored, not committed)."""
    global _model, _tokenizer
    if _model is not None:
        return
    checkpoint_dir = _checkpoint_dir()
    if not checkpoint_dir.exists():
        return
    _tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    _model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir))
    _model.eval()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    _load_model()
    if _model is None:
        raise HTTPException(status_code=501, detail="model not loaded")

    max_length = get_settings().training_max_length
    inputs = _tokenizer(request.text, return_tensors="pt", truncation=True, max_length=max_length)
    with torch.no_grad():
        logits = _model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    predicted_id = int(torch.argmax(probs).item())

    return PredictResponse(label=ID_TO_LABEL[predicted_id], confidence=float(probs[predicted_id].item()))
