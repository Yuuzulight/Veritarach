from fastapi import FastAPI, HTTPException

from .schemas import PredictRequest, PredictResponse

app = FastAPI()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    raise HTTPException(status_code=501, detail="model not loaded")
