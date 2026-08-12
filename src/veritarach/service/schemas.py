from typing import Literal

from pydantic import BaseModel


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: Literal["ai_generated", "human_written"]
    confidence: float
