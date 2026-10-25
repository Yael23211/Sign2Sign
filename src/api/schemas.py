from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

Language = Literal["ASL", "LSM"]
SessionState = Literal["capturing", "processing", "ready", "failed", "closed"]


class CreateSession(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_language: Language


class Command(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command: Literal["space", "backspace", "clear"]


class CaptureImage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    content_type: Literal["image/jpeg", "image/png"]
    data: bytes = Field(min_length=1)


class VisualItem(BaseModel):
    concept_id: str
    language: Language
    url: str
    position: int
    total: int


class StageError(BaseModel):
    stage: Literal["SC-04", "SC-05", "SC-06"]
    code: str
    message: str


class TranslationResponse(BaseModel):
    source_language: Language
    target_language: Language
    recognized_text: str
    corrected_text: str
    translated_text: str
    concepts: list[str]
    unresolved: list[str]
    resources: list[VisualItem]
    latency_seconds: float
    state: SessionState
    error: StageError | None = None


class SessionResponse(BaseModel):
    id: str
    source_language: Language
    target_language: Language
    state: SessionState
    text: str
    last_result: TranslationResponse | None = None


class PredictionCandidateResponse(BaseModel):
    label: str
    probability: float = Field(ge=0, le=1)


class RecognitionResponse(BaseModel):
    source_language: Language | None = None
    letter: str
    confidence: float = Field(ge=0, le=1)
    top_k: list[PredictionCandidateResponse]
    detection_variant: str | None = None


class CaptureResponse(BaseModel):
    session: SessionResponse
    recognition: RecognitionResponse | None
