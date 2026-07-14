from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Language(str, Enum):
    ru = "ru"
    kk = "kk"


class Label(str, Enum):
    payment = "payment"
    delivery = "delivery"
    account = "account"
    technical = "technical"


class TicketStatus(str, Enum):
    open = "open"
    reviewed = "reviewed"


class TicketCreate(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)
    language: Language

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("text must not be empty")
        return stripped


class TopPrediction(BaseModel):
    label: Label
    confidence: float


class TicketResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    text: str
    language: Language
    predicted_label: Label
    confidence: float
    top_predictions: list[TopPrediction]
    needs_review: bool
    status: TicketStatus
    created_at: datetime


class FeedbackCreate(BaseModel):
    correct_label: Label
    comment: str | None = Field(default=None, max_length=2_000)


class FeedbackResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    ticket_id: int
    correct_label: Label
    comment: str | None
    created_at: datetime


class TicketDetailResponse(TicketResponse):
    feedbacks: list[FeedbackResponse]


class TicketListResponse(BaseModel):
    items: list[TicketResponse]
    total: int
    limit: int
    offset: int
