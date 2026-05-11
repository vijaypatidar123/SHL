"""Pydantic schemas for the /chat endpoint.

The wire schema is non-negotiable per the assignment spec — keep field names,
types, and nullability stable. The grader will reject deviations.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


Role = Literal["user", "assistant", "system"]


class Message(BaseModel):
    role: Role
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)

    @field_validator("messages")
    @classmethod
    def _last_must_be_user(cls, v: list[Message]) -> list[Message]:
        if v[-1].role != "user":
            raise ValueError("last message must have role='user'")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False

    @field_validator("recommendations")
    @classmethod
    def _max_ten(cls, v: list[Recommendation]) -> list[Recommendation]:
        if len(v) > 10:
            raise ValueError("recommendations must contain at most 10 items")
        return v


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
