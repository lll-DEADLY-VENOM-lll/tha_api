"""Pydantic models for the YouTube Summarizer API.

These define the JSON shape of every API response. The summarize endpoint
validates Claude's output against `SummaryPayload` before returning it,
so the frontend can rely on a stable contract.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    """Information about the source YouTube video."""
    title: str
    channel: str
    duration_seconds: int = Field(ge=0)
    url: str
    thumbnail_url: Optional[str] = None


class TranscriptInfo(BaseModel):
    """The raw transcript extracted from the video's audio."""
    full_text: str
    word_count: int = Field(ge=0)
    language: str = "en"


class SummaryContent(BaseModel):
    """The AI-generated summary, the actual product output."""
    executive_summary: str
    key_insights: List[str] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list)
    topics_covered: List[str] = Field(default_factory=list)
    tone: Optional[str] = None  # "academic" | "casual" | "technical" | etc.


class ResponseMetadata(BaseModel):
    """Diagnostics — useful for debugging and cost tracking.

    Token breakdown lets the UI distinguish:
      - cache HIT  (savings!)
      - cache MISS, first call (next will be cheaper)
      - too short to cache (transcript under Anthropic's 1024-token minimum)
    """
    generated_at: str
    model: str
    tokens_used: int = Field(ge=0)
    cache_hit: bool = False
    pipeline_seconds: float = 0.0

    # Token-type breakdown (zero in stub/demo responses)
    input_tokens: int = Field(default=0, ge=0)
    cache_creation_tokens: int = Field(default=0, ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class SummaryPayload(BaseModel):
    """Top-level response shape for POST /api/summarize."""
    video: VideoMetadata
    transcript: TranscriptInfo
    summary: SummaryContent
    metadata: ResponseMetadata


class SummarizeRequest(BaseModel):
    """Inbound request body for POST /api/summarize."""
    url: str
    language: str = "en"
