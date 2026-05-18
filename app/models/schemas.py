from pydantic import BaseModel, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class RetrievedChunk(BaseModel):
    score: float
    text: str
    title: Optional[str] = None
    url: Optional[str] = None


class RetrieverResponse(BaseModel):
    query: str
    results: List[RetrievedChunk]


class LLMRequest(BaseModel):
    query: str
    context: List[str]


class LLMResponse(BaseModel):
    answer: str


class ChatRequest(BaseModel):
    query: str
    top_k: int = 5
    session_id: Optional[str] = Field(
        default=None,
        description="Omit to start a new conversation; reuse session_id from a prior ChatResponse to continue.",
    )


class ReferenceItem(BaseModel):
    index: int
    title: Optional[str] = None
    url: Optional[str] = None
    node_id: str
    audio_links: List[str] = Field(default_factory=list)
    video_links: List[str] = Field(default_factory=list)
    source: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    references: List[ReferenceItem]
    session_id: str