from models.models import (
    Document,
    DocumentMetadataFilter,
    Query,
    QueryResult,
)
from pydantic import BaseModel
from typing import List, Optional


class UpsertRequest(BaseModel):
    documents: List[Document]


class UpsertResponse(BaseModel):
    ids: List[str]


class QueryRequest(BaseModel):
    queries: List[Query]


class QueryResponse(BaseModel):
    results: List[QueryResult]


class DeleteRequest(BaseModel):
    ids: Optional[List[str]] = None
    filter: Optional[DocumentMetadataFilter] = None
    delete_all: Optional[bool] = False


class DeleteResponse(BaseModel):
    success: bool


class GetYoutubeVideoTranscriptRequest(BaseModel):
    video_id: str


class TranscriptItem(BaseModel):
    duration: int
    start: float
    text: str


class TopicSummary(BaseModel):
    topic: str
    summary: str


class GetYoutubeTranscriptResponse(BaseModel):
    transcript: List[TranscriptItem]
    final_summary: str
