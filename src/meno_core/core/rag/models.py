from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class RagMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    text: str


class RagRequest(BaseModel):
    question: str
    history: List[RagMessage] = Field(default_factory=list)
    mode: str = "chunk_rag"
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    language: Optional[str] = None


class RagSource(BaseModel):
    document_id: str
    document_title: str
    chunk_ids: List[str]
    source_url: str


class RagDebugInfo(BaseModel):
    rewritten_query: Optional[str] = None
    expanded_abbreviations: List[str] = Field(default_factory=list)
    resolved_coreferences: Optional[str] = None
    search_queries: List[str] = Field(default_factory=list)
    hypothetical_document: Optional[str] = None
    retrieval_stats: Dict[str, Any] = Field(default_factory=dict)


class RagResponse(BaseModel):
    answer: str
    sources: List[RagSource] = Field(default_factory=list)
    debug: RagDebugInfo = Field(default_factory=RagDebugInfo)
    insufficient_information: bool = False


# Data Models for Retrieval

class ChunkMetadata(BaseModel):
    document_id: str
    document_title: str
    section_title: Optional[str] = None
    source_url: Optional[str] = None
    page_range: Optional[str] = None
    chunk_index: int = 0
    extra: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    text: str
    text_for_dense: str
    text_for_bm25: str
    metadata: ChunkMetadata


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    source: Literal["dense", "lexical", "hybrid"] = "hybrid"


class QueryRepresentations(BaseModel):
    original_query: str
    rewritten_query: str
    resolved_coreferences: str
    expanded_abbreviations: List[str] = Field(default_factory=list)
    search_queries: List[str] = Field(default_factory=list)
    hypothetical_document: str = ""
    is_meaningful: bool = True
