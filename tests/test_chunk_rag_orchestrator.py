import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from meno_core.core.rag.models import RagRequest, QueryRepresentations, RetrievedChunk, Chunk, ChunkMetadata
from meno_core.core.rag.orchestration.orchestrator import ChunkRagOrchestrator
from meno_core.core.rag.config import ChunkRagConfig

@pytest.fixture
def mock_config():
    config = ChunkRagConfig()
    config.rewrite_enabled = True
    config.hypothetical_doc_enabled = False
    return config

@pytest.fixture
def empty_chunk_metadata():
    return ChunkMetadata(
        document_id="doc1",
        document_title="Test Doc",
        chunk_index=0
    )


@pytest.mark.asyncio
async def test_chunk_rag_pipeline(mock_config, empty_chunk_metadata):
    """
    Basic unit test to ensure the Orchestrator ties together all components without crashing.
    """
    # 1. Mocks
    mock_dense_retriever = AsyncMock()
    mock_lexical_retriever = AsyncMock()
    
    mock_chunk = Chunk(
        chunk_id="chunk1",
        text="NSU is a university.",
        text_for_dense="NSU is a university.",
        text_for_bm25="nsu is a uni",
        metadata=empty_chunk_metadata
    )
    
    # Retrievers return one dummy chunk
    retrieved = [RetrievedChunk(chunk=mock_chunk, score=0.9, source="hybrid")]
    mock_dense_retriever.retrieve.return_value = retrieved
    mock_lexical_retriever.retrieve.return_value = []
    
    orchestrator = ChunkRagOrchestrator(
        config=mock_config,
        dense_retriever=mock_dense_retriever,
        lexical_retriever=mock_lexical_retriever
    )
    
    # Mock internal processor & generator
    orchestrator.query_processor.process_query = AsyncMock(return_value=QueryRepresentations(
        original_query="What is NSU?",
        rewritten_query="What is NSU?",
        resolved_coreferences="What is NSU?",
        search_queries=["What is NSU?"],
        is_meaningful=True
    ))
    
    orchestrator.reranker.rerank = AsyncMock(return_value=retrieved)
    orchestrator.assembler.assemble = MagicMock(return_value=("Документ: Test Doc\nNSU is a university.", []))
    orchestrator.generator.generate_answer = AsyncMock(return_value=("NSU is a university.", False))
    
    # 2. Execute
    request = RagRequest(question="What is NSU?")
    response = await orchestrator.answer(request)
    
    # 3. Assert
    assert response.insufficient_information is False
    assert response.answer == "NSU is a university."
    assert orchestrator.query_processor.process_query.called
    assert orchestrator.assembler.assemble.called
    assert orchestrator.generator.generate_answer.called
