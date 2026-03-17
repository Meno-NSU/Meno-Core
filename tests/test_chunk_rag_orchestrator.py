import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from meno_core.core.rag.config import ChunkRagConfig
from meno_core.core.rag.models import Chunk, ChunkMetadata, QueryRepresentations, RagRequest, RetrievedChunk
from meno_core.core.rag.orchestration.orchestrator import ChunkRagOrchestrator
from meno_core.core.rag.rerank.qwen_reranker import QwenRerankResult


@pytest.fixture
def chunk_metadata() -> ChunkMetadata:
    return ChunkMetadata(
        document_id="doc1",
        document_title="Test Doc",
        source_url="https://example.com/doc1",
        chunk_index=0,
    )


def test_chunk_rag_pipeline_exposes_debug_payload(chunk_metadata: ChunkMetadata):
    config = ChunkRagConfig(
        rewrite_enabled=True,
        hypothetical_doc_enabled=False,
        debug_retrieval=True,
        retrieval_preview_k=3,
        top_k_dense_multilingual=2,
        top_k_dense_russian=2,
        top_k_bm25=2,
    )

    chunk = Chunk(
        chunk_id="doc1_chunk_0",
        text="NSU is a university in Novosibirsk.",
        text_for_dense="Document: Test Doc\n\nNSU is a university in Novosibirsk.",
        text_for_bm25="document test doc nsu is a university in novosibirsk",
        metadata=chunk_metadata,
    )

    multilingual_hits = [RetrievedChunk(chunk=chunk, score=0.9, source="multilingual_dense")]
    russian_hits = [RetrievedChunk(chunk=chunk, score=0.8, source="russian_dense")]
    lexical_hits = [RetrievedChunk(chunk=chunk, score=1.2, source="lexical")]

    multilingual_retriever = AsyncMock()
    multilingual_retriever.retrieve_many.return_value = [multilingual_hits, multilingual_hits]
    russian_retriever = AsyncMock()
    russian_retriever.retrieve_many.return_value = [russian_hits, russian_hits]
    lexical_retriever = AsyncMock()
    lexical_retriever.retrieve_many.return_value = [lexical_hits, lexical_hits]

    reranker = MagicMock()
    reranker.rerank = AsyncMock(
        return_value=QwenRerankResult(
            reranked_chunks=[RetrievedChunk(chunk=chunk, score=0.95, source="hybrid")],
            preview=[
                {
                    "chunk_id": chunk.chunk_id,
                    "score": 0.95,
                    "source": "hybrid",
                }
            ],
        )
    )

    orchestrator = ChunkRagOrchestrator(
        config=config,
        dense_retrievers={
            "multilingual_dense": multilingual_retriever,
            "russian_dense": russian_retriever,
        },
        lexical_retriever=lexical_retriever,
        reranker=reranker,
    )

    orchestrator.query_processor.process_query = AsyncMock(
        return_value=QueryRepresentations(
            original_query="What is NSU?",
            rewritten_query="What is NSU?",
            resolved_coreferences="What is NSU?",
            search_queries=["NSU university"],
            hypothetical_document="",
            is_meaningful=True,
        )
    )
    orchestrator.assembler.assemble = MagicMock(
        return_value=("Документ: Test Doc\nNSU is a university in Novosibirsk.", [])
    )
    orchestrator.generator.generate_answer = AsyncMock(return_value=("NSU is a university.", False))

    response = asyncio.run(orchestrator.answer(RagRequest(question="What is NSU?")))

    assert response.insufficient_information is False
    assert response.answer == "NSU is a university."
    assert response.debug.retrieval_stats["retrieved_counts"] == {
        "multilingual_dense": 1,
        "russian_dense": 1,
        "lexical": 1,
    }
    assert "retrieval_previews" in response.debug.retrieval_stats
    assert "fused_preview" in response.debug.retrieval_stats
    assert "rerank_preview" in response.debug.retrieval_stats
    assert "steps_latency_ms" in response.debug.retrieval_stats
    assert response.debug.retrieval_stats["retrieval_call_count"] == 3
    assert response.debug.retrieval_stats["retrieval_latency_summary_ms"]["multilingual_dense"]["batch_sizes"] == [2]
    assert response.debug.retrieval_stats["retrieval_latency_summary_ms"]["lexical"]["batch_sizes"] == [2]
    assert response.debug.retrieval_stats["rerank_candidate_count"] == 1
    assert response.debug.retrieval_stats["context_token_count"] > 0


def test_chunk_rag_bypasses_query_processor_when_rewrite_disabled(chunk_metadata: ChunkMetadata):
    config = ChunkRagConfig(
        rewrite_enabled=False,
        top_k_dense_multilingual=1,
        top_k_dense_russian=1,
        top_k_bm25=1,
    )

    multilingual_retriever = AsyncMock()
    multilingual_retriever.retrieve_many.return_value = [[]]
    russian_retriever = AsyncMock()
    russian_retriever.retrieve_many.return_value = [[]]
    lexical_retriever = AsyncMock()
    lexical_retriever.retrieve_many.return_value = [[]]

    reranker = MagicMock()
    reranker.rerank = AsyncMock(return_value=QwenRerankResult(reranked_chunks=[], preview=[]))

    orchestrator = ChunkRagOrchestrator(
        config=config,
        dense_retrievers={
            "multilingual_dense": multilingual_retriever,
            "russian_dense": russian_retriever,
        },
        lexical_retriever=lexical_retriever,
        reranker=reranker,
    )

    orchestrator.query_processor.process_query = AsyncMock(side_effect=AssertionError("should not be called"))

    response = asyncio.run(orchestrator.answer(RagRequest(question="Where is NSU?")))

    assert "недостаточно информации" in response.answer.lower()
    assert orchestrator.query_processor.process_query.await_count == 0
    multilingual_retriever.retrieve_many.assert_awaited_once_with(["Where is NSU?"], 1)
    lexical_retriever.retrieve_many.assert_awaited_once_with(["Where is NSU?"], 1)


def test_chunk_rag_uses_rewritten_query_and_caps_search_query_fanout(chunk_metadata: ChunkMetadata):
    config = ChunkRagConfig(
        rewrite_enabled=True,
        hypothetical_doc_enabled=True,
        top_k_dense_multilingual=2,
        top_k_dense_russian=2,
        top_k_bm25=2,
    )

    chunk = Chunk(
        chunk_id="doc1_chunk_0",
        text="NSU is a university in Novosibirsk.",
        text_for_dense="Document: Test Doc\n\nNSU is a university in Novosibirsk.",
        text_for_bm25="document test doc nsu is a university in novosibirsk",
        metadata=chunk_metadata,
    )

    multilingual_retriever = AsyncMock()
    multilingual_retriever.retrieve_many.return_value = [[RetrievedChunk(chunk=chunk, score=0.9, source="multilingual_dense")]] * 5
    russian_retriever = AsyncMock()
    russian_retriever.retrieve_many.return_value = [[RetrievedChunk(chunk=chunk, score=0.8, source="russian_dense")]] * 5
    lexical_retriever = AsyncMock()
    lexical_retriever.retrieve_many.return_value = [[RetrievedChunk(chunk=chunk, score=1.2, source="lexical")]] * 4

    reranker = MagicMock()
    reranker.rerank = AsyncMock(
        return_value=QwenRerankResult(
            reranked_chunks=[RetrievedChunk(chunk=chunk, score=0.95, source="hybrid")],
            preview=[],
        )
    )

    orchestrator = ChunkRagOrchestrator(
        config=config,
        dense_retrievers={
            "multilingual_dense": multilingual_retriever,
            "russian_dense": russian_retriever,
        },
        lexical_retriever=lexical_retriever,
        reranker=reranker,
    )

    orchestrator.query_processor.process_query = AsyncMock(
        return_value=QueryRepresentations(
            original_query="Whre is NSU?",
            rewritten_query="Where is NSU?",
            resolved_coreferences="Where is Novosibirsk State University?",
            search_queries=["NSU location", "NSU address", "NSU campus"],
            hypothetical_document="NSU is located in Novosibirsk.",
            is_meaningful=True,
        )
    )
    orchestrator.assembler.assemble = MagicMock(return_value=("Документ: Test Doc\nNSU is a university in Novosibirsk.", []))
    orchestrator.generator.generate_answer = AsyncMock(return_value=("NSU is in Novosibirsk.", False))

    asyncio.run(orchestrator.answer(RagRequest(question="Whre is NSU?")))

    multilingual_retriever.retrieve_many.assert_awaited_once_with(
        [
            "Where is NSU?",
            "Where is Novosibirsk State University?",
            "NSU location",
            "NSU address",
            "NSU is located in Novosibirsk.",
        ],
        2,
    )
    lexical_retriever.retrieve_many.assert_awaited_once_with(
        [
            "Where is NSU?",
            "Where is Novosibirsk State University?",
            "NSU location",
            "NSU address",
        ],
        2,
    )
