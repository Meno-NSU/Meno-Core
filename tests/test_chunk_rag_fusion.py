from meno_core.core.rag.fusion.merger import HybridFusion
from meno_core.core.rag.models import Chunk, ChunkMetadata, RetrievedChunk


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=f"text for {chunk_id}",
        text_for_dense=f"dense {chunk_id}",
        text_for_bm25=f"bm25 {chunk_id}",
        metadata=ChunkMetadata(
            document_id="doc",
            document_title="Doc",
            chunk_index=0,
            source_url="https://example.com",
        ),
    )


def test_hybrid_fusion_normalizes_each_retriever_and_applies_weights():
    chunk_a = _chunk("chunk-a")
    chunk_b = _chunk("chunk-b")
    chunk_c = _chunk("chunk-c")

    fusion = HybridFusion(
        weights={
            "multilingual_dense": 0.5,
            "russian_dense": 0.3,
            "lexical": 0.2,
        },
        preview_k=3,
    )

    result = fusion.fuse(
        {
            "multilingual_dense": [
                RetrievedChunk(chunk=chunk_a, score=0.2, source="multilingual_dense"),
                RetrievedChunk(chunk=chunk_b, score=0.9, source="multilingual_dense"),
            ],
            "russian_dense": [
                RetrievedChunk(chunk=chunk_a, score=0.7, source="russian_dense"),
                RetrievedChunk(chunk=chunk_c, score=0.1, source="russian_dense"),
            ],
            "lexical": [
                RetrievedChunk(chunk=chunk_c, score=10.0, source="lexical"),
            ],
        },
        top_k=3,
    )

    assert [item.chunk.chunk_id for item in result.chunks] == ["chunk-b", "chunk-a", "chunk-c"]
    assert result.chunks[0].score == 0.5
    assert result.chunks[1].score == 0.3
    assert result.chunks[2].score == 0.2
    assert result.fused_preview[0]["chunk_id"] == "chunk-b"
