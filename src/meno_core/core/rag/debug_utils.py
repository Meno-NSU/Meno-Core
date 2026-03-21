from typing import Any, List

from meno_core.core.rag.models import RetrievedChunk


def build_retrieved_chunk_preview(chunks: List[RetrievedChunk], limit: int) -> List[dict[str, Any]]:
    preview: list[dict[str, Any]] = []
    for chunk_wrapper in chunks[:limit]:
        chunk = chunk_wrapper.chunk
        meta = chunk.metadata
        preview.append(
            {
                "chunk_id": chunk.chunk_id,
                "score": float(chunk_wrapper.score),
                "source": chunk_wrapper.source,
                "document_id": meta.document_id,
                "document_title": meta.document_title,
                "chunk_index": meta.chunk_index,
                "source_url": meta.source_url,
                "text_preview": chunk.text[:180],
            }
        )
    return preview
