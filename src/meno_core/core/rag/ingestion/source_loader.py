import json
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

ChunkBuilder = Callable[..., Awaitable[Any]]


def resolve_chunk_corpus_path(corpus_path: Path) -> Path | None:
    if corpus_path.exists():
        return corpus_path
    return None


async def load_chunks_from_compiled_corpus(
    path: Path,
    build_chunk_fn: ChunkBuilder | None = None,
) -> list[Any]:
    if build_chunk_fn is None:
        from meno_core.core.rag.ingestion.chunker import build_chunk
        build_chunk_fn = build_chunk

    chunks: list[Any] = []
    with path.open("r", encoding="utf-8") as fp:
        for idx, line in enumerate(fp):
            if not line.strip():
                continue

            if idx > 0 and idx % 500 == 0:
                logger.info("Preprocessed %s compiled corpus rows from %s...", idx, path)

            raw = json.loads(line)
            extra = dict(raw.get("extra") or {})
            if raw.get("published_at") is not None:
                extra.setdefault("published_at", raw["published_at"])
            if raw.get("published_at_label"):
                extra.setdefault("published_at_label", raw["published_at_label"])
            if raw.get("token_count") is not None:
                extra.setdefault("chunk_token_count", raw["token_count"])

            chunk = await build_chunk_fn(
                chunk_index=int(raw.get("chunk_index", 0)),
                text=raw.get("text", ""),
                document_id=raw.get("document_id") or f"unknown_doc_{idx}",
                document_title=raw.get("document_title") or raw.get("document_id") or "Unknown Document",
                source_url=raw.get("source_url"),
                extra_metadata=extra,
            )
            chunks.append(chunk)

    return chunks
