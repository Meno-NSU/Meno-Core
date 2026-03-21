import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from meno_core.core.rag.ingestion.source_loader import (
    load_chunks_from_compiled_corpus,
    resolve_chunk_corpus_path,
)


async def _fake_build_chunk(**kwargs):
    return kwargs


def test_resolve_chunk_corpus_path_returns_existing_path(tmp_path: Path) -> None:
    corpus_path = tmp_path / "chunk_rag_corpus_512.jsonl"
    corpus_path.write_text("", encoding="utf-8")

    resolved = resolve_chunk_corpus_path(corpus_path=corpus_path)

    assert resolved == corpus_path


def test_load_chunks_from_compiled_corpus_uses_ready_text_without_reappending_source(tmp_path: Path) -> None:
    corpus_path = tmp_path / "chunk_rag_corpus_512.jsonl"
    row = {
        "chunk_id": "doc-1_chunk_0",
        "document_id": "doc-1",
        "document_title": "Doc 1",
        "chunk_index": 0,
        "source_url": "https://example.org/doc-1",
        "published_at": 1_700_000_000,
        "published_at_label": "14.11.2023",
        "text": "Дата публикации: 14.11.2023\nИсточник: https://example.org/doc-1\n\nГотовый текст",
        "token_count": 42,
        "extra": {
            "source_kind": "merged_latest_knowledge",
            "selected_from": "merged_latest_knowledge",
        },
    }
    corpus_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    chunks = asyncio.run(load_chunks_from_compiled_corpus(corpus_path, build_chunk_fn=_fake_build_chunk))

    assert len(chunks) == 1
    chunk_kwargs = chunks[0]
    assert chunk_kwargs["text"] == row["text"]
    assert chunk_kwargs["document_id"] == "doc-1"
    assert chunk_kwargs["document_title"] == "Doc 1"
    assert chunk_kwargs["source_url"] == "https://example.org/doc-1"
    assert chunk_kwargs["extra_metadata"]["published_at"] == 1_700_000_000
    assert chunk_kwargs["extra_metadata"]["published_at_label"] == "14.11.2023"
    assert chunk_kwargs["extra_metadata"]["chunk_token_count"] == 42
