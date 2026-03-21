import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from meno_core.core.rag.ingestion.corpus_builder import (  # noqa: E402
    NormalizedDocument,
    build_chunk_rag_corpus,
    count_tokens,
    deduplicate_documents,
    filter_documents,
    format_published_at_label,
    load_documents_from_path,
    split_into_chunks,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False))
            fp.write("\n")


def test_readers_parse_all_supported_formats(tmp_path: Path) -> None:
    merged_path = tmp_path / "merged_latest_knowledge.jsonl"
    _write_jsonl(
        merged_path,
        [
            {
                "url": "https://example.org/merged",
                "name": "Merged title",
                "content": "Merged content",
                "date": 1_700_000_000,
                "collection_date": 1_700_000_100,
            }
        ],
    )
    vk_path = tmp_path / "vk_scrapped_2012-04-03_to_2026-01-27.jsonl"
    _write_jsonl(
        vk_path,
        [
            {
                "url": "https://example.org/vk",
                "name": "VK title",
                "content": "VK content",
                "date": 1_700_000_001,
                "collection_date": 1_700_000_101,
            }
        ],
    )
    web_path = tmp_path / "web_scrapped_2026-01-27.jsonl"
    _write_jsonl(
        web_path,
        [
            {
                "url": "https://example.org/web",
                "name": "Web title",
                "content": "Web content",
                "date": None,
                "collection_date": 1_700_000_102,
            }
        ],
    )
    kv_path = tmp_path / "kv_store_text_chunks.json"
    kv_payload = {
        "chunk-1": {
            "tokens": 12,
            "content": "KV Title\n\nPart 1",
            "chunk_order_index": 1,
            "full_doc_id": "doc-kv-1",
            "file_path": "https://example.org/kv",
            "llm_cache_list": [],
            "create_time": 1,
            "update_time": 2,
            "_id": "chunk-1",
        },
        "chunk-0": {
            "tokens": 12,
            "content": "KV Title\n\nPart 0",
            "chunk_order_index": 0,
            "full_doc_id": "doc-kv-1",
            "file_path": "https://example.org/kv",
            "llm_cache_list": [],
            "create_time": 1,
            "update_time": 2,
            "_id": "chunk-0",
        },
    }
    kv_path.write_text(json.dumps(kv_payload, ensure_ascii=False), encoding="utf-8")

    merged_docs = load_documents_from_path(merged_path)
    vk_docs = load_documents_from_path(vk_path)
    web_docs = load_documents_from_path(web_path)
    kv_docs = load_documents_from_path(kv_path)

    assert len(merged_docs) == 1
    assert merged_docs[0].source_kind == "merged_latest_knowledge"
    assert merged_docs[0].published_at == 1_700_000_000

    assert len(vk_docs) == 1
    assert vk_docs[0].source_kind == "vk_scrapped"

    assert len(web_docs) == 1
    assert web_docs[0].source_kind == "web_scrapped"

    assert len(kv_docs) == 1
    assert kv_docs[0].source_kind == "kv_store_text_chunks"
    assert kv_docs[0].published_at is None
    assert kv_docs[0].source_url == "https://example.org/kv"
    assert kv_docs[0].document_title == "KV Title"
    assert kv_docs[0].content == "KV Title\n\nPart 0\n\nKV Title\n\nPart 1"
    assert kv_docs[0].extra["legacy_full_doc_id"] == "doc-kv-1"


def test_deduplicate_prefers_non_404_and_more_complete_text() -> None:
    candidates = [
        NormalizedDocument(
            source_url="https://example.org/a",
            document_title="Bad merged copy",
            published_at=None,
            content="# 404 Not Found\nnginx\n",
            source_kind="merged_latest_knowledge",
            source_file="merged",
        ),
        NormalizedDocument(
            source_url="https://example.org/a",
            document_title="Good kv copy",
            published_at=None,
            content="Полезный текст документа. " * 20,
            source_kind="kv_store_text_chunks",
            source_file="kv",
        ),
    ]

    selected = deduplicate_documents(candidates)

    assert len(selected) == 1
    assert selected[0].source_kind == "kv_store_text_chunks"
    assert "Полезный текст" in selected[0].content
    assert selected[0].extra["dedupe_candidates_count"] == 2
    assert selected[0].extra["selected_from"] == "kv_store_text_chunks"


def test_deduplicate_fills_missing_date_from_other_candidate() -> None:
    candidates = [
        NormalizedDocument(
            source_url="https://example.org/date-fill",
            document_title="Longer copy without date",
            published_at=None,
            content="Очень длинный полезный текст. " * 40,
            source_kind="merged_latest_knowledge",
            source_file="merged",
        ),
        NormalizedDocument(
            source_url="https://example.org/date-fill",
            document_title="Shorter copy with date",
            published_at=1_700_000_123,
            content="Короткий полезный текст.",
            source_kind="vk_scrapped",
            source_file="vk",
        ),
    ]

    selected = deduplicate_documents(candidates)

    assert len(selected) == 1
    assert selected[0].published_at == 1_700_000_123


def test_split_into_chunks_adds_prefix_and_respects_512_token_limit() -> None:
    paragraph = " ".join(f"слово{i}" for i in range(220))
    content = "\n\n".join([paragraph, paragraph, paragraph])
    document = NormalizedDocument(
        source_url="https://example.org/long",
        document_title="Long Document",
        published_at=1_700_000_456,
        content=content,
        source_kind="merged_latest_knowledge",
        source_file="merged",
    )

    chunks = split_into_chunks(document, max_tokens=512)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.startswith("Дата публикации: 14.11.2023\nИсточник: https://example.org/long\n\n")
        assert chunk.token_count <= 512
        assert count_tokens(chunk.text) == chunk.token_count


def test_filter_documents_drops_empty_and_404_pages() -> None:
    documents = [
        NormalizedDocument(
            source_url="https://example.org/empty",
            document_title="Empty",
            published_at=None,
            content=" \n ",
            source_kind="web_scrapped",
            source_file="web",
        ),
        NormalizedDocument(
            source_url="https://example.org/404",
            document_title="404",
            published_at=None,
            content="# 404 Not Found\nnginx\n",
            source_kind="web_scrapped",
            source_file="web",
        ),
        NormalizedDocument(
            source_url="https://example.org/good",
            document_title="Good",
            published_at=None,
            content="Нормальный текст",
            source_kind="web_scrapped",
            source_file="web",
        ),
    ]

    filtered, dropped_empty, dropped_404 = filter_documents(documents)

    assert [doc.source_url for doc in filtered] == ["https://example.org/good"]
    assert dropped_empty == 1
    assert dropped_404 == 1


def test_build_chunk_rag_corpus_smoke_from_all_formats(tmp_path: Path) -> None:
    kv_path = tmp_path / "kv_store_text_chunks.json"
    kv_payload = {
        "kv-a-0": {
            "tokens": 8,
            "content": "KV title A\n\nПолезный KV контент для URL A.",
            "chunk_order_index": 0,
            "full_doc_id": "doc-a",
            "file_path": "https://example.org/a",
            "llm_cache_list": [],
            "create_time": 1,
            "update_time": 2,
            "_id": "kv-a-0",
        },
        "kv-e-0": {
            "tokens": 8,
            "content": "KV title E\n\nУникальный KV документ.",
            "chunk_order_index": 0,
            "full_doc_id": "doc-e",
            "file_path": "https://example.org/e",
            "llm_cache_list": [],
            "create_time": 1,
            "update_time": 2,
            "_id": "kv-e-0",
        },
    }
    kv_path.write_text(json.dumps(kv_payload, ensure_ascii=False), encoding="utf-8")

    merged_path = tmp_path / "merged_latest_knowledge.jsonl"
    _write_jsonl(
        merged_path,
        [
            {
                "url": "https://example.org/a",
                "name": "Merged A",
                "content": "# 404 Not Found\nnginx\n",
                "date": None,
                "collection_date": 11,
            },
            {
                "url": "https://example.org/b",
                "name": "Merged B",
                "content": "Полезный текст B. " * 10,
                "date": None,
                "collection_date": 12,
            },
        ],
    )

    vk_path = tmp_path / "vk_scrapped_2012-04-03_to_2026-01-27.jsonl"
    _write_jsonl(
        vk_path,
        [
            {
                "url": "https://example.org/b",
                "name": "VK B",
                "content": "Короткий текст B.",
                "date": 1_700_001_000,
                "collection_date": 13,
            },
            {
                "url": "https://example.org/c",
                "name": "VK C",
                "content": "Полезный текст C.",
                "date": 1_700_001_001,
                "collection_date": 14,
            },
        ],
    )

    web_path = tmp_path / "web_scrapped_2026-01-27.jsonl"
    _write_jsonl(
        web_path,
        [
            {
                "url": "https://example.org/d",
                "name": "Web D",
                "content": "\n",
                "date": None,
                "collection_date": 15,
            }
        ],
    )

    output_path = tmp_path / "chunk_rag_corpus_512.jsonl"
    stats = build_chunk_rag_corpus(
        [kv_path, merged_path, vk_path, web_path],
        output_jsonl=output_path,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    urls = {row["source_url"] for row in rows}
    rows_by_url = {row["source_url"]: row for row in rows}

    assert stats.input_documents == 7
    assert stats.deduplicated_documents == 4
    assert stats.dropped_empty_documents == 1
    assert stats.dropped_404_documents == 0
    assert urls == {
        "https://example.org/a",
        "https://example.org/b",
        "https://example.org/c",
        "https://example.org/e",
    }
    assert rows_by_url["https://example.org/a"]["published_at"] is None
    assert rows_by_url["https://example.org/a"]["extra"]["legacy_full_doc_id"] == "doc-a"
    assert rows_by_url["https://example.org/b"]["published_at"] == 1_700_001_000
    assert rows_by_url["https://example.org/b"]["published_at_label"] == format_published_at_label(1_700_001_000)
    assert all(row["token_count"] <= 512 for row in rows)
