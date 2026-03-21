import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import tiktoken

logger = logging.getLogger(__name__)

DEFAULT_INPUT_PATHS: tuple[Path, ...] = (
    Path("resources/data/kv_store_text_chunks.json"),
    Path("resources/data/merged_latest_knowledge.jsonl"),
    Path("resources/data/vk_scrapped_2012-04-03_to_2026-01-27.jsonl"),
    Path("resources/data/web_scrapped_2026-01-27.jsonl"),
)

SOURCE_PRIORITY: dict[str, int] = {
    "merged_latest_knowledge": 4,
    "kv_store_text_chunks": 3,
    "vk_scrapped": 2,
    "web_scrapped": 1,
}

_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
_BAD_PAGE_RE = re.compile(r"^#?\s*404\s+not\s+found(?:\s+nginx)?$", re.IGNORECASE)
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


@dataclass(slots=True)
class NormalizedDocument:
    source_url: str | None
    document_title: str
    published_at: int | None
    content: str
    source_kind: str
    source_file: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CorpusChunk:
    chunk_id: str
    document_id: str
    document_title: str
    chunk_index: int
    source_url: str | None
    published_at: int | None
    published_at_label: str
    text: str
    token_count: int
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BuildStats:
    input_documents: int
    deduplicated_documents: int
    emitted_chunks: int
    dropped_empty_documents: int
    dropped_404_documents: int


def count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    lines = [_WHITESPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    normalized = "\n".join(lines)
    normalized = _BLANK_LINES_RE.sub("\n\n", normalized)
    return normalized.strip()


def is_empty_content(text: str) -> bool:
    return not normalize_text(text)


def is_404_like(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False

    collapsed = " ".join(normalized.split()).strip().lower()
    if _BAD_PAGE_RE.match(collapsed):
        return True
    if collapsed.startswith("# 404 not found") and len(collapsed) <= 200:
        return True
    if collapsed in {"404 not found", "nginx", "# 404 not found nginx"}:
        return True
    return False


def infer_source_kind(path: Path) -> str:
    if path.name == "kv_store_text_chunks.json":
        return "kv_store_text_chunks"
    if path.name == "merged_latest_knowledge.jsonl":
        return "merged_latest_knowledge"
    if path.name.startswith("vk_scrapped_"):
        return "vk_scrapped"
    if path.name.startswith("web_scrapped_"):
        return "web_scrapped"
    return path.stem


def read_jsonl_documents(path: Path, source_kind: str) -> list[NormalizedDocument]:
    documents: list[NormalizedDocument] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            raw = json.loads(line)
            documents.append(
                NormalizedDocument(
                    source_url=raw.get("url"),
                    document_title=(raw.get("name") or "").strip() or "Untitled Document",
                    published_at=raw.get("date"),
                    content=raw.get("content") or "",
                    source_kind=source_kind,
                    source_file=str(path),
                    extra={
                        "collection_date": raw.get("collection_date"),
                    },
                )
            )
    return documents


def read_kv_store_documents(path: Path) -> list[NormalizedDocument]:
    with path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in raw.values():
        grouped.setdefault(entry.get("file_path", ""), []).append(entry)

    documents: list[NormalizedDocument] = []
    for source_url, entries in grouped.items():
        ordered_entries = sorted(entries, key=lambda item: item.get("chunk_order_index", 0))
        content = "\n\n".join((entry.get("content") or "").strip() for entry in ordered_entries if entry.get("content"))
        title = _extract_document_title_from_kv_content(content)

        legacy_full_doc_id = next(
            (entry.get("full_doc_id") for entry in ordered_entries if entry.get("full_doc_id")),
            None,
        )
        documents.append(
            NormalizedDocument(
                source_url=source_url or None,
                document_title=title,
                published_at=None,
                content=content,
                source_kind="kv_store_text_chunks",
                source_file=str(path),
                extra={
                    "legacy_full_doc_id": legacy_full_doc_id,
                    "kv_chunk_count": len(ordered_entries),
                    "kv_chunk_ids": [entry.get("_id") for entry in ordered_entries if entry.get("_id")],
                    "create_time": next((entry.get("create_time") for entry in ordered_entries if entry.get("create_time")), None),
                    "update_time": next((entry.get("update_time") for entry in ordered_entries if entry.get("update_time")), None),
                },
            )
        )

    return documents


def load_documents_from_path(path: Path | str) -> list[NormalizedDocument]:
    path = Path(path)
    source_kind = infer_source_kind(path)
    if source_kind == "kv_store_text_chunks":
        return read_kv_store_documents(path)
    return read_jsonl_documents(path, source_kind)


def load_documents(paths: Sequence[Path | str]) -> list[NormalizedDocument]:
    documents: list[NormalizedDocument] = []
    for path in paths:
        documents.extend(load_documents_from_path(path))
    return documents


def select_best_document(candidates: Sequence[NormalizedDocument]) -> NormalizedDocument:
    if not candidates:
        raise ValueError("Cannot select a document from an empty candidate list")

    selected = max(candidates, key=_selection_score)
    published_at = selected.published_at
    if published_at is None:
        dated_candidates = [candidate for candidate in candidates if candidate.published_at is not None]
        if dated_candidates:
            published_at = max(dated_candidates, key=_selection_score).published_at

    document_title = selected.document_title
    if not document_title.strip():
        titled_candidates = [candidate for candidate in candidates if candidate.document_title.strip()]
        if titled_candidates:
            document_title = max(titled_candidates, key=_selection_score).document_title
        else:
            document_title = "Untitled Document"

    legacy_full_doc_id = next(
        (
            candidate.extra.get("legacy_full_doc_id")
            for candidate in candidates
            if candidate.extra.get("legacy_full_doc_id")
        ),
        None,
    )

    merged_extra = dict(selected.extra)
    merged_extra.update(
        {
            "source_kind": selected.source_kind,
            "source_file": selected.source_file,
            "dedupe_candidates_count": len(candidates),
            "selected_from": selected.source_kind,
        }
    )
    if legacy_full_doc_id:
        merged_extra["legacy_full_doc_id"] = legacy_full_doc_id

    return NormalizedDocument(
        source_url=selected.source_url,
        document_title=document_title,
        published_at=published_at,
        content=selected.content,
        source_kind=selected.source_kind,
        source_file=selected.source_file,
        extra=merged_extra,
    )


def deduplicate_documents(documents: Sequence[NormalizedDocument]) -> list[NormalizedDocument]:
    grouped: dict[str, list[NormalizedDocument]] = {}
    for document in documents:
        grouped.setdefault(_document_group_key(document), []).append(document)

    selected_documents = [select_best_document(candidates) for candidates in grouped.values()]
    selected_documents.sort(key=lambda doc: (_document_group_key(doc), doc.document_title))
    return selected_documents


def filter_documents(
    documents: Sequence[NormalizedDocument],
    *,
    drop_empty: bool = True,
    drop_404: bool = True,
) -> tuple[list[NormalizedDocument], int, int]:
    filtered: list[NormalizedDocument] = []
    dropped_empty = 0
    dropped_404 = 0

    for document in documents:
        if drop_empty and is_empty_content(document.content):
            dropped_empty += 1
            continue
        if drop_404 and is_404_like(document.content):
            dropped_404 += 1
            continue
        filtered.append(document)

    return filtered, dropped_empty, dropped_404


def build_chunk_prefix(document: NormalizedDocument) -> str:
    source = document.source_url or "неизвестен"
    return f"Дата публикации: {format_published_at_label(document.published_at)}\nИсточник: {source}\n\n"


def split_into_chunks(document: NormalizedDocument, max_tokens: int) -> list[CorpusChunk]:
    prefix = build_chunk_prefix(document)
    prefix_tokens = count_tokens(prefix)
    if prefix_tokens >= max_tokens:
        raise ValueError(
            f"Chunk prefix for document {document.source_url or document.document_title!r} "
            f"already uses {prefix_tokens} tokens, which exceeds the limit {max_tokens}"
        )

    body_budget = max_tokens - prefix_tokens
    body_chunks = _build_body_chunks(document.content, body_budget)
    if not body_chunks:
        body_chunks = [""]

    document_id = _build_document_id(document)
    published_at_label = format_published_at_label(document.published_at)
    emitted_chunks: list[CorpusChunk] = []

    for chunk_index, body in enumerate(body_chunks):
        text = prefix + body if body else prefix.rstrip()
        token_count = count_tokens(text)
        if token_count > max_tokens:
            raise ValueError(
                f"Chunk {document_id}_chunk_{chunk_index} is {token_count} tokens long, "
                f"which exceeds the limit {max_tokens}"
            )

        emitted_chunks.append(
            CorpusChunk(
                chunk_id=f"{document_id}_chunk_{chunk_index}",
                document_id=document_id,
                document_title=document.document_title,
                chunk_index=chunk_index,
                source_url=document.source_url,
                published_at=document.published_at,
                published_at_label=published_at_label,
                text=text,
                token_count=token_count,
                extra=dict(document.extra),
            )
        )

    return emitted_chunks


def build_corpus_chunks(documents: Sequence[NormalizedDocument], max_tokens: int) -> list[CorpusChunk]:
    chunks: list[CorpusChunk] = []
    for document in documents:
        chunks.extend(split_into_chunks(document, max_tokens=max_tokens))
    return chunks


def write_corpus_jsonl(chunks: Iterable[CorpusChunk], output_path: Path | str) -> int:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    emitted = 0
    with output_path.open("w", encoding="utf-8") as fp:
        for chunk in chunks:
            fp.write(
                json.dumps(
                    {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "document_title": chunk.document_title,
                        "chunk_index": chunk.chunk_index,
                        "source_url": chunk.source_url,
                        "published_at": chunk.published_at,
                        "published_at_label": chunk.published_at_label,
                        "text": chunk.text,
                        "token_count": chunk.token_count,
                        "extra": chunk.extra,
                    },
                    ensure_ascii=False,
                )
            )
            fp.write("\n")
            emitted += 1

    return emitted


def build_chunk_rag_corpus(
    input_paths: Sequence[Path | str] = DEFAULT_INPUT_PATHS,
    *,
    output_jsonl: Path | str,
    max_tokens: int = 512,
    drop_empty: bool = True,
    drop_404: bool = True,
) -> BuildStats:
    documents = load_documents(input_paths)
    deduplicated_documents = deduplicate_documents(documents)
    filtered_documents, dropped_empty, dropped_404 = filter_documents(
        deduplicated_documents,
        drop_empty=drop_empty,
        drop_404=drop_404,
    )
    chunks = build_corpus_chunks(filtered_documents, max_tokens=max_tokens)
    emitted_chunks = write_corpus_jsonl(chunks, output_path=output_jsonl)

    stats = BuildStats(
        input_documents=len(documents),
        deduplicated_documents=len(filtered_documents),
        emitted_chunks=emitted_chunks,
        dropped_empty_documents=dropped_empty,
        dropped_404_documents=dropped_404,
    )
    logger.info(
        "Built chunk corpus: %s input documents -> %s documents -> %s chunks",
        stats.input_documents,
        stats.deduplicated_documents,
        stats.emitted_chunks,
    )
    return stats


def format_published_at_label(published_at: int | None) -> str:
    if published_at is None:
        return "неизвестна"
    return datetime.fromtimestamp(published_at, tz=timezone.utc).strftime("%d.%m.%Y")


def _extract_document_title_from_kv_content(content: str) -> str:
    normalized = normalize_text(content)
    if not normalized:
        return "Untitled Document"
    first_paragraph = _PARAGRAPH_SPLIT_RE.split(normalized)[0]
    lines = [line.strip() for line in first_paragraph.splitlines() if line.strip()]
    if not lines:
        return "Untitled Document"
    return lines[0][:300]


def _selection_score(document: NormalizedDocument) -> tuple[int, int, int, int, int]:
    normalized = normalize_text(document.content)
    non_empty = int(bool(normalized))
    not_404 = int(not is_404_like(normalized))
    useful_length = len(normalized) if non_empty and not_404 else 0
    has_published_at = int(document.published_at is not None)
    source_priority = SOURCE_PRIORITY.get(document.source_kind, 0)
    return non_empty, not_404, useful_length, has_published_at, source_priority


def _document_group_key(document: NormalizedDocument) -> str:
    if document.source_url:
        return document.source_url
    fallback = f"{document.document_title}\n{normalize_text(document.content)}"
    digest = hashlib.sha1(fallback.encode("utf-8")).hexdigest()
    return f"missing-source:{digest}"


def _build_document_id(document: NormalizedDocument) -> str:
    source = document.source_url or _document_group_key(document)
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()
    return f"doc-{digest}"


def _build_body_chunks(content: str, max_body_tokens: int) -> list[str]:
    normalized = normalize_text(content)
    if not normalized:
        return []

    paragraphs = [part.strip() for part in _PARAGRAPH_SPLIT_RE.split(normalized) if part.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_parts: list[str] = []

    for paragraph in paragraphs:
        if count_tokens(paragraph) > max_body_tokens:
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
            chunks.extend(_split_large_paragraph(paragraph, max_body_tokens))
            continue

        candidate_parts = current_parts + [paragraph]
        candidate_text = "\n\n".join(candidate_parts)
        if count_tokens(candidate_text) <= max_body_tokens:
            current_parts.append(paragraph)
        else:
            if current_parts:
                chunks.append("\n\n".join(current_parts))
            current_parts = [paragraph]

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


def _split_large_paragraph(paragraph: str, max_body_tokens: int) -> list[str]:
    sentences = [sentence.strip() for sentence in _SENTENCE_SPLIT_RE.split(paragraph) if sentence.strip()]
    if len(sentences) <= 1:
        return _split_by_tokens(paragraph, max_body_tokens)

    parts: list[str] = []
    current = ""
    for sentence in sentences:
        if count_tokens(sentence) > max_body_tokens:
            if current:
                parts.append(current)
                current = ""
            parts.extend(_split_by_tokens(sentence, max_body_tokens))
            continue

        candidate = sentence if not current else f"{current} {sentence}"
        if count_tokens(candidate) <= max_body_tokens:
            current = candidate
        else:
            if current:
                parts.append(current)
            current = sentence

    if current:
        parts.append(current)

    return parts


def _split_by_tokens(text: str, max_body_tokens: int) -> list[str]:
    tokens = _TOKENIZER.encode(text)
    parts: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_body_tokens, len(tokens))
        part = _TOKENIZER.decode(tokens[start:end]).strip()
        while part and count_tokens(part) > max_body_tokens and end > start + 1:
            end -= 1
            part = _TOKENIZER.decode(tokens[start:end]).strip()
        if part:
            parts.append(part)
        start = end
    return parts
