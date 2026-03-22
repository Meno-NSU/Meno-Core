import logging
from typing import List, Dict, Tuple

import tiktoken  # type: ignore[import-untyped]

from meno_core.core.rag.models import RetrievedChunk, RagSource

logger = logging.getLogger(__name__)

try:
    _encoding: tiktoken.Encoding | None = tiktoken.get_encoding("cl100k_base")
except Exception:
    _encoding = None


def estimate_tokens(text: str) -> int:
    """Estimated token count."""
    if _encoding is not None:
        return len(_encoding.encode(text))
    return len(text.split()) * 2  # Naive fallback


class ContextAssembler:
    """
    Assembles a token-budget constrained context string from reranked chunks,
    grouping them by document to minimize repetition of titles and urls.
    """

    def __init__(self, token_budget: int = 4000):
        self.token_budget = token_budget

    # Approximate token overhead per separator between documents
    _SEPARATOR_OVERHEAD = estimate_tokens("\n\n---\n\n") if _encoding else 4

    def assemble(self, chunks: List[RetrievedChunk]) -> Tuple[str, List[RagSource]]:
        """
        Greedy packing of chunks up to token_budget.
        Uses incremental token counting to avoid O(n²) re-rendering.
        Returns:
            context_string: Text to be injected into the prompt.
            sources: List of RagSource models used in this context.
        """
        if not chunks:
            return "", []

        doc_order: list[str] = []
        doc_grouped: Dict[str, dict] = {}
        running_tokens = 0

        selected_chunk_count = 0
        for c in chunks:
            meta = c.chunk.metadata
            doc_id = meta.document_id
            is_new_doc = doc_id not in doc_grouped

            # Estimate the token cost of adding this chunk
            added_tokens = 0
            if is_new_doc:
                header = f"Документ: {meta.document_title}\n"
                url = meta.source_url or ""
                if url:
                    header += f"Источник: {url}\n"
                added_tokens += estimate_tokens(header)
                # Separator between documents (except the first)
                if doc_order:
                    added_tokens += self._SEPARATOR_OVERHEAD

            if meta.section_title:
                chunk_line = f"Раздел [{meta.section_title}]: {c.chunk.text}\n"
            else:
                chunk_line = f"{c.chunk.text}\n"
            added_tokens += estimate_tokens(chunk_line)

            candidate_tokens = running_tokens + added_tokens
            if candidate_tokens > self.token_budget and selected_chunk_count > 0:
                # Over budget and we already have at least one chunk — skip
                continue

            # Accept chunk
            if is_new_doc:
                doc_order.append(doc_id)
                doc_grouped[doc_id] = {
                    "title": meta.document_title,
                    "url": meta.source_url or "",
                    "chunks": []
                }

            doc_grouped[doc_id]["chunks"].append(c.chunk)
            running_tokens = candidate_tokens
            selected_chunk_count += 1

        return self._render(doc_order, doc_grouped)

    def _render(self, doc_order: List[str], doc_grouped: Dict[str, dict]) -> Tuple[str, List[RagSource]]:
        if not doc_order:
            return "", []

        context_parts = []
        sources = []

        for doc_id in doc_order:
            data = doc_grouped[doc_id]
            title = data["title"]
            url = data["url"]
            group_chunks = list(data["chunks"])

            group_chunks.sort(key=lambda x: x.metadata.chunk_index)

            doc_str = f"Документ: {title}\n"
            if url:
                doc_str += f"Источник: {url}\n"

            chunk_ids = []
            for chunk in group_chunks:
                if chunk.metadata.section_title:
                    doc_str += f"Раздел [{chunk.metadata.section_title}]: {chunk.text}\n"
                else:
                    doc_str += f"{chunk.text}\n"
                chunk_ids.append(chunk.chunk_id)

            context_parts.append(doc_str.strip())

            sources.append(RagSource(
                document_id=doc_id,
                document_title=title,
                source_url=url,
                chunk_ids=chunk_ids
            ))

        final_context = "\n\n---\n\n".join(context_parts)
        return final_context, sources
