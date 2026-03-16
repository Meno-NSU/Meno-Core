import logging
from typing import List, Dict, Tuple

import tiktoken  # type: ignore[import-untyped]

from meno_core.core.rag.models import RetrievedChunk, RagSource

logger = logging.getLogger(__name__)

try:
    _encoding = tiktoken.get_encoding("cl100k_base")
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

    def assemble(self, chunks: List[RetrievedChunk]) -> Tuple[str, List[RagSource]]:
        """
        Greedy packing of chunks up to token_budget.
        Returns:
            context_string: Text to be injected into the prompt.
            sources: List of RagSource models used in this context.
        """
        if not chunks:
            return "", []

        # 1. Greedy select up to budget
        selected_chunks = []
        current_tokens = 0
        
        for c in chunks:
            text = c.chunk.text
            tok_count = estimate_tokens(text)
            
            # Allow at least one chunk even if it exceeds budget slightly
            if current_tokens + tok_count > self.token_budget and current_tokens > 0:
                continue
                
            selected_chunks.append(c)
            current_tokens += tok_count
            
        # 2. Group by document
        doc_grouped: Dict[str, dict] = {}
        for c in selected_chunks:
            meta = c.chunk.metadata
            doc_id = meta.document_id
            
            if doc_id not in doc_grouped:
                doc_grouped[doc_id] = {
                    "title": meta.document_title,
                    "url": meta.source_url or "",
                    "chunks": []
                }
            
            doc_grouped[doc_id]["chunks"].append(c.chunk)

        # 3. Format Context String and prepare sources
        context_parts = []
        sources = []
        
        for doc_id, data in doc_grouped.items():
            title = data["title"]
            url = data["url"]
            group_chunks = data["chunks"]
            
            # Sort chunks in original text order if they have index
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
