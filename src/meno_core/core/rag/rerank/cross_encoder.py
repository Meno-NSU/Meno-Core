import logging
from typing import List

from meno_core.config.settings import settings
from meno_core.core.rag.models import RetrievedChunk
from meno_core.core.rag.rerank.qwen_reranker import QwenCausalReranker, load_cached_qwen_reranker_backend

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks a list of chunks based on cross-encoder similarity with the query.
    """

    def __init__(self, filter_threshold: float = -10.0):
        self.delegate = QwenCausalReranker(
            backend=load_cached_qwen_reranker_backend(settings.reranker_path),
            filter_threshold=filter_threshold,
        )

    async def rerank(self, query: str, chunks: List[RetrievedChunk], top_n: int) -> List[RetrievedChunk]:
        """
        Backward-compatible wrapper around the causal-lm reranker backend.
        """
        try:
            result = await self.delegate.rerank(query=query, chunks=chunks, top_n=top_n)
            return result.reranked_chunks
        except Exception as error:
            logger.error("Error during reranking: %s", error, exc_info=True)
            return chunks[:top_n]
