import logging
from dataclasses import dataclass

from meno_core.config.settings import settings
from meno_core.core.gte_embedding import GTEEmbedding
from meno_core.core.rag.embeddings import MultilingualDenseEmbedder, User2DenseEmbedder
from meno_core.core.rag.rerank.qwen_reranker import QwenRerankerBackend, load_cached_qwen_reranker_backend

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ChunkRagModelRegistry:
    multilingual_dense: MultilingualDenseEmbedder
    russian_dense: User2DenseEmbedder
    reranker: QwenRerankerBackend


_cached_registry: ChunkRagModelRegistry | None = None
_cached_key: tuple[str, str, str] | None = None


def load_chunk_rag_model_registry(multilingual_embedder: GTEEmbedding) -> ChunkRagModelRegistry:
    global _cached_registry, _cached_key

    cache_key = (
        settings.multilingual_embedder_path,
        settings.rus_embedder_path,
        settings.reranker_path,
    )
    if _cached_registry is not None and _cached_key == cache_key:
        return _cached_registry

    logger.info("Loading Chunk-RAG model registry...")
    registry = ChunkRagModelRegistry(
        multilingual_dense=MultilingualDenseEmbedder(
            base_embedder=multilingual_embedder,
            model_path=settings.multilingual_embedder_path,
        ),
        russian_dense=User2DenseEmbedder.from_pretrained(settings.rus_embedder_path),
        reranker=load_cached_qwen_reranker_backend(settings.reranker_path),
    )
    _cached_registry = registry
    _cached_key = cache_key
    return registry
