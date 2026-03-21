import logging
import time
from typing import List, Dict, Sequence

import zvec

from meno_core.core.rag.debug_utils import build_retrieved_chunk_preview
from meno_core.core.rag.embeddings import DenseEmbedder
from meno_core.core.rag.models import Chunk, RetrievedChunk
from meno_core.core.rag.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)
retrieval_logger = logging.getLogger("chunk_rag.retrieval")


class ZvecDenseRetriever(BaseRetriever):
    """
    Retrieves dense vector embeddings using the Zvec engine.
    """

    def __init__(
        self,
        name: str,
        embedder: DenseEmbedder,
        collection: zvec.Collection,
        chunk_map: Dict[str, Chunk],
        *,
        debug_enabled: bool = False,
        preview_k: int = 5,
    ):
        self.name = name
        self.embedder = embedder
        self.collection = collection
        self.chunk_map = chunk_map
        self.debug_enabled = debug_enabled
        self.preview_k = preview_k

    async def retrieve_many(self, queries: Sequence[str], top_k: int) -> List[List[RetrievedChunk]]:
        """
        Returns top-K retrieved chunks by dense vector similarity for each query.
        """
        try:
            started_at = time.perf_counter()
            if not queries:
                return []

            embed_started_at = time.perf_counter()
            q_vectors = self.embedder.encode_queries(list(queries)).detach().cpu().numpy().tolist()
            embed_latency_ms = (time.perf_counter() - embed_started_at) * 1000

            search_started_at = time.perf_counter()
            zvec_results_by_query = [
                self.collection.query(zvec.VectorQuery("embedding", vector=q_vec), topk=top_k)
                for q_vec in q_vectors
            ]
            search_latency_ms = (time.perf_counter() - search_started_at) * 1000

            mapping_started_at = time.perf_counter()
            results_by_query: list[list[RetrievedChunk]] = []
            for zvec_results in zvec_results_by_query:
                query_results: list[RetrievedChunk] = []
                for r in zvec_results:
                    doc_id = str(r["id"])  # type: ignore[index]
                    score = float(r.get("score", 0.0))  # type: ignore[union-attr]
                    chunk_obj = self.chunk_map.get(doc_id)
                    if chunk_obj is None:
                        logger.warning("Zvec returned ID %s but it's not in chunk map.", doc_id)
                        continue
                    query_results.append(
                        RetrievedChunk(
                            chunk=chunk_obj,
                            score=score,
                            source=self.name  # type: ignore[arg-type]
                        )
                    )
                results_by_query.append(query_results)

            mapping_latency_ms = (time.perf_counter() - mapping_started_at) * 1000
            total_latency_ms = (time.perf_counter() - started_at) * 1000

            if self.debug_enabled:
                retrieval_logger.info(
                    "retriever=%s device=%s queries=%s top_k=%s hits=%s latency_ms=%.2f embed_ms=%.2f search_ms=%.2f map_ms=%.2f",
                    self.name,
                    self.embedder.device,
                    len(queries),
                    top_k,
                    [len(results) for results in results_by_query],
                    total_latency_ms,
                    embed_latency_ms,
                    search_latency_ms,
                    mapping_latency_ms,
                )

            if self.debug_enabled and queries:
                retrieval_logger.debug(
                    "retriever=%s first_query=%r top_k=%s preview=%s",
                    self.name,
                    queries[0],
                    top_k,
                    build_retrieved_chunk_preview(results_by_query[0], self.preview_k) if results_by_query else [],
                )
            return results_by_query

        except Exception as e:
            logger.error(f"Error during Dense Retrieval: {e}", exc_info=True)
            return [[] for _ in queries]
