import logging
import time
from typing import List, Dict

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
        chunk_map: Dict[str, dict],
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

    async def retrieve(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """
        Returns top-K retrieved chunks by dense vector similarity.
        """
        try:
            started_at = time.perf_counter()
            # 1. Embed query
            embed_started_at = time.perf_counter()
            q_vec = self.embedder.encode_queries([query])[0].detach().cpu().numpy().tolist()
            embed_latency_ms = (time.perf_counter() - embed_started_at) * 1000

            # 2. Query zvec
            search_started_at = time.perf_counter()
            zvec_results = self.collection.query(
                zvec.VectorQuery("embedding", vector=q_vec),
                topk=top_k
            )
            search_latency_ms = (time.perf_counter() - search_started_at) * 1000

            # 3. Map back to RetrievedChunk
            mapping_started_at = time.perf_counter()
            results = []
            for r in zvec_results:
                doc_id = str(r['id'])
                score = float(r.get('score', 0.0))
                
                if doc_id in self.chunk_map:
                    raw_chunk = self.chunk_map[doc_id]
                    # ensure valid Chunk model
                    chunk_obj = Chunk(**raw_chunk)
                    results.append(
                        RetrievedChunk(
                            chunk=chunk_obj,
                            score=score,
                            source=self.name
                        )
                    )
                else:
                    logger.warning(f"Zvec returned ID {doc_id} but it's not in chunk map.")

            mapping_latency_ms = (time.perf_counter() - mapping_started_at) * 1000
            total_latency_ms = (time.perf_counter() - started_at) * 1000

            if self.debug_enabled:
                retrieval_logger.info(
                    "retriever=%s device=%s query=%r top_k=%s hits=%s latency_ms=%.2f embed_ms=%.2f search_ms=%.2f map_ms=%.2f",
                    self.name,
                    self.embedder.device,
                    query,
                    top_k,
                    len(results),
                    total_latency_ms,
                    embed_latency_ms,
                    search_latency_ms,
                    mapping_latency_ms,
                )

            if self.debug_enabled:
                retrieval_logger.debug(
                    "retriever=%s query=%r top_k=%s preview=%s",
                    self.name,
                    query,
                    top_k,
                    build_retrieved_chunk_preview(results, self.preview_k),
                )
            return results

        except Exception as e:
            logger.error(f"Error during Dense Retrieval: {e}", exc_info=True)
            return []
