import logging
import time
from typing import List, Dict

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from meno_core.core.lexical_normalizer import tokenize_for_bm25
from meno_core.core.rag.debug_utils import build_retrieved_chunk_preview
from meno_core.core.rag.models import Chunk, RetrievedChunk
from meno_core.core.rag.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)
retrieval_logger = logging.getLogger("chunk_rag.retrieval")


class BM25LexicalRetriever(BaseRetriever):
    """
    Retrieves text chunks using a lexical BM25 engine.
    """

    def __init__(
        self,
        bm25: BM25Okapi,
        chunk_map: Dict[str, dict],
        *,
        debug_enabled: bool = False,
        preview_k: int = 5,
    ):
        self.bm25 = bm25
        self.chunk_map = chunk_map
        self.debug_enabled = debug_enabled
        self.preview_k = preview_k
        # Build map back from sequential 0-N ids to doc strings if needed,
        # but indexing guaranteed keys in chunk_map are '0', '1', '2' ... corresponding to BM25 inserts.
        self.id_order = [str(i) for i in range(len(chunk_map))]

    async def retrieve(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """
        Returns top-K chunks by BM25 sparse similarity.
        """
        try:
            started_at = time.perf_counter()
            # 1. Normalize query
            normalize_started_at = time.perf_counter()
            query_terms = tokenize_for_bm25(query)
            normalize_latency_ms = (time.perf_counter() - normalize_started_at) * 1000

            if not query_terms:
                return []

            # 2. Get scores
            scoring_started_at = time.perf_counter()
            scores = self.bm25.get_scores(query_terms)
            scoring_latency_ms = (time.perf_counter() - scoring_started_at) * 1000

            # 3. Sort ascending then slice top K
            # zip index and score, sort descending on score
            scored_docs = sorted(enumerate(scores), key=lambda x: -x[1])[:top_k]

            mapping_started_at = time.perf_counter()
            results = []
            for doc_idx, score in scored_docs:
                if score <= 0:
                    continue  # Stop if no matching terms at all
                    
                doc_id = self.id_order[doc_idx]
                
                if doc_id in self.chunk_map:
                    raw_chunk = self.chunk_map[doc_id]
                    chunk_obj = Chunk(**raw_chunk)
                    results.append(
                        RetrievedChunk(
                            chunk=chunk_obj,
                            score=float(score),
                            source="lexical"
                        )
                    )
                else:
                    logger.warning(f"BM25 returned index {doc_idx} mapping to missing id {doc_id}")

            mapping_latency_ms = (time.perf_counter() - mapping_started_at) * 1000
            total_latency_ms = (time.perf_counter() - started_at) * 1000

            if self.debug_enabled:
                retrieval_logger.info(
                    "retriever=lexical query=%r top_k=%s hits=%s latency_ms=%.2f normalize_ms=%.2f score_ms=%.2f map_ms=%.2f",
                    query,
                    top_k,
                    len(results),
                    total_latency_ms,
                    normalize_latency_ms,
                    scoring_latency_ms,
                    mapping_latency_ms,
                )

            if self.debug_enabled:
                retrieval_logger.debug(
                    "retriever=lexical query=%r top_k=%s preview=%s",
                    query,
                    top_k,
                    build_retrieved_chunk_preview(results, self.preview_k),
                )
            return results

        except Exception as e:
            logger.error(f"Error during Lexical Retrieval: {e}", exc_info=True)
            return []
