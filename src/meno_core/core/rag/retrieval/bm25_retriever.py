import logging
import time
from typing import List, Dict, Sequence

import numpy as np

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
        chunk_map: Dict[str, Chunk],
        *,
        debug_enabled: bool = False,
        preview_k: int = 5,
    ):
        self.bm25 = bm25
        self.chunk_map = chunk_map
        self.debug_enabled = debug_enabled
        self.preview_k = preview_k
        self.id_order = self._build_id_order(chunk_map)

    def _build_id_order(self, chunk_map: Dict[str, Chunk]) -> List[str]:
        try:
            ordered = sorted(((int(doc_id), doc_id) for doc_id in chunk_map.keys()), key=lambda item: item[0])
        except ValueError as error:
            raise ValueError("BM25 chunk ids must be numeric strings") from error

        expected = list(range(len(ordered)))
        actual = [item[0] for item in ordered]
        if actual != expected:
            raise ValueError(
                f"BM25 chunk ids must be consecutive numeric strings starting at 0, got {actual[:10]}"
            )

        corpus_size = getattr(self.bm25, "corpus_size", None)
        if corpus_size is None:
            corpus_size = getattr(self.bm25, "doc_len", None)
            corpus_size = len(corpus_size) if corpus_size is not None else None
        if corpus_size is None:
            corpus_size = len(getattr(self.bm25, "corpus", [])) or None
        if corpus_size is not None and corpus_size != len(ordered):
            raise ValueError(
                f"BM25 corpus size mismatch: corpus_size={corpus_size}, chunk_map_size={len(ordered)}"
            )

        return [doc_id for _, doc_id in ordered]

    @staticmethod
    def _top_positive_indices(scores: np.ndarray, top_k: int) -> list[int]:
        positive_indices = np.flatnonzero(scores > 0)
        if positive_indices.size == 0:
            return []

        if positive_indices.size <= top_k:
            candidates = positive_indices.tolist()
        else:
            candidate_positions = np.argpartition(scores[positive_indices], -top_k)[-top_k:]
            candidates = positive_indices[candidate_positions].tolist()

        return sorted(candidates, key=lambda idx: (-float(scores[idx]), idx))

    async def retrieve_many(self, queries: Sequence[str], top_k: int) -> List[List[RetrievedChunk]]:
        """
        Returns top-K chunks by BM25 sparse similarity for each query.
        """
        try:
            started_at = time.perf_counter()
            normalize_started_at = time.perf_counter()
            normalized_queries = [tokenize_for_bm25(query) for query in queries]
            normalize_latency_ms = (time.perf_counter() - normalize_started_at) * 1000

            scoring_started_at = time.perf_counter()
            score_sets = [
                np.asarray(self.bm25.get_scores(query_terms), dtype=float)
                if query_terms else np.asarray([], dtype=float)
                for query_terms in normalized_queries
            ]
            scoring_latency_ms = (time.perf_counter() - scoring_started_at) * 1000

            mapping_started_at = time.perf_counter()
            results_by_query: list[list[RetrievedChunk]] = []
            for scores in score_sets:
                if scores.size == 0:
                    results_by_query.append([])
                    continue

                query_results: list[RetrievedChunk] = []
                for doc_idx in self._top_positive_indices(scores, top_k):
                    doc_id = self.id_order[doc_idx]
                    chunk_obj = self.chunk_map.get(doc_id)
                    if chunk_obj is None:
                        logger.warning("BM25 returned index %s mapping to missing id %s", doc_idx, doc_id)
                        continue
                    query_results.append(
                        RetrievedChunk(
                            chunk=chunk_obj,
                            score=float(scores[doc_idx]),
                            source="lexical"
                        )
                    )
                results_by_query.append(query_results)

            mapping_latency_ms = (time.perf_counter() - mapping_started_at) * 1000
            total_latency_ms = (time.perf_counter() - started_at) * 1000

            if self.debug_enabled:
                retrieval_logger.info(
                    "retriever=lexical queries=%s top_k=%s hits=%s latency_ms=%.2f normalize_ms=%.2f score_ms=%.2f map_ms=%.2f",
                    len(queries),
                    top_k,
                    [len(results) for results in results_by_query],
                    total_latency_ms,
                    normalize_latency_ms,
                    scoring_latency_ms,
                    mapping_latency_ms,
                )

            if self.debug_enabled and queries:
                retrieval_logger.debug(
                    "retriever=lexical first_query=%r top_k=%s preview=%s",
                    queries[0],
                    top_k,
                    build_retrieved_chunk_preview(results_by_query[0], self.preview_k) if results_by_query else [],
                )
            return results_by_query

        except Exception as e:
            logger.error("Lexical retrieval FAILED, returning empty results for %s queries: %s", len(queries), e, exc_info=True)
            return [[] for _ in queries]
