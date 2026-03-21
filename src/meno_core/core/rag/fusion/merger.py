import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

from meno_core.core.rag.debug_utils import build_retrieved_chunk_preview
from meno_core.core.rag.models import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FusionResult:
    chunks: List[RetrievedChunk]
    fused_preview: List[dict]


class HybridFusion:
    """
    Merges results from multiple retrievers using weighted Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, weights: Dict[str, float], preview_k: int = 5, rank_constant: int = 60):
        self.weights = weights
        self.preview_k = preview_k
        self.rank_constant = rank_constant

    @staticmethod
    def _as_ranked_lists(
        ranked_results: Sequence[RetrievedChunk] | Sequence[Sequence[RetrievedChunk]],
    ) -> Iterable[Sequence[RetrievedChunk]]:
        if not ranked_results:
            return []
        first_item = ranked_results[0]
        if isinstance(first_item, RetrievedChunk):
            return [ranked_results]  # type: ignore[list-item]
        return ranked_results  # type: ignore[return-value]

    def fuse(
        self,
        result_sets: Mapping[str, Sequence[RetrievedChunk] | Sequence[Sequence[RetrievedChunk]]],
        top_k: int
    ) -> FusionResult:
        """
        Perform weighted Reciprocal Rank Fusion across retrievers and query variants.
        """
        chunk_map: dict[str, RetrievedChunk] = {}
        fused_scores: dict[str, float] = {}

        for source_name, ranked_groups in result_sets.items():
            source_weight = self.weights.get(source_name, 0.0)
            if source_weight <= 0:
                continue
            for ranked_chunks in self._as_ranked_lists(ranked_groups):
                for rank, chunk_wrapper in enumerate(ranked_chunks, start=1):
                    chunk_id = chunk_wrapper.chunk.chunk_id
                    chunk_map[chunk_id] = chunk_wrapper
                    fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (
                        source_weight / (self.rank_constant + rank)
                    )

        # Sort by best combined score
        sorted_pairs = sorted(fused_scores.items(), key=lambda x: -x[1])[:top_k]

        # Convert back to RetrievedChunk models
        final_results = []
        for cid, score in sorted_pairs:
            chunk_wrapper = chunk_map[cid]
            final_results.append(
                RetrievedChunk(
                    chunk=chunk_wrapper.chunk,
                    score=score,
                    source="hybrid"
                )
            )

        return FusionResult(
            chunks=final_results,
            fused_preview=build_retrieved_chunk_preview(final_results, self.preview_k),
        )
