import logging
from dataclasses import dataclass
from typing import Dict, List

from meno_core.core.rag.debug_utils import build_retrieved_chunk_preview
from meno_core.core.rag.models import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FusionResult:
    chunks: List[RetrievedChunk]
    fused_preview: List[dict]


class HybridFusion:
    """
    Merges results from multiple retrievers using Weighted Sum with min-max normalisation.
    """

    def __init__(self, weights: Dict[str, float], preview_k: int = 5):
        self.weights = weights
        self.preview_k = preview_k

    def _normalize_scores(self, chunks: List[RetrievedChunk]) -> Dict[str, float]:
        """
        Applies min-max normalization to scores.
        """
        if not chunks:
            return {}
            
        scores = [c.score for c in chunks]
        min_score = min(scores)
        max_score = max(scores)
        
        normalized = {}
        for c in chunks:
            # Avoid division by zero if all scores are exactly the same
            if max_score > min_score:
                norm_v = (c.score - min_score) / (max_score - min_score)
            else:
                norm_v = 1.0  # Or 0.5, but 1.0 keeps it in play
            normalized[c.chunk.chunk_id] = norm_v
            
        return normalized

    def fuse(
        self,
        result_sets: Dict[str, List[RetrievedChunk]],
        top_k: int
    ) -> FusionResult:
        """
        Perform min-max normalization independently on each source, then sum the weighted values.
        """
        normalized_scores = {
            source_name: self._normalize_scores(chunks)
            for source_name, chunks in result_sets.items()
        }

        # Build union map chunk_id -> Chunk
        # Store max of (raw chunks) to preserve metadata
        chunk_map = {}
        for chunks in result_sets.values():
            for chunk_wrapper in chunks:
                chunk_map[chunk_wrapper.chunk.chunk_id] = chunk_wrapper.chunk

        # Calculate fused scores
        fused_scores = {}
        for cid in chunk_map.keys():
            fused_scores[cid] = sum(
                normalized_scores.get(source_name, {}).get(cid, 0.0) * self.weights.get(source_name, 0.0)
                for source_name in result_sets.keys()
            )

        # Sort by best combined score
        sorted_pairs = sorted(fused_scores.items(), key=lambda x: -x[1])[:top_k]

        # Convert back to RetrievedChunk models
        final_results = []
        for cid, score in sorted_pairs:
            final_results.append(
                RetrievedChunk(
                    chunk=chunk_map[cid],
                    score=score,
                    source="hybrid"
                )
            )

        return FusionResult(
            chunks=final_results,
            fused_preview=build_retrieved_chunk_preview(final_results, self.preview_k),
        )
