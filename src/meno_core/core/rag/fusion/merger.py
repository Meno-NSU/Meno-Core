import logging
from typing import List, Dict

from meno_core.core.rag.models import RetrievedChunk

logger = logging.getLogger(__name__)


class HybridFusion:
    """
    Merges results from multiple retrievers (e.g., dense and lexical) using Weighted Sum with Min-Max normalisation.
    """

    def __init__(self, dense_weight: float = 0.6, lexical_weight: float = 0.4):
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight

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
        dense_results: List[RetrievedChunk],
        lexical_results: List[RetrievedChunk],
        top_k: int
    ) -> List[RetrievedChunk]:
        """
        Perform min-max normalization independently on each source, then sum the weighted values.
        """
        norm_dense = self._normalize_scores(dense_results)
        norm_lexical = self._normalize_scores(lexical_results)

        # Build union map chunk_id -> Chunk
        # Store max of (raw chunks) to preserve metadata
        chunk_map = {}
        for c in dense_results + lexical_results:
            chunk_map[c.chunk.chunk_id] = c.chunk

        # Calculate fused scores
        fused_scores = {}
        for cid in chunk_map.keys():
            d_score = norm_dense.get(cid, 0.0) * self.dense_weight
            l_score = norm_lexical.get(cid, 0.0) * self.lexical_weight
            fused_scores[cid] = d_score + l_score

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

        return final_results
