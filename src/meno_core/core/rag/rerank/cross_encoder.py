import logging
import math
from typing import List

import torch

from meno_core.core.rag.models import RetrievedChunk
from meno_core.core.rag_engine import _reranker_tokenizer, _reranker_model

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks a list of chunks based on cross-encoder similarity with the query.
    """

    def __init__(self, filter_threshold: float = -10.0):
        self.tokenizer = _reranker_tokenizer
        self.model = _reranker_model
        self.filter_threshold = filter_threshold

    async def rerank(self, query: str, chunks: List[RetrievedChunk], top_n: int) -> List[RetrievedChunk]:
        """
        Cross-encode the query against chunk texts and re-order them.
        """
        if not chunks:
            return []

        if self.model is None or self.tokenizer is None:
            logger.warning("Reranker model/tokenizer not initialized. Skipping rerank.")
            return chunks[:top_n]

        device = next(self.model.parameters()).device
        docs = [c.chunk.text for c in chunks]
        
        scores = []
        minibatch_size = 4
        num_batches = math.ceil(len(docs) / minibatch_size)

        for batch_idx in range(num_batches):
            batch_start = batch_idx * minibatch_size
            batch_end = min(len(docs), batch_start + minibatch_size)
            pairs = [[query, d] for d in docs[batch_start:batch_end]]

            with torch.no_grad():
                inputs = self.tokenizer(
                    pairs, 
                    padding=True, 
                    truncation=True, 
                    return_tensors='pt',
                    max_length=512
                ).to(device)
                
                batch_scores = self.model(**inputs, return_dict=True).logits.view(-1,).float().cpu().numpy().tolist()
                scores.extend(batch_scores)

        # Update scores and sort
        reranked_chunks = []
        for chunk_wrapper, score in zip(chunks, scores):
            if score >= self.filter_threshold:
                chunk_wrapper.score = float(score)  # Update with cross-encoder score
                reranked_chunks.append(chunk_wrapper)

        reranked_chunks.sort(key=lambda x: -x.score)
        
        return reranked_chunks[:top_n]
