import logging
from typing import List, Dict

import zvec

from meno_core.core.gte_embedding import GTEEmbedding
from meno_core.core.rag.models import Chunk, RetrievedChunk
from meno_core.core.rag.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)


class ZvecDenseRetriever(BaseRetriever):
    """
    Retrieves dense vector embeddings using the Zvec engine.
    """

    def __init__(
        self,
        embedder: GTEEmbedding,
        collection: zvec.Collection,
        chunk_map: Dict[str, dict]
    ):
        self.embedder = embedder
        self.collection = collection
        self.chunk_map = chunk_map

    async def retrieve(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """
        Returns top-K retrieved chunks by dense vector similarity.
        """
        try:
            # 1. Embed query
            embed_res = self.embedder.encode([query], return_dense=True, return_sparse=False)
            q_vec = embed_res["dense_embeddings"][0].detach().cpu().numpy().tolist()

            # 2. Query zvec
            zvec_results = self.collection.query(
                zvec.VectorQuery("embedding", vector=q_vec),
                topk=top_k
            )

            # 3. Map back to RetrievedChunk
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
                            source="dense"
                        )
                    )
                else:
                    logger.warning(f"Zvec returned ID {doc_id} but it's not in chunk map.")

            return results

        except Exception as e:
            logger.error(f"Error during Dense Retrieval: {e}", exc_info=True)
            return []
