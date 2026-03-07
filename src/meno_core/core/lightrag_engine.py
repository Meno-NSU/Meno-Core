import logging
from pathlib import Path
from typing import Any, Union
from lightrag import LightRAG
from rank_bm25 import BM25Okapi
from functools import partial
import torch

from meno_core.config.settings import settings
from meno_core.core.gte_embedding import GTEEmbedding

logger = logging.getLogger(__name__)

class LightRAGEngine:
    def __init__(
            self,
            rag_instance: LightRAG,
            embedder: GTEEmbedding,
            bm25: BM25Okapi, 
            chunk_db: list[tuple[Any, Any]],
    ):
        self.rag_instance = rag_instance
        self.embedder = embedder
        self.bm25 = bm25
        self.chunk_db = chunk_db
        
    async def aquery(self, *args, **kwargs):
        """Pass-through to LightRAG's aquery"""
        return await self.rag_instance.aquery(*args, **kwargs)
        
    async def aclear_cache(self):
        """Pass-through to LightRAG's aclear_cache"""
        if hasattr(self.rag_instance, 'aclear_cache'):
            return await self.rag_instance.aclear_cache()
