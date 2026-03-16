import logging
from typing import Optional, Union
from pathlib import Path

from meno_core.core.gte_embedding import GTEEmbedding
from meno_core.core.rag.config import ChunkRagConfig
from meno_core.core.rag.orchestration.orchestrator import ChunkRagOrchestrator
from meno_core.core.rag.retrieval.zvec_retriever import ZvecDenseRetriever
from meno_core.core.rag.retrieval.bm25_retriever import BM25LexicalRetriever
from meno_core.core.rag.ingestion.indexer import Indexer

logger = logging.getLogger(__name__)


def build_chunk_rag_orchestrator(
    working_dir: Union[str, Path],
    embedder: GTEEmbedding
) -> Optional[ChunkRagOrchestrator]:
    """
    Factory function to initialize the ChunkRAG pipeline from persisted indexes.
    Returns None if the indexes do not exist yet.
    """
    try:
        indexer = Indexer(working_dir=working_dir, embedder=embedder)
        zvec_collection, bm25, chunk_map = indexer.load_indexes()
        
        dense_retriever = ZvecDenseRetriever(
            embedder=embedder,
            collection=zvec_collection,
            chunk_map=chunk_map
        )
        
        lexical_retriever = BM25LexicalRetriever(
            bm25=bm25,
            chunk_map=chunk_map
        )
        
        config = ChunkRagConfig()
        
        orchestrator = ChunkRagOrchestrator(
            config=config,
            dense_retriever=dense_retriever,
            lexical_retriever=lexical_retriever
        )
        logger.info("ChunkRAG Orchestrator successfully initialized.")
        return orchestrator
        
    except FileNotFoundError as e:
        logger.warning(f"Could not load ChunkRAG indexes: {e}. Orchestrator unavailable.")
        return None
    except Exception as e:
        logger.error(f"Error during ChunkRAG initialization: {e}", exc_info=True)
        return None
