import logging
from typing import Optional, Union
from pathlib import Path

from meno_core.config.settings import settings
from meno_core.core.gte_embedding import GTEEmbedding
from meno_core.core.rag.config import ChunkRagConfig
from meno_core.core.rag.orchestration.orchestrator import ChunkRagOrchestrator
from meno_core.core.rag.retrieval.zvec_retriever import ZvecDenseRetriever
from meno_core.core.rag.retrieval.bm25_retriever import BM25LexicalRetriever
from meno_core.core.rag.ingestion.indexer import Indexer
from meno_core.core.rag.ingestion.source_loader import load_chunks_from_compiled_corpus, resolve_chunk_corpus_path

logger = logging.getLogger(__name__)


async def build_chunk_rag_orchestrator(
    working_dir: Union[str, Path],
    embedder: GTEEmbedding
) -> Optional[ChunkRagOrchestrator]:
    """
    Factory function to initialize the ChunkRAG pipeline.
    If the indexes do not exist, it will build them from the compiled corpus JSONL.
    """
    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    indexer = Indexer(working_dir=working_dir, embedder=embedder)
    
    try:
        # Check if index exists by trying to load it
        if not indexer.chunks_meta_path.exists() or not indexer.bm25_path.exists():
            logger.info("Chunk RAG indices not found. Initializing vector DB from source corpus...")
            await _run_initialization(indexer, working_dir)
        else:
            logger.info("Found existing Chunk RAG indices. Proceeding to load...")
            
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
        logger.info("✅ ChunkRAG Orchestrator successfully initialized.")
        return orchestrator
        
    except Exception as e:
        logger.error(f"❌ Error during ChunkRAG initialization: {e}", exc_info=True)
        return None


async def _run_initialization(indexer: Indexer, working_dir: Path):
    corpus_path = resolve_chunk_corpus_path(settings.chunk_rag_corpus_path)
    if corpus_path is None:
        logger.error(
            "❌ No Chunk RAG corpus file found at %s.",
            settings.chunk_rag_corpus_path,
        )
        return

    logger.info("Reading compiled corpus chunks from: %s", corpus_path)
    try:
        chunks = await load_chunks_from_compiled_corpus(corpus_path)
        logger.info("Loaded %s chunks from %s.", len(chunks), corpus_path)

        # Pass to indexer which handles batch embedding, BM25, and metadata writes
        logger.info(f"Delegating to Indexer to build zvec/bm25 for {len(chunks)} chunks in {working_dir}...")
        await indexer.build_index(chunks=chunks, batch_size=32)
        logger.info("✅ Initialization completed successfully.")
        
    except Exception as e:
        logger.error(f"❌ Failed to parse or embed source chunks: {e}", exc_info=True)
        raise
