import logging
import json
from typing import Optional, Union
from pathlib import Path

from meno_core.config.settings import settings
from meno_core.core.gte_embedding import GTEEmbedding
from meno_core.core.rag.config import ChunkRagConfig
from meno_core.core.rag.orchestration.orchestrator import ChunkRagOrchestrator
from meno_core.core.rag.retrieval.zvec_retriever import ZvecDenseRetriever
from meno_core.core.rag.retrieval.bm25_retriever import BM25LexicalRetriever
from meno_core.core.rag.ingestion.indexer import Indexer
from meno_core.core.rag.ingestion.chunker import build_chunk

logger = logging.getLogger(__name__)


async def build_chunk_rag_orchestrator(
    working_dir: Union[str, Path],
    embedder: GTEEmbedding
) -> Optional[ChunkRagOrchestrator]:
    """
    Factory function to initialize the ChunkRAG pipeline.
    If the indexes do not exist, it will build them from settings.kv_store_text_chunks_path.
    """
    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    indexer = Indexer(working_dir=working_dir, embedder=embedder)
    
    try:
        # Check if index exists by trying to load it
        if not indexer.chunks_meta_path.exists() or not indexer.bm25_path.exists():
            logger.info("Chunk RAG indices not found. Initializing vector DB from kv_store_text_chunks...")
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
    source_path = settings.kv_store_text_chunks_path
    if not source_path.exists():
        logger.error(f"❌ Source chunks file not found at {source_path}. Cannot initialize Chunk RAG.")
        return
        
    logger.info(f"Reading chunks from: {source_path}")
    try:
        with source_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            
        chunks = []
        total_chunks = len(data)
        logger.info(f"Loaded {total_chunks} chunks from JSON. Starting preprocessing (stemming, enrichment)...")
        
        for idx, (chunk_key, chunk_data) in enumerate(data.items()):
            if idx > 0 and idx % 200 == 0:
                logger.info(f"Preprocessed {idx}/{total_chunks} chunks...")
                
            text = chunk_data.get("content", "")
            file_path = chunk_data.get("file_path", "")
            doc_id = chunk_data.get("full_doc_id", "unknown_doc")
            chunk_idx = chunk_data.get("chunk_order_index", 0)
            
            enriched_text = text
            if file_path:
                enriched_text += f"\n\nИсточник: {file_path}"
                
            chunk_obj = await build_chunk(
                chunk_index=chunk_idx,
                text=enriched_text,
                document_id=doc_id,
                document_title=doc_id, # Simplified, using doc_id as title
                source_url=file_path
            )
            chunks.append(chunk_obj)
            
        logger.info("Preprocessing complete.")
        
        # Pass to indexer which handles batch embedding, BM25, and metadata writes
        logger.info(f"Delegating to Indexer to build zvec/bm25 for {len(chunks)} chunks in {working_dir}...")
        await indexer.build_index(chunks=chunks, batch_size=32)
        logger.info("✅ Initialization completed successfully.")
        
    except Exception as e:
        logger.error(f"❌ Failed to parse or embed source chunks: {e}", exc_info=True)
        raise
