import asyncio
import json
import logging
from pathlib import Path

from meno_core.config.settings import settings
from meno_core.core.rag.ingestion.chunker import build_chunk
from meno_core.core.rag.ingestion.indexer import Indexer
from meno_core.core.rag_engine import initialize_rag

logger = logging.getLogger(__name__)


async def init_chunks_from_store():
    """
    Reads resources/lightrag_kg_v3/kv_store_text_chunks.json, 
    creates Chunk objects mapping textual data and URLs (`file_path`), 
    then uses Indexer to build zvec and BM25 representations.
    """
    logger.info("Initializing Chunk RAG indexes from kv_store_text_chunks.json...")
    
    # Use path from settings
    source_path = settings.kv_store_text_chunks_path
    if not source_path.exists():
        logger.error(f"Cannot find source file {source_path}")
        return

    with source_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Note: We need the GTE embedding instance here
    # In api.main it calls initialize_rag() which loads the embedder
    _, embedder, _, _ = await initialize_rag()

    chunks = []
    
    # We will just parse them directly, assume no full document context for LLM title extraction 
    # since documents are already chunked and we don't have document boundaries easily separated 
    # beyond `full_doc_id`.
    
    for chunk_key, chunk_data in data.items():
        text = chunk_data.get("content", "")
        file_path = chunk_data.get("file_path", "")
        doc_id = chunk_data.get("full_doc_id", "unknown_doc")
        chunk_idx = chunk_data.get("chunk_order_index", 0)
        
        # Inject file_path directly into text so LLM can easily see it
        # Actually build_chunk does text_for_dense enrichment, text should remain clean but we can append it if preferred.
        enriched_text = text
        if file_path:
            enriched_text += f"\n\nИсточник: {file_path}"
        
        # For simplicity since this is already chunked text, we pass document_title as doc_id
        chunk_obj = await build_chunk(
            chunk_index=chunk_idx,
            text=enriched_text,
            document_id=doc_id,
            document_title=doc_id,
            source_url=file_path
        )
        chunks.append(chunk_obj)
        
    logger.info(f"Loaded {len(chunks)} chunks from source.")
    
    # The output path will be under working_dir (same as light_rag) or resources
    output_dir = Path("resources") / "chunk_rag_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    indexer = Indexer(working_dir=output_dir, embedder=embedder)
    
    # This will write zvec vector index, bm25.pkl and chunk metadata json
    await indexer.build_index(chunks=chunks, batch_size=32)
    logger.info(f"Successfully generated ZVec and BM25 indices at {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(init_chunks_from_store())
