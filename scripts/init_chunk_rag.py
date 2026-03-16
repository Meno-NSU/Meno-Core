import asyncio
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from meno_core.config.settings import settings
from meno_core.core.rag.ingestion.indexer import Indexer
from meno_core.core.rag.ingestion.source_loader import load_chunks_from_compiled_corpus, resolve_chunk_corpus_path
from meno_core.core.rag_engine import initialize_rag

logger = logging.getLogger(__name__)


async def init_chunks_from_store():
    """
    Reads the compiled Chunk-RAG corpus JSONL and builds zvec and BM25 representations.
    """
    logger.info("Initializing Chunk RAG indexes from configured corpus source...")

    source_path = resolve_chunk_corpus_path(settings.chunk_rag_corpus_path)
    if source_path is None:
        logger.error(
            "Cannot find corpus file at %s",
            settings.chunk_rag_corpus_path,
        )
        return

    logger.info("Using compiled corpus source at %s", source_path)

    # Note: We need the GTE embedding instance here
    # In api.main it calls initialize_rag() which loads the embedder
    _, embedder, _, _ = await initialize_rag()

    chunks = await load_chunks_from_compiled_corpus(source_path)
    logger.info(f"Loaded {len(chunks)} chunks from source.")
    
    output_dir = settings.chunk_rag_data_path
    output_dir.mkdir(parents=True, exist_ok=True)
    
    indexer = Indexer(working_dir=output_dir, embedder=embedder)
    
    # This will write zvec vector index, bm25.pkl and chunk metadata json
    await indexer.build_index(chunks=chunks, batch_size=32)
    logger.info(f"Successfully generated ZVec and BM25 indices at {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(init_chunks_from_store())
