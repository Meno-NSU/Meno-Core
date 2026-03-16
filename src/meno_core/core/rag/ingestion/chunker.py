import logging
from typing import List, Optional

from meno_core.core.rag.models import Chunk, ChunkMetadata
from meno_core.core.rag_engine import tokenize_and_normalize, generate_with_llm
from meno_core.core.rag.prompts import TITLE_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


async def extract_document_title(chunks_text: List[str]) -> str:
    """
    Given the first few chunks of a document, uses the LLM to extract/generate a global title.
    """
    if not chunks_text:
        return "Unknown Document"
    
    # Take up to first 4 chunks
    sample_text = "\n\n---\n\n".join(chunks_text[:4])
    prompt = TITLE_EXTRACTION_PROMPT.format(chunks_text=sample_text)
    
    try:
        title = await generate_with_llm(prompt=prompt)
        return title.strip().strip('"').strip("'")
    except Exception as e:
        logger.error(f"Failed to extract document title: {e}", exc_info=True)
        return "Unknown Document"


async def build_chunk(
    chunk_index: int,
    text: str,
    document_id: str,
    document_title: str,
    section_title: Optional[str] = None,
    source_url: Optional[str] = None,
    page_range: Optional[str] = None,
    extra_metadata: Optional[dict] = None
) -> Chunk:
    """
    Constructs a Chunk object, generating the dense and bm25 representations.
    """
    chunk_id = f"{document_id}_chunk_{chunk_index}"
    
    # Enhance text for dense embedding with titles to improve retrieval context
    text_for_dense = f"Document: {document_title}\n"
    if section_title:
        text_for_dense += f"Section: {section_title}\n"
    text_for_dense += f"\n{text}"
    
    # Stem normalization for BM25
    text_for_bm25 = await tokenize_and_normalize(text_for_dense)
    
    metadata = ChunkMetadata(
        document_id=document_id,
        document_title=document_title,
        section_title=section_title,
        source_url=source_url,
        page_range=page_range,
        chunk_index=chunk_index,
        extra=extra_metadata or {}
    )
    
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        text_for_dense=text_for_dense,
        text_for_bm25=text_for_bm25,
        metadata=metadata
    )
