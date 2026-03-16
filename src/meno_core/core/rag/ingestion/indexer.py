import json
import logging
import os
import pickle
from pathlib import Path
from typing import List, Union, Tuple, Any

import zvec
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from meno_core.config.settings import settings
from meno_core.core.gte_embedding import GTEEmbedding
from meno_core.core.rag.models import Chunk

logger = logging.getLogger(__name__)


class Indexer:
    """
    Handles offline indexing of Chunks into a Zvec graph for dense retrieval
    and a serialized BM25Okapi object for lexical retrieval.
    """

    def __init__(
            self,
            working_dir: Union[str, Path],
            embedder: GTEEmbedding,
            zvec_collection_name: str = "meno_zvec_chunk_rag"
    ):
        self.working_dir = Path(working_dir)
        self.embedder = embedder
        self.zvec_collection_name = zvec_collection_name

        self.zvec_path = self.working_dir / "zvec_chunk_idx"
        self.bm25_path = self.working_dir / "bm25_stemmed_kb.pkl"
        self.chunks_meta_path = self.working_dir / "chunk_rag_metadata.json"

    async def build_index(self, chunks: List[Chunk], batch_size: int = 32):
        """
        Builds and saves the Zvec index, the BM25 index, and metadata store.
        """
        logger.info(f"Starting indexing for {len(chunks)} chunks...")

        import shutil
        import os

        # 1. Ensure working directory exists, and clean up existing zvec path if it exists
        # Zvec throws "path validate failed" if we use create_and_open on an existing path
        os.makedirs(self.working_dir, exist_ok=True)
        if self.zvec_path.exists():
            logger.info(f"Removing existing zvec index at {self.zvec_path} before re-building...")
            shutil.rmtree(self.zvec_path, ignore_errors=True)

        # 2. Extract texts for BM25 and build it
        logger.info("Building BM25 index...")
        bm25_corpus = [chunk.text_for_bm25.split() for chunk in chunks]
        bm25 = BM25Okapi(bm25_corpus)
        with open(self.bm25_path, "wb") as f:
            pickle.dump(bm25, f)
        logger.info(f"BM25 index saved to {self.bm25_path}")

        # 3. Create Zvec Collection schema
        schema = zvec.CollectionSchema(
            name=self.zvec_collection_name,
            vectors=zvec.VectorSchema(
                "embedding",
                zvec.DataType.VECTOR_FP32,
                settings.embedder_dim
            ),
        )

        # Open or overwrite zvec collection
        # zvec.create_and_open typically creates or loads.
        # If we are re-indexing everything, we might need to delete old first, but let's assume zvec creates new or appends.
        # For simplicity, we create and open.
        collection = zvec.create_and_open(path=str(self.zvec_path), schema=schema)

        # 4. Save metadata dictionary (to map chunk array index -> Chunk object)
        chunk_metadata_dict = {}

        # 5. Calculate embeddings and insert into Zvec in batches
        logger.info(f"Encoding and inserting into Zvec in batches of {batch_size}...")
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            dense_texts = [c.text_for_dense for c in batch_chunks]

            embeddings_res = self.embedder.encode(dense_texts, return_dense=True, return_sparse=False)
            embeddings = embeddings_res["dense_embeddings"].detach().cpu().numpy().tolist()

            docs = []
            for j, chunk in enumerate(batch_chunks):
                internal_id = i + j
                chunk_metadata_dict[str(internal_id)] = chunk.model_dump()
                docs.append(zvec.Doc(id=str(internal_id), vectors={"embedding": embeddings[j]}))

            collection.insert(docs)
            logger.debug(f"Inserted batch {i // batch_size} of {len(chunks) // batch_size}")

        # Write metadata to disk
        with open(self.chunks_meta_path, "w", encoding="utf-8") as f:
            json.dump(chunk_metadata_dict, f, ensure_ascii=False, indent=2)

        logger.info("Indexing complete.")

    def load_indexes(self) -> Tuple[Any, BM25Okapi, dict]:
        """
        Loads the Zvec collection, the BM25 object, and the metadata dictionary at runtime.
        Returns:
            zvec_collection, bm25, metadata_dict
        """
        if not self.chunks_meta_path.exists():
            raise FileNotFoundError(f"Chunk metadata not found at {self.chunks_meta_path}")

        with open(self.chunks_meta_path, "r", encoding="utf-8") as f:
            chunk_metadata_dict = json.load(f)

        with open(self.bm25_path, "rb") as f:
            bm25 = pickle.load(f)

        zvec_collection = zvec.open(path=str(self.zvec_path))

        return zvec_collection, bm25, chunk_metadata_dict
