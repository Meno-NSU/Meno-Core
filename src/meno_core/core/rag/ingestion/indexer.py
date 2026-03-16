import json
import logging
import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import zvec
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from meno_core.core.rag.config import ChunkRagConfig
from meno_core.core.rag.embeddings import DenseEmbedder
from meno_core.core.rag.models import Chunk

logger = logging.getLogger(__name__)


class Indexer:
    """
    Handles offline indexing of Chunks into multiple Zvec collections for dense retrieval
    and a serialized BM25Okapi object for lexical retrieval.
    """

    def __init__(
        self,
        working_dir: Union[str, Path],
        dense_embedders: Dict[str, DenseEmbedder],
        reranker_path: str,
        config: ChunkRagConfig,
    ):
        self.working_dir = Path(working_dir)
        self.dense_embedders = dense_embedders
        self.reranker_path = reranker_path
        self.config = config

        self.zvec_paths = {
            "multilingual_dense": self.working_dir / "zvec_multilingual_idx",
            "russian_dense": self.working_dir / "zvec_russian_idx",
        }
        self.bm25_path = self.working_dir / "bm25_stemmed_kb.pkl"
        self.chunks_meta_path = self.working_dir / "chunk_rag_metadata.json"
        self.manifest_path = self.working_dir / "chunk_rag_manifest.json"

    def expected_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "dense_indexes": {
                source_name: {
                    "path": str(self.zvec_paths[source_name].name),
                    "model_path": embedder.model_path,
                    "dimension": embedder.dimension,
                    "pooling_strategy": embedder.pooling_strategy,
                    "query_prefix": embedder.query_prefix,
                    "document_prefix": embedder.document_prefix,
                }
                for source_name, embedder in self.dense_embedders.items()
            },
            "bm25": {
                "normalizer_version": 2,
            },
            "reranker": {
                "model_path": self.reranker_path,
                "backend": "qwen_yes_no",
            },
            "fusion_weights": {
                "multilingual_dense": self.config.fusion_weight_multilingual,
                "russian_dense": self.config.fusion_weight_russian,
                "lexical": self.config.fusion_weight_bm25,
            },
        }

    def needs_rebuild(self) -> bool:
        if not self.chunks_meta_path.exists() or not self.bm25_path.exists() or not self.manifest_path.exists():
            return True

        if any(not path.exists() for path in self.zvec_paths.values()):
            return True

        try:
            with self.manifest_path.open("r", encoding="utf-8") as fp:
                current_manifest = json.load(fp)
        except Exception:
            return True

        return current_manifest != self.expected_manifest()

    async def build_index(self, chunks: List[Chunk], batch_size: int = 32):
        """
        Builds and saves the multi-vector Zvec indexes, the BM25 index, metadata store,
        and manifest that describes the retrieval layout.
        """
        logger.info("Starting indexing for %s chunks...", len(chunks))

        os.makedirs(self.working_dir, exist_ok=True)
        for path in self.zvec_paths.values():
            if path.exists():
                logger.info("Removing existing zvec index at %s before re-building...", path)
                shutil.rmtree(path, ignore_errors=True)

        logger.info("Building BM25 index...")
        bm25_corpus = [chunk.text_for_bm25.split() for chunk in chunks]
        bm25 = BM25Okapi(bm25_corpus)
        with self.bm25_path.open("wb") as fp:
            pickle.dump(bm25, fp)
        logger.info("BM25 index saved to %s", self.bm25_path)

        collections = {}
        for source_name, embedder in self.dense_embedders.items():
            schema = zvec.CollectionSchema(
                name=f"meno_{source_name}",
                vectors=zvec.VectorSchema(
                    "embedding",
                    zvec.DataType.VECTOR_FP32,
                    embedder.dimension,
                ),
            )
            collections[source_name] = zvec.create_and_open(path=str(self.zvec_paths[source_name]), schema=schema)

        chunk_metadata_dict: dict[str, Any] = {}

        logger.info("Encoding and inserting into Zvec in batches of %s...", batch_size)
        for batch_start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[batch_start:batch_start + batch_size]
            dense_texts = [chunk.text_for_dense for chunk in batch_chunks]
            doc_ids = [str(batch_start + offset) for offset in range(len(batch_chunks))]

            for source_name, embedder in self.dense_embedders.items():
                embeddings = embedder.encode_documents(dense_texts).detach().cpu().numpy().tolist()
                docs = [
                    zvec.Doc(id=doc_id, vectors={"embedding": embeddings[offset]})
                    for offset, doc_id in enumerate(doc_ids)
                ]
                collections[source_name].insert(docs)

            for offset, chunk in enumerate(batch_chunks):
                chunk_metadata_dict[doc_ids[offset]] = chunk.model_dump()

            logger.debug(
                "Inserted batch %s/%s into dense indexes",
                batch_start // batch_size + 1,
                (len(chunks) + batch_size - 1) // batch_size,
            )

        with self.chunks_meta_path.open("w", encoding="utf-8") as fp:
            json.dump(chunk_metadata_dict, fp, ensure_ascii=False, indent=2)

        with self.manifest_path.open("w", encoding="utf-8") as fp:
            json.dump(self.expected_manifest(), fp, ensure_ascii=False, indent=2)

        logger.info("Indexing complete.")

    def load_indexes(self) -> Tuple[Dict[str, Any], BM25Okapi, dict, dict]:
        """
        Loads the dense collections, BM25 object, metadata dictionary, and manifest.
        """
        if self.needs_rebuild():
            raise FileNotFoundError("Chunk RAG indexes are missing or incompatible with current settings")

        with self.chunks_meta_path.open("r", encoding="utf-8") as fp:
            chunk_metadata_dict = json.load(fp)

        with self.bm25_path.open("rb") as fp:
            bm25 = pickle.load(fp)

        with self.manifest_path.open("r", encoding="utf-8") as fp:
            manifest = json.load(fp)

        collections = {
            source_name: zvec.open(path=str(path))
            for source_name, path in self.zvec_paths.items()
        }

        return collections, bm25, chunk_metadata_dict, manifest
