import json
import logging
import os
import pickle
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import zvec
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from meno_core.core.rag.config import ChunkRagConfig
from meno_core.core.rag.embeddings import DenseEmbedder
from meno_core.core.rag.models import Chunk

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IndexInspectionResult:
    ready: bool
    reasons: list[str]


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
            "build_signature": {
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
            },
            "runtime_config": {
                "reranker": {
                    "model_path": self.reranker_path,
                    "backend": "qwen_yes_no",
                },
                "fusion_weights": {
                    "multilingual_dense": self.config.fusion_weight_multilingual,
                    "russian_dense": self.config.fusion_weight_russian,
                    "lexical": self.config.fusion_weight_bm25,
                },
            },
        }

    def needs_rebuild(self) -> bool:
        return not self.inspect_index_state().ready

    def inspect_index_state(self) -> IndexInspectionResult:
        reasons: list[str] = []

        if not self.chunks_meta_path.exists() or not self.bm25_path.exists() or not self.manifest_path.exists():
            if not self.chunks_meta_path.exists():
                reasons.append(f"missing metadata file: {self.chunks_meta_path}")
            if not self.bm25_path.exists():
                reasons.append(f"missing BM25 artifact: {self.bm25_path}")
            if not self.manifest_path.exists():
                reasons.append(f"missing manifest: {self.manifest_path}")

        if any(not path.exists() for path in self.zvec_paths.values()):
            for source_name, path in self.zvec_paths.items():
                if not path.exists():
                    reasons.append(f"missing dense index for {source_name}: {path}")

        if reasons:
            return IndexInspectionResult(ready=False, reasons=reasons)

        try:
            with self.manifest_path.open("r", encoding="utf-8") as fp:
                current_manifest = json.load(fp)
        except Exception as error:
            return IndexInspectionResult(
                ready=False,
                reasons=[f"cannot read manifest {self.manifest_path}: {error}"],
            )

        expected_manifest = self.expected_manifest()
        current_schema_version = current_manifest.get("schema_version")
        if current_schema_version != expected_manifest["schema_version"]:
            reasons.append(
                f"schema_version mismatch: current={current_schema_version}, expected={expected_manifest['schema_version']}"
            )

        current_build_signature = current_manifest.get("build_signature")
        expected_build_signature = expected_manifest["build_signature"]
        if current_build_signature != expected_build_signature:
            reasons.extend(self._describe_signature_mismatch(current_build_signature, expected_build_signature))

        return IndexInspectionResult(ready=not reasons, reasons=reasons)

    @staticmethod
    def _describe_signature_mismatch(current: Any, expected: Any, prefix: str = "build_signature") -> list[str]:
        if current == expected:
            return []

        differences: list[str] = []
        if isinstance(current, dict) and isinstance(expected, dict):
            keys = sorted(set(current.keys()) | set(expected.keys()))
            for key in keys:
                next_prefix = f"{prefix}.{key}"
                if key not in current:
                    differences.append(f"missing manifest key: {next_prefix}")
                    continue
                if key not in expected:
                    differences.append(f"unexpected manifest key: {next_prefix}")
                    continue
                differences.extend(Indexer._describe_signature_mismatch(current[key], expected[key], next_prefix))
                if len(differences) >= 10:
                    break
            return differences

        differences.append(f"mismatch at {prefix}: current={current!r}, expected={expected!r}")
        return differences

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes, remainder = divmod(int(seconds), 60)
        if minutes < 60:
            return f"{minutes}m {remainder}s"
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes}m {remainder}s"

    @staticmethod
    def _resolve_embedder_device(embedder: DenseEmbedder) -> str:
        model = getattr(embedder, "model", None)
        if model is not None:
            try:
                return str(next(model.parameters()).device)
            except Exception:
                device = getattr(model, "device", None)
                if device is not None:
                    return str(device)

        base_embedder = getattr(embedder, "base_embedder", None)
        if base_embedder is not None:
            model = getattr(base_embedder, "model", None)
            if model is not None:
                try:
                    return str(next(model.parameters()).device)
                except Exception:
                    device = getattr(model, "device", None)
                    if device is not None:
                        return str(device)

        return "unknown"

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
        total_chunks = len(chunks)
        total_batches = (total_chunks + batch_size - 1) // batch_size if total_chunks else 0
        progress_every = max(1, min(25, total_batches // 20 if total_batches >= 20 else 1))
        indexing_started_at = time.perf_counter()
        embedder_descriptions = ", ".join(
            f"{source_name}(device={self._resolve_embedder_device(embedder)}, dim={embedder.dimension}, model={embedder.model_path})"
            for source_name, embedder in self.dense_embedders.items()
        )

        logger.info(
            "Encoding and inserting into Zvec in batches of %s. Total batches=%s. Dense embedders: %s",
            batch_size,
            total_batches,
            embedder_descriptions,
        )
        for batch_idx, batch_start in enumerate(range(0, total_chunks, batch_size), start=1):
            batch_started_at = time.perf_counter()
            batch_chunks = chunks[batch_start:batch_start + batch_size]
            dense_texts = [chunk.text_for_dense for chunk in batch_chunks]
            doc_ids = [str(batch_start + offset) for offset in range(len(batch_chunks))]
            embedder_timings_ms: dict[str, float] = {}

            for source_name, embedder in self.dense_embedders.items():
                embedder_started_at = time.perf_counter()
                embeddings = embedder.encode_documents(dense_texts).detach().cpu().numpy().tolist()
                docs = [
                    zvec.Doc(id=doc_id, vectors={"embedding": embeddings[offset]})
                    for offset, doc_id in enumerate(doc_ids)
                ]
                collections[source_name].insert(docs)
                embedder_timings_ms[source_name] = round((time.perf_counter() - embedder_started_at) * 1000, 2)

            for offset, chunk in enumerate(batch_chunks):
                chunk_metadata_dict[doc_ids[offset]] = chunk.model_dump()

            processed_chunks = min(batch_start + len(batch_chunks), total_chunks)
            elapsed_seconds = time.perf_counter() - indexing_started_at
            chunks_per_second = processed_chunks / elapsed_seconds if elapsed_seconds > 0 else 0.0
            remaining_chunks = max(total_chunks - processed_chunks, 0)
            eta_seconds = remaining_chunks / chunks_per_second if chunks_per_second > 0 else 0.0
            batch_elapsed_ms = round((time.perf_counter() - batch_started_at) * 1000, 2)

            if batch_idx == 1 or batch_idx == total_batches or batch_idx % progress_every == 0:
                logger.info(
                    "Chunk-RAG indexing progress: batch %s/%s, chunks %s/%s, %.2f chunks/s, ETA %s, last_batch=%sms, embedder_ms=%s",
                    batch_idx,
                    total_batches,
                    processed_chunks,
                    total_chunks,
                    chunks_per_second,
                    self._format_duration(eta_seconds),
                    batch_elapsed_ms,
                    embedder_timings_ms,
                )

            logger.debug(
                "Inserted batch %s/%s into dense indexes",
                batch_idx,
                total_batches,
            )

        with self.chunks_meta_path.open("w", encoding="utf-8") as fp:
            json.dump(chunk_metadata_dict, fp, ensure_ascii=False, indent=2)

        with self.manifest_path.open("w", encoding="utf-8") as fp:
            json.dump(self.expected_manifest(), fp, ensure_ascii=False, indent=2)

        logger.info(
            "Indexing complete in %s. Metadata saved to %s, manifest saved to %s.",
            self._format_duration(time.perf_counter() - indexing_started_at),
            self.chunks_meta_path,
            self.manifest_path,
        )

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
