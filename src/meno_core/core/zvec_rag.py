import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Dict, Union

import numpy as np
import zvec
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from meno_core.config.settings import settings
from meno_core.core.gte_embedding import GTEEmbedding
from meno_core.core.lexical_normalizer import tokenize_for_bm25
from meno_core.core.prompts import (
    SEARCH_SYSTEM_PROMPT,
    MEANINGLESS_REQUEST_ANSWER,
    EXAMPLES_OF_SEARCH_QUERIES,
)
from meno_core.core.rag.config import ChunkRagConfig
from meno_core.core.rag.fusion.merger import HybridFusion
from meno_core.core.rag.generation.context_assembler import ContextAssembler
from meno_core.core.rag.models import Chunk, ChunkMetadata, RetrievedChunk
from meno_core.core.rag.rerank.qwen_reranker import QwenCausalReranker
from meno_core.core.rag_engine import (
    generate_with_llm,
)
from meno_core.core.rag.rerank.qwen_reranker import load_cached_qwen_reranker_backend

logger = logging.getLogger(__name__)

LEXICAL_DISTANCE_THRESHOLD: float = 0.05
MAX_NUMBER_OF_ANSWER_VARIANTS: int = 3


class ZvecRAGEngine:
    def __init__(
            self,
            working_dir: Union[str, Path],
            embedder: GTEEmbedding,
            bm25: BM25Okapi,
            chunk_db: list[tuple[Any, Any]],
    ):
        self.working_dir = Path(working_dir)
        self.embedder = embedder
        self.bm25 = bm25
        self.chunk_db = chunk_db
        self.zvec_path = self.working_dir / "zvec_index"
        self.reranker_backend = load_cached_qwen_reranker_backend(settings.reranker_path)
        self.config = ChunkRagConfig()
        self.fusion = HybridFusion(
            weights={
                "multilingual_dense": self.config.fusion_weight_multilingual,
                "lexical": self.config.fusion_weight_bm25,
            }
        )
        self.reranker = QwenCausalReranker(self.reranker_backend, filter_threshold=0.0)
        self.assembler = ContextAssembler(token_budget=self.config.token_budget)
        self.legacy_chunks = [
            self._build_legacy_chunk(idx, content, full_doc_id)
            for idx, (content, full_doc_id) in enumerate(chunk_db)
        ]

        # We will initialize zvec collection inline here or load it.
        self._init_zvec_collection()

    def _init_zvec_collection(self):
        schema = zvec.CollectionSchema(
            name="meno_zvec",
            vectors=zvec.VectorSchema(
                "embedding",
                zvec.DataType.VECTOR_FP32,
                self._resolve_embedder_dimension()
            ),
        )

        need_build = not (self.zvec_path.exists() and any(self.zvec_path.iterdir()))

        if need_build:
            logger.info(f"Building ZVEC index at {self.zvec_path}...")
            os.makedirs(self.zvec_path, exist_ok=True)
            self.collection = zvec.create_and_open(path=str(self.zvec_path), schema=schema)
            # Encode and insert chunks
            batch_size = 32
            for i in range(0, len(self.chunk_db), batch_size):
                batch_chunks = self.chunk_db[i:i + batch_size]
                texts = [c[0] for c in batch_chunks]
                embeddings_res = self.embedder.encode(texts, return_dense=True, return_sparse=False)
                embeddings = embeddings_res["dense_embeddings"].detach().cpu().numpy().tolist()

                docs = []
                for j, (text, full_doc_id) in enumerate(batch_chunks):
                    # We use i+j as our internal index so we can map back to chunk_db later.
                    doc_id = str(i + j)
                    docs.append(zvec.Doc(id=doc_id, vectors={"embedding": embeddings[j]}))

                self.collection.insert(docs)
                logger.debug(f"Inserted batch {i // batch_size} of {len(self.chunk_db) // batch_size}")
            logger.info("ZVEC index build complete.")
        else:
            logger.info(f"Loading existing ZVEC index from {self.zvec_path}")
            self.collection = zvec.open(path=str(self.zvec_path))

    async def aclear_cache(self):
        """Mock method to be compatible with main.py calls"""
        pass

    async def aquery(self, query: str, param=None, system_prompt: Optional[str] = None):
        """
        Mimics LightRAG pipeline using custom ZVEC workflow as described by the user.
        """
        # Step 1: Generate search queries
        logger.info(f"Expanding query: {query}")
        # Note: in a real implementation we might pass LLM tokenizer specifically, but here we can 
        # use generate_with_llm directly using standard text replacements just like in rag_engine.py

        prompt_for_search = self._prepare_search_prompt(query)
        search_queries_str = await generate_with_llm(prompt=prompt_for_search)

        list_of_search_queries = self._parse_search_queries(search_queries_str)

        is_meaningless = False
        for cur_query in list_of_search_queries:
            dist = self._calculate_dist(MEANINGLESS_REQUEST_ANSWER, cur_query)
            if dist <= LEXICAL_DISTANCE_THRESHOLD:
                is_meaningless = True
                break

        if is_meaningless:
            return "К сожалению, не удалось сформулировать поисковые запросы для этого вопроса."

        logger.info(f"Generated search queries: {list_of_search_queries}")

        # Step 2: Retrieve chunks
        top_k = getattr(param, 'chunk_top_k', settings.chunk_top_k) if param else settings.chunk_top_k
        retrieval_queries = self._stable_unique([query] + list_of_search_queries)
        grouped_results: Dict[str, list[list[RetrievedChunk]]] = {
            "multilingual_dense": [],
            "lexical": [],
        }

        for cur_search_query in retrieval_queries:
            grouped_results["multilingual_dense"].append(self._dense_results(cur_search_query, top_k))
            grouped_results["lexical"].append(self._bm25_results(cur_search_query, top_k))

        if not any(result for result_lists in grouped_results.values() for result in result_lists):
            return "К сожалению, в базе данных недостаточно информации для ответа на этот вопрос."

        fusion_result = self.fusion.fuse(grouped_results, top_k=top_k)
        if not fusion_result.chunks:
            return "К сожалению, в базе данных недостаточно информации для ответа на этот вопрос."

        rerank_result = await self.reranker.rerank(query=query, chunks=fusion_result.chunks, top_n=top_k)
        context_str, _sources = self.assembler.assemble(rerank_result.reranked_chunks)
        if not context_str.strip():
            return "К сожалению, в базе данных недостаточно информации для ответа на этот вопрос."
        qa_prompt = f"Контекст:\n{context_str}\n\nВопрос:\n{query}"

        # Step 5: Generation
        logger.info("Generating final answer with Zvec RAG...")
        stream = getattr(param, 'stream', False) if param else False
        history_messages = getattr(param, 'conversation_history', []) if param else []

        # Just use generation with LightRAG's wrapper utility
        answer = await generate_with_llm(
            prompt=qa_prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            stream=stream
        )

        return answer

    def _prepare_search_prompt(self, query: str) -> str:
        prompt = SEARCH_SYSTEM_PROMPT + "\n\n"
        for ex in EXAMPLES_OF_SEARCH_QUERIES:
            prompt += f"Пользователь: {ex['question']}\nАгент:\n" + "\n".join(ex['search_queries']) + "\n\n"
        prompt += f"Пользователь: {query}\nАгент:\n"
        return prompt

    def _calculate_dist(self, t1: str, t2: str) -> float:
        # Simple distance placeholder. In full implementation, import cer from jiwer.
        # But for Meno-Core we can reuse the lexical distance function if imported.
        # Here we just check equality to avoid external deps if not strictly needed.
        return 0.0 if t1.strip().lower() == t2.strip().lower() else 1.0

    def _dense_results(self, query: str, top_k: int) -> list[RetrievedChunk]:
        question_vector_res = self.embedder.encode([query], return_dense=True, return_sparse=False)
        q_vec = question_vector_res["dense_embeddings"][0].detach().cpu().numpy().tolist()
        zvec_results = self.collection.query(
            zvec.VectorQuery("embedding", vector=q_vec),
            topk=top_k
        )
        results: list[RetrievedChunk] = []
        for result in zvec_results:
            idx = int(result["id"])
            if idx >= len(self.legacy_chunks):
                continue
            results.append(
                RetrievedChunk(
                    chunk=self.legacy_chunks[idx],
                    score=float(result.get("score", 0.0)),
                    source="multilingual_dense",
                )
            )
        return results

    def _bm25_results(self, query: str, top_k: int) -> list[RetrievedChunk]:
        bm25_terms = tokenize_for_bm25(query)
        if not bm25_terms:
            return []
        chunk_scores = np.asarray(self.bm25.get_scores(bm25_terms), dtype=float)
        indices = self._top_positive_indices(chunk_scores, top_k)
        return [
            RetrievedChunk(
                chunk=self.legacy_chunks[idx],
                score=float(chunk_scores[idx]),
                source="lexical",
            )
            for idx in indices
            if idx < len(self.legacy_chunks)
        ]

    @staticmethod
    def _top_positive_indices(scores: np.ndarray, top_k: int) -> list[int]:
        positive_indices = np.flatnonzero(scores > 0)
        if positive_indices.size == 0:
            return []
        if positive_indices.size <= top_k:
            candidates = positive_indices.tolist()
        else:
            candidate_positions = np.argpartition(scores[positive_indices], -top_k)[-top_k:]
            candidates = positive_indices[candidate_positions].tolist()
        return sorted(candidates, key=lambda idx: (-float(scores[idx]), idx))

    @staticmethod
    def _stable_unique(items: List[str]) -> List[str]:
        result: list[str] = []
        seen: set[str] = set()
        for item in items:
            normalized = item.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return result

    def _parse_search_queries(self, search_queries_str: str) -> list[str]:
        return self._stable_unique(search_queries_str.splitlines())[:2]

    def _resolve_embedder_dimension(self) -> int:
        return int(getattr(self.embedder.model.config, "hidden_size"))

    @staticmethod
    def _normalize_document_id(full_doc_id: str) -> str:
        chunk_marker = full_doc_id.find("_chunk")
        return full_doc_id[:chunk_marker] if chunk_marker > 0 else full_doc_id

    def _build_legacy_chunk(self, idx: int, content: str, full_doc_id: str) -> Chunk:
        document_id = self._normalize_document_id(full_doc_id)
        return Chunk(
            chunk_id=f"legacy_chunk_{idx}",
            text=content,
            text_for_dense=content,
            text_for_bm25=" ".join(tokenize_for_bm25(content)),
            metadata=ChunkMetadata(
                document_id=document_id,
                document_title=document_id,
                chunk_index=idx,
            ),
        )
