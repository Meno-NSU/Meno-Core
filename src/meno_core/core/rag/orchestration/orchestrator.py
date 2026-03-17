import asyncio
import logging
import time
from typing import Any, Dict, List

from meno_core.core.rag.config import ChunkRagConfig
from meno_core.core.rag.debug_utils import build_retrieved_chunk_preview
from meno_core.core.rag.fusion.merger import HybridFusion
from meno_core.core.rag.generation.context_assembler import ContextAssembler
from meno_core.core.rag.generation.generator import AnswerGenerator
from meno_core.core.rag.models import RagDebugInfo, RagRequest, RagResponse, RetrievedChunk
from meno_core.core.rag.query_processor import QueryProcessor
from meno_core.core.rag.rerank.qwen_reranker import QwenCausalReranker
from meno_core.core.rag.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)
retrieval_logger = logging.getLogger("chunk_rag.retrieval")
pipeline_logger = logging.getLogger("chunk_rag.pipeline")


class ChunkRagOrchestrator:
    """
    Ties together the Chunk RAG pipeline:
    1. Query rewrite
    2. Multi-retriever retrieval (multilingual dense + russian dense + BM25)
    3. Late fusion
    4. Qwen rerank
    5. Context assembly
    6. Final answer generation
    """

    def __init__(
        self,
        config: ChunkRagConfig,
        dense_retrievers: Dict[str, BaseRetriever],
        lexical_retriever: BaseRetriever,
        reranker: QwenCausalReranker,
    ):
        self.config = config
        self.dense_retrievers = dense_retrievers
        self.lexical_retriever = lexical_retriever
        self.reranker = reranker

        self.query_processor = QueryProcessor()
        self.fusion = HybridFusion(
            weights={
                "multilingual_dense": self.config.fusion_weight_multilingual,
                "russian_dense": self.config.fusion_weight_russian,
                "lexical": self.config.fusion_weight_bm25,
            },
            preview_k=self.config.retrieval_preview_k,
        )
        self.assembler = ContextAssembler(token_budget=self.config.token_budget)
        self.generator = AnswerGenerator(
            reliability_mode_enabled=self.config.reliability_mode_enabled,
            hallucination_threshold=self.config.hallucination_threshold,
        )

    async def _timed_retrieve(
        self,
        source_name: str,
        query: str,
        retriever: BaseRetriever,
        top_k: int,
    ) -> tuple[str, str, list[RetrievedChunk], float, int]:
        started_at = time.perf_counter()
        result = await retriever.retrieve(query, top_k)
        return source_name, query, result, round((time.perf_counter() - started_at) * 1000, 2), top_k

    async def answer(self, request: RagRequest) -> RagResponse:
        start_time = time.time()
        telemetry: Dict[str, Any] = {"steps_latency_ms": {}}
        debug_info = RagDebugInfo()
        request_id = request.request_id or "-"
        session_id = request.session_id or "-"

        pipeline_logger.info(
            "request-start request_id=%s session_id=%s question_len=%s history_messages=%s",
            request_id,
            session_id,
            len(request.question),
            len(request.history),
        )

        try:
            step_start = time.time()
            if self.config.rewrite_enabled:
                representations = await self.query_processor.process_query(request.question, request.history)
            else:
                representations = await self.query_processor.process_query(request.question, [])

            telemetry["steps_latency_ms"]["query_rewrite"] = round((time.time() - step_start) * 1000, 2)
            debug_info.rewritten_query = representations.rewritten_query
            debug_info.resolved_coreferences = representations.resolved_coreferences
            debug_info.expanded_abbreviations = representations.expanded_abbreviations
            debug_info.search_queries = representations.search_queries
            debug_info.hypothetical_document = representations.hypothetical_document

            if not representations.is_meaningful:
                pipeline_logger.info(
                    "request-finished request_id=%s session_id=%s total_ms=%.2f meaningful=false",
                    request_id,
                    session_id,
                    (time.time() - start_time) * 1000,
                )
                return RagResponse(
                    answer="Похоже, что ваш запрос не содержит конкретного вопроса или не относится к НГУ. Пожалуйста, уточните ваш запрос.",
                    insufficient_information=False,
                    debug=debug_info,
                )

            step_start = time.time()
            shared_queries = self._stable_unique(
                [request.question, representations.resolved_coreferences] + representations.search_queries
            )
            dense_queries = list(shared_queries)
            if self.config.hypothetical_doc_enabled and representations.hypothetical_document:
                dense_queries = self._stable_unique(dense_queries + [representations.hypothetical_document])

            retrieval_requests: list[asyncio.Future] = []
            for source_name, retriever in self.dense_retrievers.items():
                top_k = (
                    self.config.top_k_dense_multilingual
                    if source_name == "multilingual_dense"
                    else self.config.top_k_dense_russian
                )
                for query in dense_queries:
                    retrieval_requests.append(self._timed_retrieve(source_name, query, retriever, top_k))

            for query in shared_queries:
                retrieval_requests.append(
                    self._timed_retrieve("lexical", query, self.lexical_retriever, self.config.top_k_bm25)
                )

            raw_results = await asyncio.gather(
                *retrieval_requests,
                return_exceptions=True,
            )

            grouped_results: Dict[str, List[List[RetrievedChunk]]] = {
                "multilingual_dense": [],
                "russian_dense": [],
                "lexical": [],
            }
            retrieval_latency_entries: Dict[str, List[Dict[str, Any]]] = {
                "multilingual_dense": [],
                "russian_dense": [],
                "lexical": [],
            }
            if self.config.debug_retrieval:
                telemetry["retrieval_previews"] = {
                    "multilingual_dense": [],
                    "russian_dense": [],
                    "lexical": [],
                }

            for result in raw_results:
                if isinstance(result, Exception):
                    logger.error("Retriever failed: %s", result, exc_info=result)
                    continue

                source_name, query, retrieved_chunks, latency_ms, top_k = result

                grouped_results[source_name].append(retrieved_chunks)
                retrieval_latency_entries[source_name].append(
                    {
                        "query": query,
                        "top_k": top_k,
                        "latency_ms": latency_ms,
                        "hits": len(retrieved_chunks),
                    }
                )
                if self.config.debug_retrieval:
                    telemetry["retrieval_previews"][source_name].append(
                        {
                            "query": query,
                            "hits": build_retrieved_chunk_preview(retrieved_chunks, self.config.retrieval_preview_k),
                        }
                    )

            pooled_results = {
                source_name: self._flatten_and_deduplicate(results)
                for source_name, results in grouped_results.items()
            }
            telemetry["steps_latency_ms"]["retrieval"] = round((time.time() - step_start) * 1000, 2)
            telemetry["retrieved_counts"] = {
                source_name: len(results)
                for source_name, results in pooled_results.items()
            }
            telemetry["retrieval_latency_ms"] = retrieval_latency_entries
            telemetry["retrieval_latency_summary_ms"] = {
                source_name: {
                    "total_ms": round(sum(item["latency_ms"] for item in entries), 2),
                    "max_ms": round(max((item["latency_ms"] for item in entries), default=0.0), 2),
                    "calls": len(entries),
                }
                for source_name, entries in retrieval_latency_entries.items()
            }
            if self.config.debug_retrieval:
                retrieval_logger.info(
                    "retrieval-summary question=%r dense_queries=%s lexical_queries=%s latency_summary_ms=%s retrieved_counts=%s",
                    request.question,
                    len(dense_queries),
                    len(shared_queries),
                    telemetry["retrieval_latency_summary_ms"],
                    telemetry["retrieved_counts"],
                )

            step_start = time.time()
            fusion_result = self.fusion.fuse(pooled_results, top_k=self.config.top_k_after_fusion)
            telemetry["steps_latency_ms"]["fusion"] = round((time.time() - step_start) * 1000, 2)
            if self.config.debug_retrieval:
                telemetry["fused_preview"] = fusion_result.fused_preview

            step_start = time.time()
            rerank_result = await self.reranker.rerank(
                query=representations.resolved_coreferences,
                chunks=fusion_result.chunks,
                top_n=self.config.top_n_after_rerank,
            )
            telemetry["steps_latency_ms"]["rerank"] = round((time.time() - step_start) * 1000, 2)
            if self.config.debug_retrieval:
                telemetry["rerank_preview"] = rerank_result.preview

            step_start = time.time()
            context_str, sources = self.assembler.assemble(rerank_result.reranked_chunks)
            telemetry["steps_latency_ms"]["context_assembly"] = round((time.time() - step_start) * 1000, 2)

            step_start = time.time()
            answer_text, insuff_flag = await self.generator.generate_answer(
                question=representations.resolved_coreferences,
                context=context_str,
                sources=sources,
                history=request.history,
                stream=False,
            )
            telemetry["steps_latency_ms"]["generation"] = round((time.time() - step_start) * 1000, 2)
            telemetry["total_latency_ms"] = round((time.time() - start_time) * 1000, 2)
            debug_info.retrieval_stats = telemetry
            pipeline_logger.info(
                "request-finished request_id=%s session_id=%s total_ms=%s steps_ms=%s retrieval_summary_ms=%s retrieved_counts=%s dense_queries=%s lexical_queries=%s sources=%s insufficient=%s",
                request_id,
                session_id,
                telemetry["total_latency_ms"],
                telemetry["steps_latency_ms"],
                telemetry["retrieval_latency_summary_ms"],
                telemetry["retrieved_counts"],
                len(dense_queries),
                len(shared_queries),
                len(sources),
                insuff_flag,
            )

            return RagResponse(
                answer=answer_text,
                sources=sources,
                debug=debug_info,
                insufficient_information=insuff_flag,
            )
        except Exception as error:
            logger.error("Error in chunk RAG orchestrator: %s", error, exc_info=True)
            pipeline_logger.error(
                "request-failed request_id=%s session_id=%s total_ms=%.2f error=%s",
                request_id,
                session_id,
                (time.time() - start_time) * 1000,
                error,
            )
            return RagResponse(
                answer="Произошла системная ошибка при обработке вашего запроса.",
                insufficient_information=True,
            )

    @staticmethod
    def _stable_unique(items: List[str]) -> List[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for item in items:
            if not item or item in seen:
                continue
            seen.add(item)
            unique.append(item)
        return unique

    def _flatten_and_deduplicate(self, lists_of_chunks: List[List[RetrievedChunk]]) -> List[RetrievedChunk]:
        chunk_map = {}
        for chunk_list in lists_of_chunks:
            for chunk_wrapper in chunk_list:
                chunk_id = chunk_wrapper.chunk.chunk_id
                if chunk_id not in chunk_map or chunk_wrapper.score > chunk_map[chunk_id].score:
                    chunk_map[chunk_id] = chunk_wrapper
        return list(chunk_map.values())
