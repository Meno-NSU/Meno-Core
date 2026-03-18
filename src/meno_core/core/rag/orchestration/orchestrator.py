import logging
import time
from typing import Any, Dict, List

from meno_core.core.lightrag_timing import get_current_rag_trace
from meno_core.core.rag.config import ChunkRagConfig
from meno_core.core.rag.debug_utils import build_retrieved_chunk_preview
from meno_core.core.rag.fusion.merger import HybridFusion
from meno_core.core.rag.generation.context_assembler import ContextAssembler, estimate_tokens
from meno_core.core.rag.generation.generator import AnswerGenerator
from meno_core.core.rag.models import QueryRepresentations, RagDebugInfo, RagRequest, RagResponse, RetrievedChunk
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

    async def _timed_retrieve_many(
        self,
        source_name: str,
        queries: List[str],
        retriever: BaseRetriever,
        top_k: int,
    ) -> tuple[str, List[str], list[list[RetrievedChunk]], float, int]:
        started_at = time.perf_counter()
        result = await retriever.retrieve_many(queries, top_k)
        return source_name, queries, result, round((time.perf_counter() - started_at) * 1000, 2), top_k

    async def answer(self, request: RagRequest) -> RagResponse:
        start_time = time.time()
        telemetry: Dict[str, Any] = {"steps_latency_ms": {}}
        debug_info = RagDebugInfo()
        request_id = request.request_id or "-"
        session_id = request.session_id or "-"
        trace = get_current_rag_trace()

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
                representations = self._build_passthrough_representations(request.question)

            rewrite_ms = round((time.time() - step_start) * 1000, 2)
            telemetry["steps_latency_ms"]["query_rewrite"] = rewrite_ms
            if trace is not None:
                trace.record_stage(
                    "query_rewrite",
                    rewrite_ms,
                    meta={"search_queries": len(representations.search_queries)},
                )
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
            dense_queries, lexical_queries = self._build_retrieval_queries(request.question, representations)
            retrieval_batches = []
            for source_name, retriever in self.dense_retrievers.items():
                top_k = (
                    self.config.top_k_dense_multilingual
                    if source_name == "multilingual_dense"
                    else self.config.top_k_dense_russian
                )
                retrieval_batches.append(
                    await self._timed_retrieve_many(source_name, dense_queries, retriever, top_k)
                )
            retrieval_batches.append(
                await self._timed_retrieve_many("lexical", lexical_queries, self.lexical_retriever, self.config.top_k_bm25)
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

            for source_name, queries, batch_results, latency_ms, top_k in retrieval_batches:
                grouped_results[source_name].extend(batch_results)
                query_entries = []
                for query, retrieved_chunks in zip(queries, batch_results):
                    query_entries.append(
                        {
                            "query": query,
                            "top_k": top_k,
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
                retrieval_latency_entries[source_name].append(
                    {
                        "queries": query_entries,
                        "latency_ms": latency_ms,
                        "batch_size": len(queries),
                        "top_k": top_k,
                    }
                )

            pooled_counts = {
                source_name: len(self._flatten_and_deduplicate(results))
                for source_name, results in grouped_results.items()
            }
            retrieval_ms = round((time.time() - step_start) * 1000, 2)
            telemetry["steps_latency_ms"]["vector_retrieval"] = retrieval_ms
            telemetry["retrieved_counts"] = pooled_counts
            telemetry["retrieval_latency_ms"] = retrieval_latency_entries
            telemetry["retrieval_latency_summary_ms"] = {
                source_name: {
                    "total_ms": round(sum(item["latency_ms"] for item in entries), 2),
                    "max_ms": round(max((item["latency_ms"] for item in entries), default=0.0), 2),
                    "calls": len(entries),
                    "batch_sizes": [item["batch_size"] for item in entries],
                }
                for source_name, entries in retrieval_latency_entries.items()
            }
            telemetry["retrieval_call_count"] = sum(len(entries) for entries in retrieval_latency_entries.values())
            if trace is not None:
                trace.increment_counter("retrieval_calls", telemetry["retrieval_call_count"])
                trace.record_stage(
                    "vector_retrieval",
                    retrieval_ms,
                    meta={
                        "dense_queries": len(dense_queries),
                        "lexical_queries": len(lexical_queries),
                        "retrieved_counts": pooled_counts,
                    },
                )
            if self.config.debug_retrieval:
                retrieval_logger.info(
                    "retrieval-summary question=%r dense_queries=%s lexical_queries=%s latency_summary_ms=%s retrieved_counts=%s",
                    request.question,
                    len(dense_queries),
                    len(lexical_queries),
                    telemetry["retrieval_latency_summary_ms"],
                    telemetry["retrieved_counts"],
                )

            step_start = time.time()
            fusion_result = self.fusion.fuse(grouped_results, top_k=self.config.top_k_after_fusion)
            fusion_ms = round((time.time() - step_start) * 1000, 2)
            telemetry["steps_latency_ms"]["fusion"] = fusion_ms
            telemetry["fusion_candidate_count"] = len(fusion_result.chunks)
            if trace is not None:
                trace.increment_counter("fusion_candidates", len(fusion_result.chunks))
                trace.record_stage(
                    "fusion",
                    fusion_ms,
                    meta={"candidates": len(fusion_result.chunks)},
                )
            if self.config.debug_retrieval:
                telemetry["fused_preview"] = fusion_result.fused_preview

            step_start = time.time()
            rerank_result = await self.reranker.rerank(
                query=representations.resolved_coreferences,
                chunks=fusion_result.chunks,
                top_n=self.config.top_n_after_rerank,
            )
            rerank_ms = round((time.time() - step_start) * 1000, 2)
            telemetry["steps_latency_ms"]["rerank"] = rerank_ms
            telemetry["rerank_candidate_count"] = len(fusion_result.chunks)
            telemetry["rerank_kept_count"] = len(rerank_result.reranked_chunks)
            if trace is not None:
                trace.increment_counter("rerank_candidates", len(fusion_result.chunks))
                trace.increment_counter("rerank_kept", len(rerank_result.reranked_chunks))
                trace.record_stage(
                    "rerank",
                    rerank_ms,
                    meta={
                        "candidates": len(fusion_result.chunks),
                        "kept": len(rerank_result.reranked_chunks),
                    },
                )
            if self.config.debug_retrieval:
                telemetry["rerank_preview"] = rerank_result.preview

            step_start = time.time()
            context_str, sources = self.assembler.assemble(rerank_result.reranked_chunks)
            context_ms = round((time.time() - step_start) * 1000, 2)
            telemetry["steps_latency_ms"]["context_build"] = context_ms
            telemetry["context_token_count"] = estimate_tokens(context_str) if context_str else 0
            if trace is not None:
                trace.increment_counter("context_sources", len(sources))
                trace.record_stage(
                    "context_build",
                    context_ms,
                    meta={
                        "sources": len(sources),
                        "context_tokens": telemetry["context_token_count"],
                    },
                )

            step_start = time.time()
            answer_text, insuff_flag = await self.generator.generate_answer(
                question=representations.resolved_coreferences,
                context=context_str,
                sources=sources,
                history=request.history,
                stream=False,
            )
            generation_ms = round((time.time() - step_start) * 1000, 2)
            telemetry["steps_latency_ms"]["llm_nonstream"] = generation_ms
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
                len(lexical_queries),
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
    def _build_passthrough_representations(question: str) -> QueryRepresentations:
        return QueryRepresentations(
            original_query=question,
            rewritten_query=question,
            resolved_coreferences=question,
            search_queries=[question],
            hypothetical_document="",
            is_meaningful=True,
        )

    @classmethod
    def _build_retrieval_queries(
        cls,
        original_question: str,
        representations: QueryRepresentations,
    ) -> tuple[list[str], list[str]]:
        search_queries = cls._stable_unique(representations.search_queries)[:2]
        lexical_queries = cls._stable_unique(
            [
                representations.rewritten_query,
                representations.resolved_coreferences,
                *search_queries,
            ]
        )
        dense_queries = list(lexical_queries)
        if representations.hypothetical_document:
            dense_queries = cls._stable_unique(dense_queries + [representations.hypothetical_document])
        if not dense_queries:
            dense_queries = [original_question]
        if not lexical_queries:
            lexical_queries = [original_question]
        return dense_queries, lexical_queries

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
