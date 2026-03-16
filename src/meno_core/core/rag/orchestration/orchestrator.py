import asyncio
import logging
import time
from typing import Dict, Any, List

from meno_core.core.rag.config import ChunkRagConfig
from meno_core.core.rag.models import RagRequest, RagResponse, RagDebugInfo, RagSource, RetrievedChunk
from meno_core.core.rag.query_processor import QueryProcessor
from meno_core.core.rag.retrieval.base import BaseRetriever
from meno_core.core.rag.fusion.merger import HybridFusion
from meno_core.core.rag.rerank.cross_encoder import CrossEncoderReranker
from meno_core.core.rag.generation.context_assembler import ContextAssembler
from meno_core.core.rag.generation.generator import AnswerGenerator

logger = logging.getLogger(__name__)


class ChunkRagOrchestrator:
    """
    Ties together the entire Chunk RAG pipeline:
    1. Parse & Normalize Request
    2. Query Rewrite
    3. Hybrid Retrieve (Dense + Lexical)
    4. Fusion & Rerank
    5. Context Assembly
    6. LLM Generation & Telemetry
    """

    def __init__(
        self,
        config: ChunkRagConfig,
        dense_retriever: BaseRetriever,
        lexical_retriever: BaseRetriever,
    ):
        self.config = config
        self.dense_retriever = dense_retriever
        self.lexical_retriever = lexical_retriever
        
        # Pipeline components
        self.query_processor = QueryProcessor()
        self.fusion = HybridFusion(dense_weight=0.6, lexical_weight=0.4)
        self.reranker = CrossEncoderReranker(filter_threshold=-10.0) # adjust if needed
        self.assembler = ContextAssembler(token_budget=self.config.token_budget)
        self.generator = AnswerGenerator(
            reliability_mode_enabled=self.config.reliability_mode_enabled,
            hallucination_threshold=self.config.hallucination_threshold
        )

    async def answer(self, request: RagRequest) -> RagResponse:
        """
        Main entry point for answering a query.
        """
        start_time = time.time()
        telemetry: Dict[str, Any] = {"steps_latency_ms": {}}
        debug_info = RagDebugInfo()

        try:
            # 1. Query Rewrite
            step_start = time.time()
            if self.config.rewrite_enabled:
                representations = await self.query_processor.process_query(request.question, request.history)
            else:
                representations = await self.query_processor.process_query(request.question, [])  # Minimal processing fallback
            
            telemetry["steps_latency_ms"]["query_rewrite"] = round((time.time() - step_start) * 1000, 2)
            
            # Populate Debug Info
            debug_info.rewritten_query = representations.rewritten_query
            debug_info.resolved_coreferences = representations.resolved_coreferences
            debug_info.expanded_abbreviations = representations.expanded_abbreviations
            debug_info.search_queries = representations.search_queries
            debug_info.hypothetical_document = representations.hypothetical_document

            # 2. Meaningfulness Check
            if not representations.is_meaningful:
                return RagResponse(
                    answer="Похоже, что ваш запрос не содержит конкретного вопроса или не относится к НГУ. Пожалуйста, уточните ваш запрос.",
                    insufficient_information=False,
                    debug=debug_info
                )

            # 3. Parallel Retrieval
            step_start = time.time()
            # Which queries to run?
            queries_to_run = list(set([request.question, representations.resolved_coreferences] + representations.search_queries))
            
            # If hypothetical doc enabled, append to dense queries only ideally, 
            # but for simplicity we can include it in the generic search query pool.
            if self.config.hypothetical_doc_enabled and representations.hypothetical_document:
                queries_to_run.append(representations.hypothetical_document)

            dense_tasks = [self.dense_retriever.retrieve(q, self.config.top_k_dense) for q in queries_to_run]
            lexical_tasks = [self.lexical_retriever.retrieve(q, self.config.top_k_bm25) for q in queries_to_run]

            all_results = await asyncio.gather(*(dense_tasks + lexical_tasks), return_exceptions=True)
            
            dense_lists = [r for r in all_results[:len(dense_tasks)] if not isinstance(r, Exception)]
            lexical_lists = [r for r in all_results[len(dense_tasks):] if not isinstance(r, Exception)]
            
            # Flatten lists to pool all retrieved chunks
            pooled_dense = self._flatten_and_deduplicate(dense_lists)
            pooled_lexical = self._flatten_and_deduplicate(lexical_lists)
            
            telemetry["steps_latency_ms"]["retrieval"] = round((time.time() - step_start) * 1000, 2)
            telemetry["retrieved_dense_count"] = len(pooled_dense)
            telemetry["retrieved_lexical_count"] = len(pooled_lexical)

            # 4. Fusion
            step_start = time.time()
            fused_chunks = self.fusion.fuse(pooled_dense, pooled_lexical, top_k=self.config.top_k_after_fusion)
            telemetry["steps_latency_ms"]["fusion"] = round((time.time() - step_start) * 1000, 2)

            # 5. Rerank
            step_start = time.time()
            reranked_chunks = await self.reranker.rerank(
                query=representations.resolved_coreferences, 
                chunks=fused_chunks, 
                top_n=self.config.top_n_after_rerank
            )
            telemetry["steps_latency_ms"]["rerank"] = round((time.time() - step_start) * 1000, 2)

            # 6. Context Assembly
            step_start = time.time()
            context_str, sources = self.assembler.assemble(reranked_chunks)
            telemetry["steps_latency_ms"]["context_assembly"] = round((time.time() - step_start) * 1000, 2)

            # 7. Final Generation
            step_start = time.time()
            answer_text, insuff_flag = await self.generator.generate_answer(
                question=representations.resolved_coreferences,
                context=context_str,
                sources=sources,
                history=request.history,
                stream=False
            )
            telemetry["steps_latency_ms"]["generation"] = round((time.time() - step_start) * 1000, 2)
            
            total_time = round((time.time() - start_time) * 1000, 2)
            telemetry["total_latency_ms"] = total_time
            debug_info.retrieval_stats = telemetry

            return RagResponse(
                answer=answer_text,
                sources=sources,
                debug=debug_info,
                insufficient_information=insuff_flag
            )

        except Exception as e:
            logger.error(f"Error in chunk RAG orchestrator: {e}", exc_info=True)
            return RagResponse(
                answer="Произошла системная ошибка при обработке вашего запроса.",
                insufficient_information=True
            )
            
    def _flatten_and_deduplicate(self, lists_of_chunks: List[List[RetrievedChunk]]) -> List[RetrievedChunk]:
        """Flatten nested list and keep only highest score for duplicates."""
        chunk_map = {}
        for lst in lists_of_chunks:
            for chunk in lst:
                cid = chunk.chunk.chunk_id
                if cid not in chunk_map or chunk.score > chunk_map[cid].score:
                    chunk_map[cid] = chunk
        return list(chunk_map.values())
