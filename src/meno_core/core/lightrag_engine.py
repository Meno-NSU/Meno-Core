import logging
from collections.abc import AsyncIterator, MutableMapping
from typing import TYPE_CHECKING, Any

from meno_core.config.settings import settings
from meno_core.core.rag_engine import (
    CHUNK_TOP_K,
    ENTITY_MAX_TOKENS,
    QUERY_MAX_TOKENS,
    RELATION_MAX_TOKENS,
    TOP_K,
)
from meno_core.core.lightrag_timing import (
    install_lightrag_timing_hooks,
    reset_rag_request_trace,
    start_rag_request_trace,
)
from meno_core.core.rag_runtime import RagChatRequest

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from lightrag import LightRAG  # type: ignore[import-untyped]
    from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]
    from meno_core.core.gte_embedding import GTEEmbedding
else:
    LightRAG = Any
    BM25Okapi = Any
    GTEEmbedding = Any


class LightRAGEngine:
    def __init__(
            self,
            rag_instance: LightRAG,
            embedder: GTEEmbedding,
            bm25: BM25Okapi,
            chunk_db: list[tuple[Any, Any]],
    ):
        self.rag_instance = rag_instance
        self.embedder = embedder
        self.bm25 = bm25
        self.chunk_db = chunk_db
        install_lightrag_timing_hooks()

    async def aquery(self, *args, **kwargs):
        """Run LightRAG query with request-scoped timing instrumentation."""
        request_id = str(kwargs.pop("request_id", "unknown-request"))
        session_id = str(kwargs.pop("session_id", "unknown-session"))
        knowledge_base_id = str(kwargs.pop("knowledge_base_id", "unknown-kb"))
        rag_engine_id = str(kwargs.pop("rag_engine_id", "lightrag"))
        model = str(kwargs.pop("model", "unknown-model"))
        kwargs.pop("base_url", None)
        route_reason = str(kwargs.pop("route_reason", "direct lightrag query"))
        timings_sink = kwargs.pop("timings_sink", None)
        if timings_sink is not None and not isinstance(timings_sink, MutableMapping):
            timings_sink = None

        param = kwargs.get("param")
        if param is None and len(args) > 1:
            param = args[1]
        stream = bool(getattr(param, "stream", False))

        trace, token = start_rag_request_trace(
            request_id=request_id,
            session_id=session_id,
            knowledge_base_id=knowledge_base_id,
            rag_engine_id=rag_engine_id,
            model=model,
            stream=stream,
            route_reason=route_reason,
        )
        try:
            result = await self.rag_instance.aquery(*args, **kwargs)
        except Exception as error:
            trace.finalize(timings_sink=timings_sink, error=error)
            reset_rag_request_trace(token)
            raise

        if isinstance(result, AsyncIterator) or hasattr(result, "__aiter__"):
            async def _wrapped_stream():
                import time as _time

                # Emit individual sub-stage dicts from timing hooks
                for stage_name, duration_ms, meta in trace.stage_events:
                    yield {
                        "_stage": stage_name,
                        "status": "completed",
                        "duration_ms": round(duration_ms, 2),
                        "detail": meta or None,
                    }
                yield {"_stage": "generation", "status": "started"}

                gen_start = _time.time()
                first = True
                try:
                    async for part in result:
                        if part and first:
                            trace.mark_llm_stream_first_chunk()
                            first = False
                        yield str(part)
                    trace.mark_llm_stream_complete()

                    gen_ms = round((_time.time() - gen_start) * 1000, 2)
                    yield {"_stage": "generation", "status": "completed", "duration_ms": gen_ms}

                    trace.finalize(timings_sink=timings_sink)
                except Exception as error:
                    trace.finalize(timings_sink=timings_sink, error=error)
                    raise
                finally:
                    reset_rag_request_trace(token)

            return _wrapped_stream()

        trace.finalize(timings_sink=timings_sink)
        reset_rag_request_trace(token)
        return result

    async def answer(
        self,
        request: RagChatRequest,
        timings_sink: dict[str, float] | None = None,
    ) -> str | AsyncIterator[str]:
        from lightrag import QueryParam  # type: ignore[import-untyped]

        return await self.aquery(
            request.question,
            param=QueryParam(
                mode=settings.query_mode,
                top_k=TOP_K,
                chunk_top_k=CHUNK_TOP_K,
                max_total_tokens=QUERY_MAX_TOKENS,
                history_turns=len(request.history),
                conversation_history=request.history,
                max_entity_tokens=ENTITY_MAX_TOKENS,
                max_relation_tokens=RELATION_MAX_TOKENS,
                stream=request.stream,
            ),
            system_prompt=request.system_prompt,
            request_id=request.request_id,
            session_id=request.session_id,
            knowledge_base_id=request.knowledge_base_id,
            rag_engine_id=request.rag_engine_id,
            model=request.model,
            base_url=request.base_url,
            route_reason=request.route_reason,
            timings_sink=timings_sink,
        )

    async def aclear_cache(self):
        """Pass-through to LightRAG's aclear_cache"""
        if hasattr(self.rag_instance, 'aclear_cache'):
            return await self.rag_instance.aclear_cache()
