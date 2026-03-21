from __future__ import annotations

import contextvars
import logging
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, MutableMapping

request_logger = logging.getLogger("meno_core.request")
logger = logging.getLogger(__name__)

_current_trace: contextvars.ContextVar["RagRequestTrace | None"] = contextvars.ContextVar(
    "rag_request_trace",
    default=None,
)
_hooks_installed = False


def _safe_int(value: Any) -> int | None:
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _build_context_meta(_: tuple[Any, ...], __: dict[str, Any], result: Any) -> dict[str, Any]:
    if not isinstance(result, tuple) or len(result) != 2:
        return {}
    _context, raw_data = result
    if not isinstance(raw_data, dict):
        return {}
    data = raw_data.get("data", {})
    if not isinstance(data, dict):
        return {}
    return {
        "entities": len(data.get("entities", [])),
        "relations": len(data.get("relationships", [])),
        "chunks": len(data.get("chunks", [])),
    }


def _keywords_meta(_: tuple[Any, ...], __: dict[str, Any], result: Any) -> dict[str, Any]:
    if not isinstance(result, tuple) or len(result) != 2:
        return {}
    high_level, low_level = result
    return {
        "high_level_keywords": len(high_level or []),
        "low_level_keywords": len(low_level or []),
    }


def _local_query_meta(_: tuple[Any, ...], __: dict[str, Any], result: Any) -> dict[str, Any]:
    if not isinstance(result, tuple) or len(result) != 2:
        return {}
    entities, relations = result
    return {
        "entities": len(entities or []),
        "relations": len(relations or []),
    }


def _global_query_meta(_: tuple[Any, ...], __: dict[str, Any], result: Any) -> dict[str, Any]:
    if not isinstance(result, tuple) or len(result) != 2:
        return {}
    relations, entities = result
    return {
        "entities": len(entities or []),
        "relations": len(relations or []),
    }


def _vector_context_meta(_: tuple[Any, ...], __: dict[str, Any], result: Any) -> dict[str, Any]:
    return {"chunks": len(result or [])} if isinstance(result, list) else {}


def _truncation_meta(_: tuple[Any, ...], __: dict[str, Any], result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    return {
        "entities": len(result.get("entities_context", [])),
        "relations": len(result.get("relations_context", [])),
    }


def _entity_chunk_meta(args: tuple[Any, ...], __: dict[str, Any], result: Any) -> dict[str, Any]:
    entity_count = len(args[0] or []) if args else 0
    return {
        "entities": entity_count,
        "selected_chunks": len(result or []),
    }


def _relation_chunk_meta(args: tuple[Any, ...], __: dict[str, Any], result: Any) -> dict[str, Any]:
    relation_count = len(args[0] or []) if args else 0
    return {
        "relations": relation_count,
        "selected_chunks": len(result or []),
    }


def _merge_chunks_meta(_: tuple[Any, ...], __: dict[str, Any], result: Any) -> dict[str, Any]:
    return {"merged_chunks": len(result or [])} if isinstance(result, list) else {}


def _vector_similarity_meta(args: tuple[Any, ...], kwargs: dict[str, Any], result: Any) -> dict[str, Any]:
    target_chunks = kwargs.get("num_of_chunks")
    if target_chunks is None and len(args) > 3:
        target_chunks = args[3]
    groups = kwargs.get("entity_info")
    if groups is None and len(args) > 4:
        groups = args[4]
    return {
        "groups": len(groups or []),
        "target_chunks": _safe_int(target_chunks),
        "selected_chunks": len(result or []),
    }


def _chunk_finalize_meta(_: tuple[Any, ...], __: dict[str, Any], result: Any) -> dict[str, Any]:
    return {"final_chunks": len(result or [])} if isinstance(result, list) else {}


@dataclass
class RagRequestTrace:
    request_id: str
    session_id: str
    knowledge_base_id: str
    rag_engine_id: str
    model: str
    stream: bool
    route_reason: str
    started_at: float = field(default_factory=time.perf_counter)
    stage_totals_ms: dict[str, float] = field(default_factory=dict)
    stage_counts: dict[str, int] = field(default_factory=dict)
    counters: dict[str, float] = field(default_factory=dict)
    llm_stream_started_at: float | None = None
    llm_first_chunk_logged: bool = False
    summary_logged: bool = False

    def record_stage(self, stage: str, duration_ms: float, meta: dict[str, Any] | None = None) -> None:
        self.stage_totals_ms[stage] = self.stage_totals_ms.get(stage, 0.0) + duration_ms
        self.stage_counts[stage] = self.stage_counts.get(stage, 0) + 1
        request_logger.info(
            "rag-stage request_id=%s session_id=%s knowledge_base_id=%s rag_engine_id=%s stage=%s call=%s ms=%.2f meta=%s",
            self.request_id,
            self.session_id,
            self.knowledge_base_id,
            self.rag_engine_id,
            stage,
            self.stage_counts[stage],
            duration_ms,
            meta or {},
        )

    def increment_counter(self, key: str, amount: int | float = 1) -> None:
        self.counters[key] = self.counters.get(key, 0.0) + amount

    def mark_llm_stream_open(self) -> None:
        self.llm_stream_started_at = time.perf_counter()

    def mark_llm_stream_first_chunk(self) -> None:
        if self.llm_stream_started_at is None or self.llm_first_chunk_logged:
            return
        self.llm_first_chunk_logged = True
        self.record_stage(
            "llm_first_chunk",
            (time.perf_counter() - self.llm_stream_started_at) * 1000,
        )

    def mark_llm_stream_complete(self) -> None:
        if self.llm_stream_started_at is None:
            return
        self.record_stage(
            "llm_stream",
            (time.perf_counter() - self.llm_stream_started_at) * 1000,
        )
        self.llm_stream_started_at = None

    def export_stage_timings(self) -> dict[str, float]:
        timings: dict[str, float] = {
            stage: round(duration_ms, 2)
            for stage, duration_ms in self.stage_totals_ms.items()
        }
        timings["request_total"] = round((time.perf_counter() - self.started_at) * 1000, 2)
        return timings

    def finalize(
        self,
        timings_sink: MutableMapping[str, float] | None = None,
        error: Exception | None = None,
    ) -> None:
        if self.summary_logged:
            return
        self.summary_logged = True
        stage_ms = {
            stage: round(duration_ms, 2)
            for stage, duration_ms in self.stage_totals_ms.items()
        }
        total_ms = round((time.perf_counter() - self.started_at) * 1000, 2)
        counters = {
            key: int(value) if float(value).is_integer() else round(value, 2)
            for key, value in self.counters.items()
        }
        if error is not None:
            counters["error"] = type(error).__name__
        if timings_sink is not None:
            timings_sink.update(self.export_stage_timings())
        request_logger.info(
            "rag-summary request_id=%s session_id=%s knowledge_base_id=%s rag_engine_id=%s model=%s stream=%s route_reason=%s total_ms=%.2f stage_ms=%s counters=%s",
            self.request_id,
            self.session_id,
            self.knowledge_base_id,
            self.rag_engine_id,
            self.model,
            self.stream,
            self.route_reason,
            total_ms,
            stage_ms,
            counters,
        )


LegacyLightRAGTrace = RagRequestTrace


def get_current_rag_trace() -> RagRequestTrace | None:
    return _current_trace.get()


def get_current_legacy_trace() -> RagRequestTrace | None:
    return get_current_rag_trace()


def start_rag_request_trace(
    request_id: str,
    session_id: str,
    knowledge_base_id: str,
    rag_engine_id: str,
    model: str,
    stream: bool,
    route_reason: str,
) -> tuple[RagRequestTrace, contextvars.Token[RagRequestTrace | None]]:
    trace = RagRequestTrace(
        request_id=request_id,
        session_id=session_id,
        knowledge_base_id=knowledge_base_id,
        rag_engine_id=rag_engine_id,
        model=model,
        stream=stream,
        route_reason=route_reason,
    )
    token = _current_trace.set(trace)
    return trace, token


def start_legacy_lightrag_trace(
    request_id: str,
    session_id: str,
    stream: bool,
) -> tuple[RagRequestTrace, contextvars.Token[RagRequestTrace | None]]:
    return start_rag_request_trace(
        request_id=request_id,
        session_id=session_id,
        knowledge_base_id="unknown-kb",
        rag_engine_id="lightrag",
        model="unknown-model",
        stream=stream,
        route_reason="legacy compatibility trace",
    )


def reset_rag_request_trace(token: contextvars.Token[RagRequestTrace | None]) -> None:
    _current_trace.reset(token)


def reset_legacy_lightrag_trace(token: contextvars.Token[RagRequestTrace | None]) -> None:
    reset_rag_request_trace(token)


def _wrap_async_stage(
    original: Callable[..., Any],
    stage_name: str,
    meta_builder: Callable[[tuple[Any, ...], dict[str, Any], Any], dict[str, Any]] | None = None,
) -> Callable[..., Any]:
    if getattr(original, "_meno_rag_timing_wrapped", False):
        return original

    @wraps(original)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        trace = get_current_rag_trace()
        if trace is None:
            return await original(*args, **kwargs)

        started_at = time.perf_counter()
        result: Any = None
        error: Exception | None = None
        try:
            result = await original(*args, **kwargs)
            return result
        except Exception as exc:  # pragma: no cover - exercised indirectly
            error = exc
            raise
        finally:
            meta: dict[str, Any] = {}
            if meta_builder is not None:
                try:
                    meta = meta_builder(args, kwargs, result)
                except Exception as meta_error:  # pragma: no cover - defensive
                    meta = {"meta_error": str(meta_error)}
            if error is not None:
                meta["error"] = type(error).__name__
            trace.record_stage(stage_name, (time.perf_counter() - started_at) * 1000, meta)

    setattr(wrapped, "_meno_rag_timing_wrapped", True)
    return wrapped


def install_lightrag_timing_hooks() -> None:
    global _hooks_installed
    if _hooks_installed:
        return

    try:
        import lightrag.operate as lightrag_operate  # type: ignore[import-untyped]

        import lightrag.utils as lightrag_utils  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        logger.warning("LightRAG timing hooks were skipped because lightrag is not installed.")
        return

    hook_specs = [
        ("extract_keywords_only", "keywords_extract", _keywords_meta),
        ("_get_node_data", "graph_local_retrieval", _local_query_meta),
        ("_get_edge_data", "graph_global_retrieval", _global_query_meta),
        ("_get_vector_context", "vector_retrieval", _vector_context_meta),
        ("_apply_token_truncation", "context_build", _truncation_meta),
        ("_find_related_text_unit_from_entities", "graph_local_retrieval", _entity_chunk_meta),
        ("_find_related_text_unit_from_relations", "graph_global_retrieval", _relation_chunk_meta),
        ("_merge_all_chunks", "fusion", _merge_chunks_meta),
        ("_build_context_str", "context_build", _build_context_meta),
        ("pick_by_vector_similarity", "vector_retrieval", _vector_similarity_meta),
        ("process_chunks_unified", "context_build", _chunk_finalize_meta),
    ]

    for attr_name, stage_name, meta_builder in hook_specs:
        original = getattr(lightrag_operate, attr_name, None)
        if callable(original):
            setattr(
                lightrag_operate,
                attr_name,
                _wrap_async_stage(original, stage_name, meta_builder),
            )

    rerank_func = getattr(lightrag_utils, "apply_rerank_if_enabled", None)
    if callable(rerank_func):
        setattr(
            lightrag_utils,
            "apply_rerank_if_enabled",
            _wrap_async_stage(
                rerank_func,
                "rerank",
                lambda args, kwargs, result: {
                    "input_chunks": len(kwargs.get("retrieved_docs") or (args[1] if len(args) > 1 else []) or []),
                    "output_chunks": len(result or []),
                    "top_n": _safe_int(kwargs.get("top_n")),
                },
            ),
        )

    _hooks_installed = True
    logger.info("LightRAG timing hooks installed.")


def install_legacy_lightrag_timing_hooks() -> None:
    install_lightrag_timing_hooks()
