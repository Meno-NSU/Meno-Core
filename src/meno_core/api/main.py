import asyncio
import codecs
import io
import json
import logging
import mimetypes
import os
import random
import re
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from logging import Logger
from pathlib import Path
from typing import List, Dict, Optional, Union, AsyncIterator, Any
from typing import Literal, Tuple

from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore[import-untyped]
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from meno_core.api.pipeline_stages import (
    StageName, PipelineTimer, StageTracker, ThinkingTokenSplitter,
)
from meno_core.config.settings import settings
from meno_core.core.lightrag_engine import LightRAGEngine
from meno_core.core.link_correcter import LinkCorrecter
from meno_core.core.link_searcher import LinkSearcher
from meno_core.core.rag_engine import (
    explain_abbreviations,
    get_current_period,
    initialize_rag,
    llm_request_scope,
    resolve_anaphora,
)
from meno_core.core.rag_runtime import (
    RagBackendRegistry,
    RagChatRequest,
    RagSelectionError,
    ChunkRagChatBackend,
    build_public_rag_backend_registry,
)
from meno_core.core.prompts import SYSTEM_PROMPT_FOR_MENO
from meno_core.core.vllm_registry import VLLMRegistry
from meno_core.infrastructure.logdb.log_collector import LogCollector
from meno_core.api.arena import arena_router

LINKS_LOG_PATH: str = getattr(settings, "links_log_path", "logs/links_debug.log")
LINKS_LOG_LEVEL: str = getattr(settings, "links_log_level", "INFO")  # DEBUG/INFO/WARNING
LINKS_LOG_MAX_BYTES: int = getattr(settings, "links_log_max_bytes", 10 * 1024 * 1024)
LINKS_LOG_BACKUP_COUNT: int = getattr(settings, "links_log_backup_count", 5)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger: Logger = logging.getLogger(__name__)
request_logger: Logger = logging.getLogger("meno_core.request")
collector: Optional[LogCollector] = None
_collector_disabled_reason: Optional[str] = None

try:
    collector = LogCollector()
except Exception as e:
    logger.exception(f"Failed to initialize LogCollector: {str(e)}")
    pass

# user_id -> [{"role": "user"/"assistant", "content": "..."}]
dialogue_histories: Dict[str, List[Dict[str, str]]] = defaultdict(list)

SCORES_CACHE_FILE: str = os.getenv("SCORES_CACHE_FILE", "question_scores.json")

_scores_cache: Dict[str, Dict[str, Optional[str]]] = {}


def _disable_collector(reason: Exception | str) -> None:
    global collector, _collector_disabled_reason
    if _collector_disabled_reason is None:
        _collector_disabled_reason = str(reason)
        logger.warning("Disabling LogCollector due to runtime failure: %s", _collector_disabled_reason)
    collector = None


def setup_links_logger(path: str, level: str = "DEBUG",
                       max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5) -> logging.Logger:
    links_logger: Logger = logging.getLogger("links")
    links_logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))
    links_logger.propagate = True
    return links_logger


@asynccontextmanager
async def lifespan(_: FastAPI):
    global rag_instance, abbreviations, ref_searcher, ref_corrector, scheduler, embedder_instance, bm25_instance
    global chunk_db_instance, vllm_registry, rag_backend_registry
    links_logger: Logger = setup_links_logger(
        LINKS_LOG_PATH, LINKS_LOG_LEVEL, LINKS_LOG_MAX_BYTES, LINKS_LOG_BACKUP_COUNT
    )
    light_rag_logger = logging.getLogger("light_rag_log")
    light_rag_logger.setLevel(logging.WARNING)
    light_rag_logger.handlers.clear()
    light_rag_logger.propagate = True
    rag_instance, embedder_instance, bm25_instance, chunk_db_instance = await initialize_rag()
    if not isinstance(rag_instance, LightRAGEngine):
        raise RuntimeError("Public lightrag backend is unavailable. initialize_rag() must return LightRAGEngine.")

    from meno_core.core.rag.factory import build_chunk_rag_orchestrator
    chunk_rag_orchestrator = await build_chunk_rag_orchestrator(
        working_dir=settings.chunk_rag_data_path,
        embedder=embedder_instance
    )
    lightrag_kb_id = settings.working_dir.name if settings.working_dir else "default-kb"
    rag_backend_registry = build_public_rag_backend_registry(
        lightrag_kb_id=lightrag_kb_id,
        lightrag_backend=rag_instance,
        chunk_rag_backend=ChunkRagChatBackend(chunk_rag_orchestrator),
    )
    logger.info("✅ Public RAG backend registry initialized with %d knowledge base(s).",
                len(rag_backend_registry.list_knowledge_bases()))
    logger.info("All backend logs are routed to stdout/stderr; separate file handlers are disabled.")
    logger.info("Background thread setup logic finished.")
    # ref_searcher = ReferenceSearcher(URLS_FNAME, model_name=LOCAL_EMBEDDER_NAME, threshold=0.75)
    if settings.enable_links_addition:
        ref_searcher = LinkSearcher(
            settings.urls_path,
            rag_instance,
            top_k=settings.top_k,
            dist_threshold=settings.dist_threshold,  # для совместимости
            max_links=settings.max_links,
            embedder=embedder_instance,
            bm25=bm25_instance,
            chunk_db=chunk_db_instance,
            dense_weight=getattr(settings, "dense_weight", 1.0),
            sparse_weight=getattr(settings, "sparse_weight", 0.22),
            hybrid_similarity_threshold=getattr(settings, "hybrid_similarity_threshold", 2.6),
            per_chunk_top_k=getattr(settings, "per_chunk_top_k", 10),
            logger=links_logger,
        )
    if settings.enable_links_correction:
        ref_corrector = LinkCorrecter(settings.urls_path, settings.correct_dist_threshold)
    scheduler = AsyncIOScheduler(timezone="Asia/Novosibirsk")  # timezone
    # Clear cache daily at 00:00
    if scheduler is not None:
        scheduler.start()
        logger.info("⏰ Cache-clearing scheduler started")
    try:
        # ensure path is str for mypy (settings.abbreviations_file may be Path)
        abbr_path: str = str(settings.abbreviations_path)
        with codecs.open(abbr_path, mode='r', encoding='utf-8') as fp:
            abbreviations = json.load(fp)
            logger.info(f"📚 Successfully load {len(abbreviations)} abbreviations.")
    except Exception as load_abbreviations_error:
        logger.exception("Unable to load abbreviations", exc_info=load_abbreviations_error)

    _vllm_endpoint_list = [ep.strip() for ep in settings.vllm_endpoints.split(",") if ep.strip()]
    if _vllm_endpoint_list:
        vllm_registry = VLLMRegistry(_vllm_endpoint_list)
        try:
            models = await vllm_registry.discover()
            logger.info("🔍 Discovered %d model(s) across %d vLLM endpoint(s)",
                        len(models), len(_vllm_endpoint_list))
            if models:
                available_ids = [m["id"] for m in models]
                # Verify the configured model name actually exists on the endpoint.
                # If it is missing (or not set at all), fall back to the first real model.
                if settings.llm_model_name not in available_ids:
                    resolved = available_ids[0]
                    logger.warning(
                        "⚠️  LLM_MODEL_NAME='%s' is not available on vLLM "
                        "(available: %s). Overriding with '%s'.",
                        settings.llm_model_name, available_ids, resolved,
                    )
                    settings.llm_model_name = resolved
                else:
                    logger.info("✅ Configured model '%s' confirmed on vLLM.",
                                settings.llm_model_name)
        except Exception as vllm_err:
            logger.warning("vLLM discovery failed at startup: %s", vllm_err)
    else:
        logger.info("No VLLM_ENDPOINTS configured — model list will use fallback")

    yield  # <-- здесь FastAPI продолжает работу
    # Здесь можно вызвать await rag_instance.cleanup(), если нужно
    # Shutdown scheduler on app exit
    if scheduler is not None:
        scheduler.shutdown()
    logger.info("⏰ Cache-clearing scheduler stopped")


def create_app() -> FastAPI:
    new_app = FastAPI(lifespan=lifespan)
    new_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return new_app


app = create_app()
app.include_router(arena_router)

rag_instance: Optional[LightRAGEngine] = None
embedder_instance = None
bm25_instance = None
chunk_db_instance = None
abbreviations: Dict[str, str] = {}
ref_searcher: Optional[LinkSearcher] = None
ref_corrector: Optional[LinkCorrecter] = None
scheduler: Optional[AsyncIOScheduler] = None
vllm_registry: Optional[VLLMRegistry] = None
rag_backend_registry: Optional[RagBackendRegistry] = None


class ResetRequest(BaseModel):
    chat_id: str


class ResetResponse(BaseModel):
    chat_id: str
    status: str


class OAIMsg(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LinkOnlyResponse(BaseModel):
    url: str


class LinkOnlyRequest(BaseModel):
    messages: List[OAIMsg]


class ImageOnlyRequest(BaseModel):
    messages: List[OAIMsg]


class OAIChatCompletionsRequest(BaseModel):
    model: Optional[str] = None
    messages: List[OAIMsg]
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    user: Optional[str] = None
    knowledge_base: Optional[str] = None
    knowledge_base_id: Optional[str] = None
    rag_engine: Optional[str] = None
    rag_engine_id: Optional[str] = None

    model_config = {
        "extra": "allow"
    }


async def _build_prompt_and_history(messages: List[OAIMsg]) -> Tuple[str, str, List[Dict[str, str]]]:
    """Получаем system_prompt, последнюю user-реплику (query) и последние 4 раунда истории."""
    # system prompt
    sys_msgs = [m.content for m in messages if m.role == "system"]
    if sys_msgs:
        formatted_system_prompt = "\n".join(sys_msgs)
    else:
        current_date_str = await get_current_period()
        formatted_system_prompt = SYSTEM_PROMPT_FOR_MENO.replace("{current_date}", current_date_str)

    # query + history
    last_user_idx = max(i for i, m in enumerate(messages) if m.role == "user")
    query = messages[last_user_idx].content.strip()

    raw_hist = messages[:last_user_idx]
    history = [{"role": m.role, "content": m.content}
               for m in raw_hist if m.role in ("user", "assistant")][-4:]
    return formatted_system_prompt, query, history


_JSON_BLOCK_RE = re.compile(r"```json\s*([\s\S]*?)\s*```", re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*}", re.MULTILINE)


def _log_request_summary(
    *,
    request_id: str,
    session_id: str,
    knowledge_base_id: str,
    rag_engine_id: str,
    model: str,
    route_reason: str,
    started_at: float,
    stage_ms: dict[str, float],
    response_len: int,
    stream: bool,
) -> None:
    request_logger.info(
        "request-summary request_id=%s session_id=%s knowledge_base_id=%s rag_engine_id=%s model=%s route_reason=%s total_ms=%.2f stage_ms=%s response_len=%s stream=%s",
        request_id,
        session_id,
        knowledge_base_id,
        rag_engine_id,
        model,
        route_reason,
        (time.perf_counter() - started_at) * 1000,
        stage_ms,
        response_len,
        stream,
    )


def _default_model_name() -> str:
    return settings.llm_model_name or "menon-1"


def _build_model_not_found_response(model_id: str, available_models: list[str]) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": (
                    f"Model '{model_id}' is not available on this vLLM endpoint. "
                    f"Available models: {available_models}"
                ),
                "type": "invalid_request_error",
                "code": "model_not_found",
                "param": "model",
            }
        },
    )


async def _resolve_requested_model(requested_model: Optional[str]) -> tuple[str, Optional[str], Optional[JSONResponse]]:
    normalized_model = requested_model.strip() if isinstance(requested_model, str) and requested_model.strip() else None
    default_model = _default_model_name()

    if vllm_registry is None:
        return normalized_model or default_model, None, None

    models = await vllm_registry.list_models()
    available_models = [m["id"] for m in models]

    if not available_models:
        model_id = normalized_model or default_model
        if normalized_model and model_id != default_model:
            logger.warning(
                "Rejected request for unknown model '%s'. Available: %s",
                model_id,
                available_models,
            )
            return model_id, None, _build_model_not_found_response(model_id, available_models)

        logger.warning(
            "vLLM registry returned no models; falling back to configured default model '%s'.",
            model_id,
        )
        return model_id, None, None

    if default_model not in available_models:
        logger.warning(
            "Configured default model '%s' is unavailable in the registry; falling back to '%s'.",
            default_model,
            available_models[0],
        )
        default_model = available_models[0]

    model_id = normalized_model or default_model
    if model_id not in available_models:
        logger.warning(
            "Rejected request for unknown model '%s'. Available: %s",
            model_id,
            available_models,
        )
        return model_id, None, _build_model_not_found_response(model_id, available_models)

    endpoint = vllm_registry.lookup_endpoint(model_id)
    return model_id, f"{endpoint}/v1" if endpoint else None, None


@app.post("/v1/chat/completions")
async def chat_completions(request_body: OAIChatCompletionsRequest, request: Request):
    if rag_backend_registry is None:
        raise RuntimeError("RAG is not initialized.")
    created_ts: int = int(time.time())
    completion_id: str = f"chatcmpl-{uuid.uuid4().hex}"
    request_started_at = time.perf_counter()
    request_timings_ms: Dict[str, float] = {}
    selected_knowledge_base_id = "unknown"
    selected_rag_engine_id = "unknown"
    route_reason = "unknown"
    emit_stages: bool = request.headers.get("x-pipeline-stages", "").lower() == "true"
    model_id, explicit_base_url, model_error = await _resolve_requested_model(request_body.model)
    if model_error is not None:
        return model_error
    session_id: str = request_body.user or f"session-{completion_id}"

    if collector is not None:
        try:
            collector.create_message(session_id=session_id)
        except Exception as creating_message_error:
            _disable_collector(creating_message_error)

    prompt_started_at = time.perf_counter()
    formatted_system_prompt, query, history = await _build_prompt_and_history(request_body.messages)
    request_timings_ms["prompt_build"] = round((time.perf_counter() - prompt_started_at) * 1000, 2)

    if collector is not None:
        try:
            collector.add_question(session_id=session_id, text=query)
        except Exception as adding_question_error:
            _disable_collector(adding_question_error)

    # --- Helper: run preprocessing with stage tracking ---
    async def _run_preprocessing(timer: PipelineTimer) -> tuple[str, str]:
        async with StageTracker(StageName.ABBREVIATION_EXPANSION, logger) as tracker:
            try:
                expanded = await explain_abbreviations(query, abbreviations, override_model=model_id, override_base_url=explicit_base_url)
                tracker.detail = {"original": query, "expanded": expanded}
            except Exception as explain_error:
                logger.exception("Abbreviation explanation failed", exc_info=explain_error)
                expanded = query
                tracker.detail = {"original": query, "expanded": query, "fallback": True}
        timer.record(tracker)
        request_timings_ms["expand"] = tracker.duration_ms

        async with StageTracker(StageName.ANAPHORA_RESOLUTION, logger) as tracker:
            try:
                resolved = await resolve_anaphora(expanded, history, override_model=model_id, override_base_url=explicit_base_url)
                tracker.detail = {"resolved_query": resolved}
            except Exception as resolve_error:
                logger.exception("Anaphora resolution failed", exc_info=resolve_error)
                resolved = expanded
                tracker.detail = {"resolved_query": expanded, "fallback": True}
        timer.record(tracker)
        request_timings_ms["resolve"] = tracker.duration_ms
        request_timings_ms["request_prepare"] = round(
            request_timings_ms["prompt_build"] + request_timings_ms["expand"] + request_timings_ms["resolve"],
            2,
        )

        if collector is not None:
            try:
                collector.add_expanded_question(session_id=session_id, text=expanded)
                collector.add_resolved_question(session_id=session_id, text=resolved)
            except Exception as collector_update_error:
                _disable_collector(collector_update_error)

        if collector is not None:
            try:
                collector.update_time(session_id=session_id)
            except Exception as collector_time_error:
                _disable_collector(collector_time_error)

        return expanded, resolved

    # --- Helper: resolve RAG backend ---
    def _resolve_backend():
        extra_payload = request_body.model_extra or {}
        requested_kb = request_body.knowledge_base_id or request_body.knowledge_base or extra_payload.get("kb_id")
        requested_engine = (
            request_body.rag_engine_id
            or request_body.rag_engine
            or extra_payload.get("engine_id")
            or extra_payload.get("rag_engine")
        )
        return rag_backend_registry.resolve(requested_kb, requested_engine)

    # --- Non-streaming path ---
    if not request_body.stream:
        timer = PipelineTimer()

        try:
            _expanded_query, resolved_query = await _run_preprocessing(timer)

            dispatch_started_at = time.perf_counter()
            try:
                backend_entry, route_reason = _resolve_backend()
            except RagSelectionError as selection_error:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "message": str(selection_error),
                            "type": "invalid_request_error",
                            "param": "knowledge_base_id/rag_engine_id",
                        }
                    },
                )
            request_timings_ms["dispatch"] = round((time.perf_counter() - dispatch_started_at) * 1000, 2)
            selected_knowledge_base_id = backend_entry.knowledge_base_id
            selected_rag_engine_id = backend_entry.rag_engine_id
            logger.info(
                "Routing request_id=%s session_id=%s knowledge_base_id=%s rag_engine_id=%s (%s).",
                completion_id, session_id, selected_knowledge_base_id, selected_rag_engine_id, route_reason,
            )

            backend_request = RagChatRequest(
                question=resolved_query,
                history=history,
                system_prompt=formatted_system_prompt,
                stream=False,
                session_id=session_id,
                request_id=completion_id,
                model=model_id,
                knowledge_base_id=selected_knowledge_base_id,
                rag_engine_id=selected_rag_engine_id,
                route_reason=route_reason,
                base_url=explicit_base_url,
            )

            async with StageTracker(StageName.RETRIEVAL_AND_GENERATION, logger) as rag_tracker:
                with llm_request_scope(model_id, explicit_base_url):
                    response = await backend_entry.backend.answer(
                        backend_request,
                        timings_sink=request_timings_ms,
                    )
                if hasattr(response, "__aiter__"):
                    chunks = []
                    async for part in response:
                        if part:
                            chunks.append(str(part))
                    response = "".join(chunks)
                content = str(response)
            timer.record(rag_tracker)

            # Link addition
            if settings.enable_links_addition and ref_searcher is not None:
                async with StageTracker(StageName.LINK_ADDITION, logger) as link_tracker:
                    links = await ref_searcher.get_links_from_answer(content)
                    link_tracker.detail = {"links_found": len(links)}
                timer.record(link_tracker)

            # Link correction
            if settings.enable_links_correction and ref_corrector is not None:
                async with StageTracker(StageName.LINK_CORRECTION, logger) as corr_tracker:
                    content = await ref_corrector.replace_markdown_links(content)
                timer.record(corr_tracker)

            if collector is not None:
                try:
                    collector.add_model_answer(session_id=session_id, text=content)
                except Exception as collector_answer_error:
                    _disable_collector(collector_answer_error)

            summary = timer.summary()
            _log_request_summary(
                request_id=completion_id,
                session_id=session_id,
                knowledge_base_id=selected_knowledge_base_id,
                rag_engine_id=selected_rag_engine_id,
                model=model_id,
                route_reason=route_reason,
                started_at=request_started_at,
                stage_ms=request_timings_ms,
                response_len=len(content),
                stream=False,
            )

        except Exception as non_stream_error:
            logger.exception("chat.completions non-stream error")
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": str(non_stream_error),
                        "type": "server_error",
                    }
                }
            )

        resp: dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created_ts,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
        }
        if emit_stages:
            summary = timer.summary()
            resp["_pipeline_stages"] = {"total_ms": summary.total_ms, "stages": summary.stages}
        return resp

    # --- Streaming path ---
    async def sse_generator():
        nonlocal selected_knowledge_base_id, selected_rag_engine_id, route_reason
        timer = PipelineTimer()

        def mk_chunk(delta: dict, finish_reason=None):
            return {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": finish_reason,
                        "logprobs": None,
                    }
                ],
            }

        def stage_sse(tracker: StageTracker, which: str) -> str:
            evt = tracker.start_event if which == "start" else tracker.end_event
            return evt.to_sse() if evt else ""

        try:
            # --- Preprocessing stages (inside generator for real-time SSE) ---
            abbr_tracker = StageTracker(StageName.ABBREVIATION_EXPANSION, logger)
            async with abbr_tracker:
                try:
                    expanded_query = await explain_abbreviations(query, abbreviations, override_model=model_id, override_base_url=explicit_base_url)
                    abbr_tracker.detail = {"original": query, "expanded": expanded_query}
                except Exception:
                    logger.exception("Abbreviation explanation failed")
                    expanded_query = query
                    abbr_tracker.detail = {"original": query, "expanded": query, "fallback": True}
            timer.record(abbr_tracker)
            request_timings_ms["expand"] = abbr_tracker.duration_ms
            if emit_stages:
                yield stage_sse(abbr_tracker, "start")
                yield stage_sse(abbr_tracker, "end")

            ana_tracker = StageTracker(StageName.ANAPHORA_RESOLUTION, logger)
            async with ana_tracker:
                try:
                    resolved_query = await resolve_anaphora(expanded_query, history, override_model=model_id, override_base_url=explicit_base_url)
                    ana_tracker.detail = {"resolved_query": resolved_query}
                except Exception:
                    logger.exception("Anaphora resolution failed")
                    resolved_query = expanded_query
                    ana_tracker.detail = {"resolved_query": expanded_query, "fallback": True}
            timer.record(ana_tracker)
            request_timings_ms["resolve"] = ana_tracker.duration_ms
            request_timings_ms["request_prepare"] = round(
                request_timings_ms["prompt_build"] + request_timings_ms["expand"] + request_timings_ms["resolve"],
                2,
            )
            if emit_stages:
                yield stage_sse(ana_tracker, "start")
                yield stage_sse(ana_tracker, "end")

            if collector is not None:
                try:
                    collector.add_expanded_question(session_id=session_id, text=expanded_query)
                    collector.add_resolved_question(session_id=session_id, text=resolved_query)
                except Exception:
                    logger.exception("Failed to add expanded/resolved question")
            if collector is not None:
                try:
                    collector.update_time(session_id=session_id)
                except Exception:
                    logger.exception("Failed to update time")

            # --- Backend dispatch ---
            dispatch_started_at = time.perf_counter()
            backend_entry, route_reason = _resolve_backend()
            request_timings_ms["dispatch"] = round((time.perf_counter() - dispatch_started_at) * 1000, 2)
            selected_knowledge_base_id = backend_entry.knowledge_base_id
            selected_rag_engine_id = backend_entry.rag_engine_id
            logger.info(
                "Routing request_id=%s session_id=%s knowledge_base_id=%s rag_engine_id=%s (%s).",
                completion_id, session_id, selected_knowledge_base_id, selected_rag_engine_id, route_reason,
            )

            backend_request = RagChatRequest(
                question=resolved_query,
                history=history,
                system_prompt=formatted_system_prompt,
                stream=True,
                session_id=session_id,
                request_id=completion_id,
                model=model_id,
                knowledge_base_id=selected_knowledge_base_id,
                rag_engine_id=selected_rag_engine_id,
                route_reason=route_reason,
                base_url=explicit_base_url,
            )

            # --- RAG query stage ---
            rag_tracker = StageTracker(StageName.RETRIEVAL_AND_GENERATION, logger)
            await rag_tracker.__aenter__()
            if emit_stages:
                yield stage_sse(rag_tracker, "start")

            # Role chunk
            first = mk_chunk({"role": "assistant"})
            yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"

            accumulated: list[str] = []
            with llm_request_scope(model_id, explicit_base_url):
                result: str | AsyncIterator[str] = await backend_entry.backend.answer(
                    backend_request,
                    timings_sink=request_timings_ms,
                )

                async def iter_pieces():
                    if hasattr(result, "__aiter__"):
                        async for part in result:
                            if part:
                                yield str(part)
                    else:
                        text: str = str(result)
                        step: int = 512
                        for i in range(0, len(text), step):
                            yield text[i:i + step]

                splitter = ThinkingTokenSplitter()
                async for piece in iter_pieces():
                    for segment_text, is_thinking in splitter.feed(piece):
                        if is_thinking:
                            if emit_stages:
                                yield f"event: thinking\ndata: {json.dumps({'id': completion_id, 'content': segment_text}, ensure_ascii=False)}\n\n"
                        else:
                            accumulated.append(segment_text)
                            data = mk_chunk({"content": segment_text})
                            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

                # Flush remaining buffer
                for segment_text, is_thinking in splitter.flush():
                    if is_thinking:
                        if emit_stages:
                            yield f"event: thinking\ndata: {json.dumps({'id': completion_id, 'content': segment_text}, ensure_ascii=False)}\n\n"
                    else:
                        accumulated.append(segment_text)
                        data = mk_chunk({"content": segment_text})
                        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

            await rag_tracker.__aexit__(None, None, None)
            timer.record(rag_tracker)
            if emit_stages:
                yield stage_sse(rag_tracker, "end")

            full_answer = "".join(accumulated)

            # --- Link addition stage ---
            if settings.enable_links_addition and ref_searcher is not None:
                link_tracker = StageTracker(StageName.LINK_ADDITION, logger)
                async with link_tracker:
                    links = await ref_searcher.get_links_from_answer(full_answer)
                    link_tracker.detail = {"links_found": len(links)}
                timer.record(link_tracker)
                if emit_stages:
                    yield stage_sse(link_tracker, "start")
                    yield stage_sse(link_tracker, "end")

            # --- Link correction stage ---
            if settings.enable_links_correction and ref_corrector is not None:
                corr_tracker = StageTracker(StageName.LINK_CORRECTION, logger)
                async with corr_tracker:
                    corrected = await ref_corrector.replace_markdown_links(full_answer)
                    if corrected != full_answer:
                        corr_tracker.detail = {"corrected": True}
                timer.record(corr_tracker)
                if emit_stages:
                    yield stage_sse(corr_tracker, "start")
                    yield stage_sse(corr_tracker, "end")

            # Done chunk
            done = mk_chunk({}, finish_reason="stop")
            yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"

            # Summary
            summary = timer.summary()
            logger.info("Pipeline latency: total=%.1fms stages=%s", summary.total_ms, summary.stages)
            if emit_stages:
                yield summary.to_sse()

            yield "data: [DONE]\n\n"

            if collector is not None:
                try:
                    collector.add_model_answer(session_id=session_id, text=full_answer)
                except Exception as collector_error:
                    _disable_collector(collector_error)

            _log_request_summary(
                request_id=completion_id,
                session_id=session_id,
                knowledge_base_id=selected_knowledge_base_id,
                rag_engine_id=selected_rag_engine_id,
                model=model_id,
                route_reason=route_reason,
                started_at=request_started_at,
                stage_ms=request_timings_ms,
                response_len=len(full_answer),
                stream=True,
            )

        except Exception as stream_error:
            logger.exception("chat.completions stream error")
            err_done = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "error",
                        "logprobs": None,
                    }
                ],
                "error": {"message": str(stream_error), "type": "server_error"},
            }
            yield f"data: {json.dumps(err_done, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/v1/chat/completions/clear_history", response_model=ResetResponse)
async def reset_history(request: ResetRequest):
    chat_id = request.chat_id
    cleared = False

    if chat_id in dialogue_histories:
        dialogue_histories.pop(chat_id)
        cleared = True
        logger.info(f"dialogue_histories очищена для пользователя {chat_id}")

    try:
        if collector is not None and hasattr(collector, '_unreleased_dtos'):
            if chat_id in collector._unreleased_dtos:
                collector._unreleased_dtos.pop(chat_id)
                cleared = True
                logger.info(f"История LogCollector очищена для пользователя {chat_id}")
    except Exception as clear_collector_error:
        logger.exception(f"Ошибка при очистке истории LogCollector для {chat_id}", exc_info=clear_collector_error)

    if cleared:
        logger.info(f"✅ История успешно очищена для пользователя {chat_id}")
    else:
        logger.info(f"ℹ️ История для пользователя {chat_id} не найдена или уже была очищена")

    return ResetResponse(chat_id=chat_id, status="ok")


@app.post("/v1/link_from_text", response_model=LinkOnlyResponse)
async def link_from_text(req: LinkOnlyRequest):
    return LinkOnlyResponse(url="https://nsu.ru/")


ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


async def _build_image_response(path: Path) -> StreamingResponse:
    """Читает картинку и строит StreamingResponse."""
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        data: bytes = await asyncio.to_thread(path.read_bytes)
    except Exception as image_reading_error:
        logger.exception(f"Failed to read image: {path}")
        raise image_reading_error

    ctype, _ = mimetypes.guess_type(str(path))
    if not ctype:
        ctype = "application/octet-stream"

    filename = path.name or f"image-{uuid.uuid4().hex}"
    return StreamingResponse(
        io.BytesIO(data),
        media_type=ctype,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@app.post("/v1/image_from_text")
async def image_from_text(req: ImageOnlyRequest):
    """
    Временная заглушка: игнорируем текст запроса и возвращаем байты локального изображения.
    """
    images_dir = Path("../../../resources/images")
    stub_path = images_dir / "1.jpg"

    random_image_path: Path | None = None

    try:
        if images_dir.exists() and images_dir.is_dir():
            image_files = [
                p for p in images_dir.iterdir()
                if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
            ]
            if image_files:
                random_image_path = random.choice(image_files)
            else:
                logger.warning(f"No image files found in {images_dir.resolve()}")
        else:
            logger.warning(f"Images directory not found or is not a dir: {images_dir.resolve()}")
    except Exception as image_reading_error:
        logger.exception(f"Failed to read image: {images_dir.resolve()}", exc_info=image_reading_error)
    if random_image_path is not None:
        try:
            return await _build_image_response(random_image_path)
        except Exception as image_reading_error:
            logger.exception(f"Failed to read image: {images_dir.resolve()}", exc_info=image_reading_error)

    try:
        return await _build_image_response(stub_path)
    except FileNotFoundError:
        logger.error(f"Stub image not found: {stub_path.resolve()}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "stub_image_not_found",
                "path": str(stub_path),
            },
        )
    except Exception as stub_reading_error:
        logger.exception("Failed to read stub image")
        return JSONResponse(
            status_code=500,
            content={
                "error": "read_failed",
                "message": str(stub_reading_error),
                "path": str(stub_path),
            },
        )


@app.get("/v1/models")
async def list_models():
    if vllm_registry is not None:
        models = await vllm_registry.list_models()
        if models:
            return {"object": "list", "data": models}
    # Fallback: return the single configured model
    model_name = _default_model_name()
    return {
        "object": "list",
        "data": [{
            "id": model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "menon",
        }]
    }


@app.post("/v1/models/refresh")
async def refresh_models():
    if vllm_registry is None:
        return JSONResponse(
            status_code=400,
            content={"error": "No VLLM_ENDPOINTS configured"},
        )
    models = await vllm_registry.refresh()
    logger.info("🔄 Models refreshed: %d model(s) found", len(models))
    return {"object": "list", "data": models}


@app.get("/v1/knowledge-bases")
async def list_knowledge_bases():
    if rag_backend_registry is None:
        raise RuntimeError("RAG registry is not initialized.")
    return {
        "object": "list",
        "data": rag_backend_registry.list_knowledge_bases(),
        "default_selection": rag_backend_registry.default_payload(),
    }
