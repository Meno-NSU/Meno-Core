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
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from meno_core.config.settings import settings
from meno_core.core.lightrag_engine import LightRAGEngine
from meno_core.core.link_correcter import LinkCorrecter
from meno_core.core.link_searcher import LinkSearcher
from meno_core.core.rag_engine import initialize_rag, resolve_anaphora, explain_abbreviations, \
    get_current_period, _current_model_override
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
    model: str
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


@app.post("/v1/chat/completions")
async def chat_completions(request: OAIChatCompletionsRequest):
    if rag_backend_registry is None:
        raise RuntimeError("RAG is not initialized.")
    created_ts: int = int(time.time())
    completion_id: str = f"chatcmpl-{uuid.uuid4().hex}"
    model_id: str = request.model or "menon-1"
    request_started_at = time.perf_counter()
    request_timings_ms: Dict[str, float] = {}
    selected_knowledge_base_id = "unknown"
    selected_rag_engine_id = "unknown"
    route_reason = "unknown"

    if vllm_registry is not None:
        if not await vllm_registry.is_valid_model(model_id):
            available_models = [m["id"] for m in await vllm_registry.list_models()]
            logger.warning(
                "Rejected request for unknown model '%s'. Available: %s",
                model_id, available_models,
            )
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
        
        endpoint = vllm_registry.lookup_endpoint(model_id)
        if endpoint:
            from meno_core.core.rag_engine import _current_base_url_override
            _current_base_url_override.set(f"{endpoint}/v1")

    # Set the model override so rag_engine LLM calls use the UI-selected model
    _current_model_override.set(model_id)
    session_id: str = request.user or f"session-{completion_id}"

    if collector is not None:
        try:
            collector.create_message(session_id=session_id)
        except Exception as creating_message_error:
            _disable_collector(creating_message_error)

    prompt_started_at = time.perf_counter()
    formatted_system_prompt, query, history = await _build_prompt_and_history(request.messages)
    request_timings_ms["prompt_build"] = round((time.perf_counter() - prompt_started_at) * 1000, 2)

    if collector is not None:
        try:
            collector.add_question(session_id=session_id, text=query)
        except Exception as adding_question_error:
            _disable_collector(adding_question_error)

    expand_started_at = time.perf_counter()
    try:
        expanded_query: str = await explain_abbreviations(query, abbreviations)

    except Exception as explain_error:
        logger.exception("Abbreviation explanation failed", exc_info=explain_error)
        expanded_query = query
    request_timings_ms["expand"] = round((time.perf_counter() - expand_started_at) * 1000, 2)

    resolve_started_at = time.perf_counter()
    try:
        resolved_query: str = await resolve_anaphora(expanded_query, history)
    except Exception as resolve_error:
        logger.exception("Anaphora resolution failed", exc_info=resolve_error)
        resolved_query = expanded_query
    request_timings_ms["resolve"] = round((time.perf_counter() - resolve_started_at) * 1000, 2)
    request_timings_ms["request_prepare"] = round(
        request_timings_ms["prompt_build"] + request_timings_ms["expand"] + request_timings_ms["resolve"],
        2,
    )

    if collector is not None:
        try:
            collector.add_expanded_question(session_id=session_id, text=expanded_query)
            collector.add_resolved_question(session_id=session_id, text=resolved_query)
        except Exception as collector_update_error:
            _disable_collector(collector_update_error)

    if collector is not None:
        try:
            collector.update_time(session_id=session_id)
        except Exception as collector_time_error:
            _disable_collector(collector_time_error)

    extra_payload = request.model_extra or {}
    requested_kb = request.knowledge_base_id or request.knowledge_base or extra_payload.get("kb_id")
    requested_engine = (
        request.rag_engine_id
        or request.rag_engine
        or extra_payload.get("engine_id")
        or extra_payload.get("rag_engine")
    )
    dispatch_started_at = time.perf_counter()
    try:
        backend_entry, route_reason = rag_backend_registry.resolve(requested_kb, requested_engine)
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
        completion_id,
        session_id,
        selected_knowledge_base_id,
        selected_rag_engine_id,
        route_reason,
    )

    backend_request = RagChatRequest(
        question=resolved_query,
        history=history,
        system_prompt=formatted_system_prompt,
        stream=request.stream,
        session_id=session_id,
        request_id=completion_id,
        model=model_id,
        knowledge_base_id=selected_knowledge_base_id,
        rag_engine_id=selected_rag_engine_id,
        route_reason=route_reason,
    )

    async def run_rag_backend():
        try:
            return await backend_entry.backend.answer(
                backend_request,
                timings_sink=request_timings_ms,
            )
        except Exception as e:
            logger.error(f"Error querying RAG backend: {e}", exc_info=True)
            raise

    if not request.stream:
        try:
            response = await run_rag_backend()  # type: ignore[misc]
            if hasattr(response, "__aiter__"):
                chunks = []
                async for part in response:
                    if part:
                        chunks.append(str(part))
                response = "".join(chunks)
            content = str(response)
            # prompt_for_first_answer = await rag_instance.aquery(
            #     resolved_query,
            #     param=QueryParam(
            #         mode=QUERY_MODE,
            #         conversation_history=history,
            #         enable_rerank=True,
            #         only_need_prompt=True
            #     ),
            # )
            # try:
            #     is_hallucination, relevance_score = await is_likely_hallucination(
            #         original_prompt=prompt_for_first_answer,
            #         llm_answer=content,
            #     )
            #     logger.info(
            #         "Hallucination result: is_hallucination=%s, score=%.4f",
            #         is_hallucination,
            #         relevance_score,
            #     )
            #     if is_hallucination:
            #         content = "Спасибо за сложный вопрос! Кажется, я не очень уверен в ответе, поэтому заранее приношу извинения за неточности и возможные ошибки!\n\n" + content
            # except Exception:
            #     logger.exception("Hallucination scoring failed")
            #     pass

            if collector is not None:
                try:
                    collector.add_model_answer(session_id=session_id, text=content)
                except Exception as collector_answer_error:
                    _disable_collector(collector_answer_error)

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
                stream=request.stream,
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

        return {
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

    async def sse_generator():
        def chunk(delta: dict, finish_reason=None):
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

        first = chunk({"role": "assistant"})
        yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"

        accumulated: list[str] = []

        try:
            result: str | AsyncIterator[str] = await run_rag_backend()

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

            async for piece in iter_pieces():
                accumulated.append(piece)
                data: dict[str, str | int | list[dict[str, int | dict | None | Any]]] = chunk({"content": piece})
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

            done: dict[str, str | int | list[dict[str, int | dict | None | Any]]] = chunk({}, finish_reason="stop")
            yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            full_answer = "".join(accumulated)
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
                stream=request.stream,
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
    model_name = settings.llm_model_name or "menon-1"
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
