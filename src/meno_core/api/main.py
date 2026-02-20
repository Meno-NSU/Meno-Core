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
from datetime import datetime
from logging import Logger, Formatter
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Dict, Optional, Union, AsyncIterator, Any
from typing import Literal, Tuple

import pytz  # type: ignore[import-untyped]
from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore[import-untyped]
from apscheduler.triggers.cron import CronTrigger  # type: ignore[import-untyped]
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from lightrag import QueryParam, LightRAG  # type: ignore[import-untyped]
from lightrag.utils import setup_logger  # type: ignore[import-untyped]
from pydantic import BaseModel

from meno_core.config.settings import settings
from meno_core.core.link_correcter import LinkCorrecter
from meno_core.core.link_searcher import LinkSearcher
from meno_core.core.rag_engine import initialize_rag, SYSTEM_PROMPT_FOR_MENO, QUERY_MAX_TOKENS, TOP_K, resolve_anaphora, \
    explain_abbreviations, get_current_period
from meno_core.infrastructure.logdb.log_collector import LogCollector

QUERY_MODE: Literal["local", "global", "hybrid", "naive", "mix"] = settings.query_mode

LINKS_LOG_PATH: str = getattr(settings, "links_log_path", "logs/links_debug.log")
LINKS_LOG_LEVEL: str = getattr(settings, "links_log_level", "DEBUG")  # DEBUG/INFO/WARNING
LINKS_LOG_MAX_BYTES: int = getattr(settings, "links_log_max_bytes", 10 * 1024 * 1024)
LINKS_LOG_BACKUP_COUNT: int = getattr(settings, "links_log_backup_count", 5)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger: Logger = logging.getLogger(__name__)

try:
    collector: LogCollector = LogCollector()
except Exception as e:
    logger.exception(f"Failed to initialize LogCollector: {str(e)}")
    pass

# user_id -> [{"role": "user"/"assistant", "content": "..."}]
dialogue_histories: Dict[str, List[Dict[str, str]]] = defaultdict(list)

SCORES_CACHE_FILE: str = os.getenv("SCORES_CACHE_FILE", "question_scores.json")

_scores_cache: Dict[str, Dict[str, Optional[str]]] = {}


def setup_links_logger(path: str, level: str = "DEBUG",
                       max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5) -> logging.Logger:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    links_logger: Logger = logging.getLogger("links")
    links_logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))
    # чтобы не дублировать записи при hot-reload
    if not any(isinstance(h, RotatingFileHandler) and getattr(h, 'baseFilename', '') == os.path.abspath(path)
               for h in links_logger.handlers):
        fh: RotatingFileHandler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count,
                                                      encoding="utf-8")
        fmt: Formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(fmt)
        links_logger.addHandler(fh)
    return links_logger


async def clear_rag_cache():
    """Clear LightRAG cache"""
    try:
        current_time: datetime = datetime.now(pytz.timezone("Asia/Novosibirsk"))
        logger.info(
            f"⏰ Clearing cache at: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z%z')}")

        await rag_instance.aclear_cache()
        logger.info("✅ LightRAG cache cleared successfully")
    except Exception as clear_cache_error:
        logger.error(f"❌ Failed to clear cache: {str(clear_cache_error)}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    global rag_instance, abbreviations, ref_searcher, ref_corrector, scheduler
    links_logger: Logger = setup_links_logger(
        LINKS_LOG_PATH, LINKS_LOG_LEVEL, LINKS_LOG_MAX_BYTES, LINKS_LOG_BACKUP_COUNT
    )
    setup_logger("light_rag_log", "DEBUG", False, str(settings.log_file_path))
    rag_instance, embedder, bm25, chunk_db = await initialize_rag()
    # ref_searcher = ReferenceSearcher(URLS_FNAME, model_name=LOCAL_EMBEDDER_NAME, threshold=0.75)
    if settings.enable_links_addition:
        ref_searcher = LinkSearcher(
            settings.urls_path,
            rag_instance,
            top_k=settings.top_k,
            dist_threshold=settings.dist_threshold,  # для совместимости
            max_links=settings.max_links,
            embedder=embedder,
            bm25=bm25,
            chunk_db=chunk_db,
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
        scheduler.add_job(
            clear_rag_cache,
            trigger=CronTrigger(hour=0, minute=0),
            name="clear_rag_cache_daily"
        )
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

    yield  # <-- здесь FastAPI продолжает работу
    # Здесь можно вызвать await rag_instance.cleanup(), если нужно
    # Shutdown scheduler on app exit
    if scheduler is not None:
        scheduler.shutdown()
    logger.info("⏰ Cache-clearing scheduler stopped")


def create_app() -> FastAPI:
    new_app = FastAPI(lifespan=lifespan)
    return new_app


app = create_app()

rag_instance: Optional[LightRAG] = None
abbreviations: Dict[str, str] = {}
ref_searcher: Optional[LinkSearcher] = None
ref_corrector: Optional[LinkCorrecter] = None
scheduler: Optional[AsyncIOScheduler] = None


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


@app.post("/v1/chat/completions")
async def chat_completions(request: OAIChatCompletionsRequest):
    if rag_instance is None:
        raise RuntimeError("RAG is not initialized.")
    created_ts: int = int(time.time())
    completion_id: str = f"chatcmpl-{uuid.uuid4().hex}"
    model_id: str = request.model or "menon-1"
    session_id: str = request.user or f"session-{completion_id}"

    try:
        collector.create_message(session_id=session_id)
    except Exception as creating_message_error:
        logger.exception("Failed to create message", exc_info=creating_message_error)
        pass

    formatted_system_prompt, query, history = await _build_prompt_and_history(request.messages)

    try:
        collector.add_question(session_id=session_id, text=query)
    except Exception as adding_question_error:
        logger.exception("Failed to add question", exc_info=adding_question_error)
        pass

    try:
        expanded_query: str = await explain_abbreviations(query, abbreviations)

    except Exception as explain_error:
        logger.exception("Abbreviation explanation failed", exc_info=explain_error)
        expanded_query = query
    try:
        resolved_query: str = await resolve_anaphora(expanded_query, history)
    except Exception as resolve_error:
        logger.exception("Anaphora resolution failed", exc_info=resolve_error)
        resolved_query = expanded_query

    try:
        collector.add_expanded_question(session_id=session_id, text=expanded_query)
        collector.add_resolved_question(session_id=session_id, text=resolved_query)
    except Exception as resolve_error:
        logger.exception("Failed to add expanded/resolved question", exc_info=resolve_error)
        pass

    collector.update_time(session_id=session_id)

    async def run_lightrag():
        if settings.clear_cache:
            await rag_instance.aclear_cache()
            logger.info("LightRAG cache cleared successfully")
        logger.info(f"Running lightrag with query: ```\n{resolved_query}\n```\nhistory: ```\n{history}\n```")
        ans = await rag_instance.aquery(
            resolved_query,
            param=QueryParam(
                mode=QUERY_MODE,
                top_k=TOP_K,
                chunk_top_k=settings.chunk_top_k,
                max_total_tokens=QUERY_MAX_TOKENS,
                max_entity_tokens=settings.query_max_entity_tokens,
                max_relation_tokens=settings.query_max_relational_tokens,
                history_turns=len(history),
                conversation_history=history,
                stream=request.stream,
            ),
            system_prompt=formatted_system_prompt
        )
        logger.info(f"LightRAG answer: ```\n{ans}\n```")
        return ans

    if not request.stream:
        try:
            result = await run_lightrag()
            if hasattr(result, "__aiter__"):
                logger.info("Found chunks in result")
                chunks = []
                async for part in result:
                    if part:
                        chunks.append(str(part))
                result = "".join(chunks)
            content = str(result)
            logger.info(f"LightRAG answer after reconstruction: ```\n{content}\n```")
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

            try:
                collector.add_model_answer(session_id=session_id, text=content)
                collector.print_dto(session_id=session_id)
            except Exception as err:
                logger.exception("Failed to add model answer", exc_info=err)
                pass

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
            result: str | AsyncIterator[str] = await run_lightrag()

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
                    collector.print_dto(session_id=session_id)
                except Exception as collector_error:
                    logger.exception("Failed to add model answer (stream)", exc_info=collector_error)

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


@app.get("/llm_address")
async def get_ip():
    return {"ip": settings.openai_base_url}


@app.post("/llm_address")
async def set_ip(ip: str):
    old_ip = settings.openai_base_url
    settings.openai_base_url = ip
    return {
        "ip": settings.openai_base_url,
        "old_ip": old_ip
    }


@app.get("/settings")
async def get_settings():
    return {name: str(value) for name, value in settings.__dict__.items()}


@app.post("/settings")
async def set_setting(name: str, value: str):
    old = getattr(settings, name, None)
    setattr(settings, name, value)
    return {
        "name": name,
        "value": value,
        "old": old
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": "menon-1",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "menon",
        }]
    }
