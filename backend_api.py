import asyncio
import codecs
import io
import json
import logging
import mimetypes
import os
import re
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Dict, Optional, Union
from typing import Literal

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from config import settings
from lightrag import QueryParam, LightRAG
from lightrag.utils import setup_logger
from link_correcter import LinkCorrecter
from link_searcher import LinkSearcher
from rag_engine import initialize_rag, SYSTEM_PROMPT_FOR_MENO, QUERY_MAX_TOKENS, TOP_K, resolve_anaphora, \
    explain_abbreviations, get_current_period

from logdb.log_collector import LogCollector

QUERY_MODE: Literal["local", "global", "hybrid", "naive", "mix"] = settings.query_mode

LINKS_LOG_PATH = getattr(settings, "links_log_path", "logs/links_debug.log")
LINKS_LOG_LEVEL = getattr(settings, "links_log_level", "DEBUG")  # DEBUG/INFO/WARNING
LINKS_LOG_MAX_BYTES = getattr(settings, "links_log_max_bytes", 10 * 1024 * 1024)
LINKS_LOG_BACKUP_COUNT = getattr(settings, "links_log_backup_count", 5)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

collector = LogCollector()

# user_id -> [{"role": "user"/"assistant", "content": "..."}]
dialogue_histories: Dict[str, List[Dict[str, str]]] = defaultdict(list)

SCORES_CACHE_FILE = os.getenv("SCORES_CACHE_FILE", "question_scores.json")

_scores_cache: Dict[str, Dict[str, Optional[str]]] = {}


def setup_links_logger(path: str, level: str = "DEBUG",
                       max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5) -> logging.Logger:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger = logging.getLogger("links")
    logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))
    # чтобы не дублировать записи при hot-reload
    if not any(isinstance(h, RotatingFileHandler) and getattr(h, 'baseFilename', '') == os.path.abspath(path)
               for h in logger.handlers):
        fh = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
        fmt = logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


async def clear_rag_cache():
    """Clear LightRAG cache"""
    try:
        current_time = datetime.now(pytz.timezone("Asia/Novosibirsk"))
        logger.info(
            f"⏰ Clearing cache at: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z%z')}")

        await rag_instance.aclear_cache()
        logger.info("✅ LightRAG cache cleared successfully")
    except Exception as e:
        logger.error(f"❌ Failed to clear cache: {str(e)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_instance, abbreviations, ref_searcher, ref_corrector, scheduler
    links_logger = setup_links_logger(
        LINKS_LOG_PATH, LINKS_LOG_LEVEL, LINKS_LOG_MAX_BYTES, LINKS_LOG_BACKUP_COUNT
    )
    setup_logger("light_rag_log", "WARNING", False, str(settings.log_file_path))
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
    scheduler.add_job(
        clear_rag_cache,
        trigger=CronTrigger(hour=0, minute=0),
        name="clear_rag_cache_daily"
    )
    scheduler.start()
    logger.info("⏰ Cache-clearing scheduler started")
    try:
        with codecs.open(settings.abbreviations_file, mode='r', encoding='utf-8') as fp:
            abbreviations = json.load(fp)
            logger.info(f"📚 Successfully load {len(abbreviations)} abbreviations.")
    except Exception as e:
        logger.exception("Unable to load abbreviations", exc_info=e)

    yield  # <-- здесь FastAPI продолжает работу
    # Здесь можно вызвать await rag_instance.cleanup(), если нужно
    # Shutdown scheduler on app exit
    scheduler.shutdown()
    logger.info("⏰ Cache-clearing scheduler stopped")


app = FastAPI(lifespan=lifespan)
rag_instance: LightRAG
abbreviations = {}


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


async def _build_prompt_and_history(messages: List[OAIMsg]) -> tuple[str, str, List[dict]]:
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


# def _try_extract_response(text: str) -> tuple[bool, str]:
#     """
#     Пытается достать поле 'response' из текстового ответа:
#     - ищем ```json ...``` и парсим объект;
#     - если нет, ищем первый {...} и парсим;
#     Возвращает (found, value). Если не нашли корректный JSON — value=исходный text.
#     """
#     if not isinstance(text, str):
#         text = "" if text is None else str(text)
#
#     payload = None
#     m = _JSON_BLOCK_RE.search(text)
#     if m:
#         payload = m.group(1).strip()
#     else:
#         m2 = _JSON_OBJECT_RE.search(text)
#         if m2:
#             payload = m2.group(0).strip()
#
#     if payload:
#         try:
#             obj = json.loads(payload)
#             resp = (
#                     obj.get("response")
#                     or obj.get("answer")
#                     or obj.get("output")
#                     or obj.get("content")
#             )
#             if isinstance(resp, str):
#                 return True, resp
#         except Exception:
#             pass
#
#     return False, text


@app.post("/v1/chat/completions")
async def chat_completions(req: OAIChatCompletionsRequest):
    if rag_instance is None:
        raise RuntimeError("RAG is not initialized.")
    created_ts: int = int(time.time())
    completion_id: str = f"chatcmpl-{uuid.uuid4().hex}"
    model_id: str = req.model or "menon-1"
    # hardcoded
    session_id = 'id'


    collector.create_message(session_id=session_id)

    formatted_system_prompt, query, history = await _build_prompt_and_history(req.messages)

    collector.add_question(session_id=session_id, text=query)

    try:
        expanded_query: str = await explain_abbreviations(query, abbreviations)

    except Exception:
        expanded_query = query
    try:
        resolved_query: str = await resolve_anaphora(expanded_query, history)
    except Exception:
        resolved_query = expanded_query

    collector.add_expanded_question(session_id=session_id, text=expanded_query)
    collector.add_resolved_question(session_id=session_id, text=resolved_query)

    collector.update_time(session_id=session_id)

    collector.print_dto(session_id=session_id)

    async def run_lightrag():
        return await rag_instance.aquery(
            resolved_query,
            param=QueryParam(
                mode=QUERY_MODE,
                top_k=TOP_K,
                max_total_tokens=QUERY_MAX_TOKENS,
                history_turns=len(history),
                conversation_history=history,
                stream=req.stream,
            ),
            system_prompt=formatted_system_prompt
        )

    if not req.stream:
        try:
            result = await run_lightrag()
            if hasattr(result, "__aiter__"):
                chunks = []
                async for part in result:
                    if part:
                        chunks.append(str(part))
                result = "".join(chunks)
            content = str(result)
            collector.add_model_answer(session_id=session_id, text=content)
            collector.print_dto(session_id=session_id)
        except Exception as e:
            logger.exception("chat.completions non-stream error")
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": str(e),
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

        try:
            result = await run_lightrag()

            async def iter_pieces():
                if hasattr(result, "__aiter__"):
                    async for part in result:
                        if part:
                            yield str(part)
                else:
                    text = str(result)
                    step = 512
                    for i in range(0, len(text), step):
                        yield text[i:i + step]

            async for piece in iter_pieces():
                data = chunk({"content": piece})
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

            done = chunk({}, finish_reason="stop")
            yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
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
                "error": {"message": str(e), "type": "server_error"},
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


@app.post("/clear_history", response_model=ResetResponse)
async def reset_history(request: ResetRequest):
    chat_id = request.chat_id

    if chat_id in dialogue_histories:
        dialogue_histories.pop(chat_id)
        logger.info(f"История очищена для пользователя {chat_id}")
    else:
        logger.info(
            f"Попытка очистки истории: история для пользователя {chat_id} не найдена")

    return ResetResponse(chat_id=chat_id, status="ok")


@app.post("/v1/link_from_text", response_model=LinkOnlyResponse)
async def link_from_text(req: LinkOnlyRequest):
    return LinkOnlyResponse(url="https://nsu.ru/")


@app.post("/v1/image_from_text")
async def image_from_text(req: ImageOnlyRequest):
    """
    Временная заглушка: игнорируем текст запроса и возвращаем байты локального изображения.
    """
    path = Path("resources/image.jpg")
    if not path.exists() or not path.is_file():
        logger.error(f"Stub image not found: {path.resolve()}")
        return JSONResponse(status_code=500, content={"error": "stub_image_not_found", "path": str(path)})

    try:
        data: bytes = await asyncio.to_thread(path.read_bytes)
    except Exception as e:
        logger.exception("Failed to read stub image")
        return JSONResponse(status_code=500, content={"error": "read_failed", "message": str(e)})

    ctype, _ = mimetypes.guess_type(str(path))
    if not ctype:
        ctype = "application/octet-stream"

    filename = path.name or f"image-{uuid.uuid4().hex}"
    return StreamingResponse(
        io.BytesIO(data),
        media_type=ctype,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


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
