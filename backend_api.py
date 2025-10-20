import codecs
import json
import logging
import os
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from logging.handlers import RotatingFileHandler
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
            logger.info(f"📚 Загружено сокращений: {len(abbreviations)}")
    except Exception as e:
        logger.exception("Не удалось загрузить файл сокращений")

    yield  # <-- здесь FastAPI продолжает работу
    # Здесь можно вызвать await rag_instance.cleanup(), если нужно
    # Shutdown scheduler on app exit
    scheduler.shutdown()
    logger.info("⏰ Cache-clearing scheduler stopped")


app = FastAPI(lifespan=lifespan)
rag_instance: LightRAG
abbreviations = {}


class ChatRequest(BaseModel):
    chat_id: str
    message: str


class ChatResponse(BaseModel):
    chat_id: str
    response: str


class ResetRequest(BaseModel):
    chat_id: str


class ResetResponse(BaseModel):
    chat_id: str
    status: str


class OAIMsg(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


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


@app.post("/v1/chat/completions")
async def chat_completions(req: OAIChatCompletionsRequest):
    if rag_instance is None:
        raise RuntimeError("RAG is not initialized.")
    created_ts: int = int(time.time())
    completion_id: str = f"chatcmpl-{uuid.uuid4().hex}"
    model_id: str = req.model or "menon-1"

    formatted_system_prompt, query, history = await _build_prompt_and_history(req.messages)

    try:
        expanded_query: str = await explain_abbreviations(query, abbreviations)
    except Exception:
        expanded_query = query
    try:
        resolved_query: str = await resolve_anaphora(expanded_query, history)
    except Exception:
        resolved_query = expanded_query

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
                    "message": {"role": "assistant", "content": result},
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
    # main_answer: str = response_text
    # # if settings.enable_links_correction:
    # #     try:
    # #         main_answer = await ref_corrector.replace_markdown_links(main_answer)
    # #     except Exception:
    # #         logger.exception("LinkCorrecter failed; continue without correction")
    #
    # if settings.enable_links_addition:
    #     try:
    #         links: list[str] = await ref_searcher.get_links_from_answer(main_answer)
    #     except Exception:
    #         logger.exception("LinkSearcher failed; continue without links")
    #         links: list[str] = []
    #     if links:
    #         main_answer: str = f"{main_answer}\n\nИнтересные ссылки:\n- " + "\n- ".join(links)
    #
    # payload = {
    #     "id": chatcmpl_id,
    #     "object": "chat.completion",
    #     "created": created,
    #     "model": model_name,
    #     "choices": [{
    #         "index": 0,
    #         "message": {"role": "assistant", "content": main_answer},
    #         "finish_reason": "stop"
    #     }],
    #     "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    # }
    # return JSONResponse(payload)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    print(f"Получен запрос от {request.chat_id}: {request.message}")
    global rag_instance

    if rag_instance is None:
        raise RuntimeError("RAG is not initialized.")

    try:
        chat_id = request.chat_id
        query = request.message.strip()
        logger.info(f"New request from user {request.chat_id}: {query}")
        history = dialogue_histories[chat_id][-4:]

        expanded_query = await explain_abbreviations(query, abbreviations)
        logger.info(f"Query after expanding abbreviations: {expanded_query}")

        resolved_query = await resolve_anaphora(expanded_query, history)
        logger.info(f"После разрешения анафор: {resolved_query}")

        current_date_str = await get_current_period()
        formatted_system_prompt = SYSTEM_PROMPT_FOR_MENO.replace(
            "{current_date}",
            current_date_str
        )

        formatted_system_prompt = formatted_system_prompt.replace(
            "{conversation_history}",
            str(history)
        )
        logger.info(f"Formatted system prompt: {formatted_system_prompt}")

        response_text = await rag_instance.aquery(
            resolved_query,
            param=QueryParam(
                mode=QUERY_MODE,
                top_k=TOP_K,
                max_total_tokens=QUERY_MAX_TOKENS,
                max_=QUERY_MAX_TOKENS,
                history_turns=len(history),
                conversation_history=history,
            ),
            system_prompt=formatted_system_prompt
        )
        answer = response_text
        # if settings.enable_links_correction:
        #     answer = await ref_corrector.replace_markdown_links(answer)
        # if settings.enable_links_addition:
        #     answer = await ref_searcher.get_formated_answer(answer)
        dialogue_histories[chat_id].append({"role": "user", "content": query})
        dialogue_histories[chat_id].append(
            {"role": "assistant", "content": answer})
        logger.info(f"Ответ сформирован для {chat_id}: {answer}")
        return ChatResponse(chat_id=request.chat_id, response=answer)
    except Exception as e:
        logger.exception(
            f"Error while processing request from user {request.chat_id}")
        return ChatResponse(chat_id=request.chat_id, response="Произошла ошибка при обработке запроса.")


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
