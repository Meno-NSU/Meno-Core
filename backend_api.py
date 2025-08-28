import codecs
import json
import logging
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple, re, Optional
from typing import Literal
from dateutil import parser as dtparser

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI
from pydantic import BaseModel

from config import settings
from lightrag import QueryParam, LightRAG
from lightrag.utils import setup_logger
from link_correcter import LinkCorrecter
# from reference_searcher import ReferenceSearcher
from link_searcher import LinkSearcher
from rag_engine import initialize_rag, SYSTEM_PROMPT_FOR_MENO, QUERY_MAX_TOKENS, TOP_K, resolve_anaphora, \
    explain_abbreviations, get_current_period

QUERY_MODE: Literal["local", "global", "hybrid", "naive", "mix"] = settings.query_mode

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# user_id -> [{"role": "user"/"assistant", "content": "..."}]
dialogue_histories: Dict[str, List[Dict[str, str]]] = defaultdict(list)


@dataclass
class UserQuestion:
    msg_id: str  # уникальный id события (можно chat_id + порядковый номер)
    chat_id: str
    content: str
    created_at_utc: datetime  # always timezone-aware UTC
    tokens_est: int  # грубая оценка длины (для веса)
    is_question: bool


RECENT_QUESTIONS_BUFFER: deque[UserQuestion] = deque(maxlen=1000)
all_user_questions: list[UserQuestion] = []


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
    setup_logger("light_rag_log", "WARNING", False, str(settings.log_file_path))
    rag_instance = await initialize_rag()
    # ref_searcher = ReferenceSearcher(URLS_FNAME, model_name=LOCAL_EMBEDDER_NAME, threshold=0.75)
    if settings.enable_links_addition:
        ref_searcher = LinkSearcher(settings.urls_path, rag_instance, settings.top_k,
                                    dist_threshold=settings.dist_threshold, max_links=settings.max_links)
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


def _now_utc() -> datetime:
    return datetime.utcnow().replace(tzinfo=pytz.UTC)


def parse_time_range(start_str: str, end_str: str, tz_str: str) -> Tuple[datetime, datetime]:
    """
    Принимает строки дат/времени (например, '2025-08-27 12:00') и tz (например, 'Asia/Novosibirsk').
    Возвращает границы в UTC [start_utc, end_utc).
    """
    tz = pytz.timezone(tz_str)
    start_local = tz.localize(dtparser.parse(start_str, dayfirst=False))
    end_local = tz.localize(dtparser.parse(end_str, dayfirst=False))
    start_utc = start_local.astimezone(pytz.UTC)
    end_utc = end_local.astimezone(pytz.UTC)
    if end_utc <= start_utc:
        raise ValueError("end must be after start")
    return start_utc, end_utc


# ДОБАВЬ рядом с утилитами
def _build_selection_prompt(candidates: list[UserQuestion]) -> str:
    """
    Формируем user-промпт. Системный промпт будет строго: "Выбери наилучший вопрос".
    Просим вернуть JSON — чтобы легко парсить.
    """
    lines = []
    lines.append(
        "Ниже дан список пользовательских вопросов. "
        "Выбери ровно ОДИН наилучший вопрос, оценивая ясность формулировки, информативность и новизну относительно других в списке. "
        "Верни строго JSON без препамбулы и комментариев, формат:\n"
        '{"winner_msg_id":"<msg_id>", "reason":"<краткое обоснование на русском>"}\n'
        "Список вопросов:"
    )
    for i, q in enumerate(candidates, 1):
        lines.append(
            f'{i}) msg_id="{q.msg_id}" | chat_id="{q.chat_id}" | time_utc="{q.created_at_utc.isoformat()}"\n'
            f'   {q.content}'
        )
    return "\n".join(lines)


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


class CandidateItem(BaseModel):
    rank: int
    msg_id: str
    chat_id: str
    content: str
    created_at_iso: str
    is_question: bool
    prescore: Optional[float] = None  # None, если прескор не считали


class PickBestRequest(BaseModel):
    start: str  # напр. "2025-08-27 12:00"
    end: str  # напр. "2025-08-27 18:00"
    tz: str = "Asia/Novosibirsk"
    # чтобы не слать в LLM сотни строк — берем топ по нашему скореру и только их отдаём модели
    candidate_limit: int = 200
    # если хочешь совсем без нашего скорера и просто «как есть» — поставь False
    use_prescoring: bool = False
    dedupe: bool = False  # влияет на новизну в скорере


class PickBestResponse(BaseModel):
    winner_msg_id: str | None
    winner_content: str | None
    model_reason: str | None
    raw_model_output: str | None
    candidates_count: int
    candidates: List[CandidateItem]  # <-- полный список кандидатов


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    print(f"Получен запрос от {request.chat_id}: {request.message}")
    global rag_instance

    if rag_instance is None:
        raise RuntimeError("RAG is not initialized.")

    try:
        chat_id = request.chat_id
        query = request.message.strip()
        created_at = _now_utc()
        user_question = UserQuestion(
            msg_id=f"{chat_id}:{created_at.timestamp()}",
            chat_id=chat_id,
            content=query,
            created_at_utc=created_at,
            tokens_est=len(query),
            is_question=True
        )
        all_user_questions.append(user_question)
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
        logger.info(f"Formatted system prompt: {formatted_system_prompt}")

        response_text = await rag_instance.aquery(
            resolved_query,
            param=QueryParam(
                mode=QUERY_MODE,
                top_k=TOP_K,
                max_token_for_text_unit=QUERY_MAX_TOKENS,
                max_token_for_global_context=QUERY_MAX_TOKENS,
                max_token_for_local_context=QUERY_MAX_TOKENS,
                history_turns=len(history)
            ),
            system_prompt=formatted_system_prompt
        )
        # answer = ref_searcher.replace_references(response_text)
        answer = response_text
        if settings.enable_links_correction:
            answer = await ref_corrector.replace_markdown_links(answer)
        if settings.enable_links_addition:
            answer = await ref_searcher.get_formated_answer(resolved_query, answer)
        # answer = response_text
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


@app.post("/pick_best_question", response_model=PickBestResponse)
async def pick_best_question(req: PickBestRequest):
    """
    1) фильтруем вопросы по окну [start, end) в req.tz
    2) (опционально) прескорим и возьмём топ-N кандидатов
    3) отправим список в LLM с системным промптом "Выбери наилучший вопрос"
    4) распарсим ответ JSON и вернём msg_id и текст вопроса
    """
    global rag_instance

    try:
        start_utc, end_utc = parse_time_range(req.start, req.end, req.tz)

        window = [q for q in all_user_questions if start_utc <= q.created_at_utc < end_utc]
        if not window:
            return PickBestResponse(
                winner_msg_id=None,
                winner_content=None,
                model_reason=None,
                raw_model_output=None,
                candidates_count=0,
                candidates=[],
            )
        candidates_list = window[: max(1, req.candidate_limit)]
        candidates_response: List[CandidateItem] = []
        for idx, q in enumerate(candidates_list, start=1):
            candidates_response.append(
                CandidateItem(rank=idx, msg_id=q.msg_id, chat_id=q.chat_id, content=q.content,
                              created_at_iso=q.created_at_utc.isoformat(), is_question=q.is_question)
            )

        # прескрининг по нашему скореру, чтобы не перегружать контекст
        candidates: list[UserQuestion]
        # Без прескора берём просто первые N по времени (или отсортируй как тебе удобно)
        window.sort(key=lambda q: q.created_at_utc)
        candidates = window[: max(1, req.candidate_limit)]

        user_prompt = _build_selection_prompt(candidates)
        system_prompt = "Выбери наилучший вопрос"

        # Вызов твоей LLM. Для выбора тут не нужен глобальный/локальный контекст, поэтому режим "naive".
        # Параметры top_k/макс. токены выставляю минимально необходимыми.
        llm_output = await rag_instance.aquery(
            user_prompt,
            param=QueryParam(
                mode="naive",
                top_k=0,
                max_token_for_text_unit=256,
                max_token_for_global_context=256,
                max_token_for_local_context=256,
                history_turns=0,
            ),
            system_prompt=system_prompt
        )

        # Парсим JSON из ответа модели (очень строго не делаем — но стараемся)
        winner_msg_id = None
        model_reason = None
        try:
            # модель могла обернуть что-то ещё — попробуем вытащить первый JSON-объект
            match = re.search(r"\{.*\}", llm_output, flags=re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                winner_msg_id = data.get("winner_msg_id")
                model_reason = data.get("reason")
        except Exception:
            logger.exception("Failed to parse LLM JSON output")

        # находим текст вопроса по msg_id
        winner_content = None
        if winner_msg_id:
            for q in candidates:  # сужаем поиск до кандидатов
                if q.msg_id == winner_msg_id:
                    winner_content = q.content
                    break

        return PickBestResponse(
            winner_msg_id=winner_msg_id,
            winner_content=winner_content,
            model_reason=model_reason,
            raw_model_output=llm_output,
            candidates_count=len(candidates),
            candidates=candidates_response
        )

    except Exception:
        logger.exception("pick_best_question failed")
        return PickBestResponse(
            winner_msg_id=None,
            winner_content=None,
            model_reason=None,
            raw_model_output=None,
            candidates_count=0,
            candidates=[],
        )