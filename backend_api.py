import codecs
import json
import logging
import os
import re
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from typing import Literal

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from dateutil import parser as dtparser
from fastapi import FastAPI
from pydantic import BaseModel

from config import settings
from lightrag import QueryParam, LightRAG
from lightrag.utils import setup_logger
from link_correcter import LinkCorrecter
# from reference_searcher import ReferenceSearcher
from link_searcher import LinkSearcher
from pathlib import Path
from dataclasses import dataclass, asdict
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

SCORES_CACHE_FILE = os.getenv("SCORES_CACHE_FILE", "question_scores.json")


@dataclass
class UserQuestion:
    msg_id: str  # уникальный id события (chat_id + порядковый номер)
    chat_id: str
    content: str
    created_at_utc: datetime  # always timezone-aware UTC
    tokens_est: int  # грубая оценка длины (для веса)
    is_question: bool
    answer: Optional[str] = None  # <-- добавлено
    model_score: Optional[float] = None  # <-- оценка 0..100 (по итогу /pick_best_question)
    model_reason: Optional[str] = None  # <-- краткое обоснование


RECENT_QUESTIONS_BUFFER: deque[UserQuestion] = deque(maxlen=1000)
all_user_questions: list[UserQuestion] = []
# Файл-хранилище
QUESTIONS_FILE = Path(os.getenv("QUESTIONS_FILE", "questions.ndjson"))


def _ensure_questions_file():
    QUESTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not QUESTIONS_FILE.exists():
        QUESTIONS_FILE.touch()


def save_question_to_file(q: UserQuestion) -> None:
    """Append one record to NDJSON (atomic-like)."""
    _ensure_questions_file()
    with QUESTIONS_FILE.open("a", encoding="utf-8") as f:
        rec = asdict(q).copy()
        # datetime в iso
        rec["created_at_utc"] = q.created_at_utc.isoformat()
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_questions_window(start_utc: datetime, end_utc: datetime) -> List[UserQuestion]:
    """Stream read; фильтруем по окну времени."""
    if not QUESTIONS_FILE.exists():
        return []
    out: List[UserQuestion] = []
    with QUESTIONS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                t = dtparser.parse(obj["created_at_utc"])
                if start_utc <= t < end_utc:
                    out.append(UserQuestion(
                        msg_id=obj["msg_id"],
                        chat_id=obj["chat_id"],
                        content=obj["content"],
                        created_at_utc=t,
                        tokens_est=obj.get("tokens_est", len(obj["content"])),
                        is_question=obj.get("is_question", True),
                        answer=obj.get("answer"),
                        model_score=obj.get("model_score"),
                        model_reason=obj.get("model_reason"),
                    ))
            except Exception:
                logger.exception("Failed to parse questions.ndjson line")
    # по времени
    out.sort(key=lambda q: q.created_at_utc)
    return out


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
    answer: Optional[str] = None  # <-- добавлено
    created_at_iso: str
    is_question: bool
    prescore: Optional[float] = None
    model_score: Optional[float] = None  # <-- добавлено
    model_reason: Optional[str] = None  # <-- добавлено


class PickBestRequest(BaseModel):
    start: str
    end: str
    tz: str = "Asia/Novosibirsk"
    candidate_limit: int = 200
    use_prescoring: bool = False
    dedupe: bool = False
    # новое:
    scoring_criteria: str = (
        "- Насколько интересен и полезен вопрос для широкой аудитории;\n"
        "- Сколько актуальных тем он затрагивает и насколько глубоко;\n"
        "- Ясность формулировки и конкретика;\n"
        "- Новизна по сравнению с типичными вопросами."
    )
    do_final_llm_selection: bool = True  # если False — победитель = argmax(score)


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
        try:
            user_question.answer = answer
            save_question_to_file(user_question)
        except Exception:
            logger.exception("Failed to persist question/answer to file")

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
    1) читаем кандидатов из FILE по окну [start, end)
    2) (опц.) прескорим/сузим до candidate_limit (у тебя выключено по умолчанию)
    3) просим LLM оценить КАЖДЫЙ вопрос (0..100) по заданным критериям
    4) доп. финальный выбор победителя одной записью LLM (или argmax)
    """
    global rag_instance
    try:
        start_utc, end_utc = parse_time_range(req.start, req.end, req.tz)

        # --- 1) из файла (персистентный источник)
        window: List[UserQuestion] = load_questions_window(start_utc, end_utc)
        if not window:
            return PickBestResponse(
                winner_msg_id=None, winner_content=None,
                model_reason=None, raw_model_output=None,
                candidates_count=0, candidates=[],
            )

        # --- 2) сузить (если надо)
        candidates = window[: max(1, req.candidate_limit)]

        # --- 3) оценка КАЖДОГО вопроса (одним батч-промптом)
        scoring_user_prompt = build_per_question_scoring_prompt(candidates, req.scoring_criteria)
        scoring_system_prompt = "Ты строгий судья. Верни только JSON, без комментариев."

        scoring_output = await rag_instance.aquery(
            scoring_user_prompt,
            param=QueryParam(
                mode=QUERY_MODE,
                top_k=0,
                max_token_for_text_unit=QUERY_MAX_TOKENS,
                max_token_for_global_context=QUERY_MAX_TOKENS,
                max_token_for_local_context=QUERY_MAX_TOKENS,
                history_turns=0,
            ),
            system_prompt=scoring_system_prompt
        )

        parsed_scores = extract_json(scoring_output)
        # ожидаем список объектов [{msg_id, score, reason}]
        if isinstance(parsed_scores, list):
            reason_map: Dict[str, Tuple[Optional[float], Optional[str]]] = {}
            for obj in parsed_scores:
                mid = obj.get("msg_id")
                sc = obj.get("score")
                rs = obj.get("reason")
                try:
                    if sc is not None:
                        sc = float(sc)
                        sc = max(0.0, min(100.0, sc))
                except Exception:
                    sc = None
                if mid:
                    reason_map[mid] = (sc, rs)
            # присвоим
            for q in candidates:
                if q.msg_id in reason_map:
                    sc, rs = reason_map[q.msg_id]
                    q.model_score = sc
                    q.model_reason = rs
        else:
            logger.warning("LLM scoring JSON not parsed; leaving scores as None.")

        # --- 4) финальный выбор победителя
        winner_msg_id = None
        final_reason = None
        final_output_text = None

        if req.do_final_llm_selection:
            final_user_prompt = build_final_selection_prompt(candidates)
            final_system_prompt = "Выбери наилучший вопрос. Верни только JSON-объект."

            final_output_text = await rag_instance.aquery(
                final_user_prompt,
                param=QueryParam(
                    mode=QUERY_MODE,
                    top_k=0,
                    max_token_for_text_unit=QUERY_MAX_TOKENS,
                    max_token_for_global_context=QUERY_MAX_TOKENS,
                    max_token_for_local_context=QUERY_MAX_TOKENS,
                    history_turns=0,
                ),
                system_prompt=final_system_prompt
            )

            obj = extract_json(final_output_text)
            if isinstance(obj, dict):
                winner_msg_id = obj.get("winner_msg_id")
                final_reason = obj.get("reason")
        else:
            # argmax по score, tie-breaker: более высокий tokens_est, потом более ранний
            scored = [q for q in candidates if q.model_score is not None]
            if scored:
                scored.sort(key=lambda x: (x.model_score, x.tokens_est, -x.created_at_utc.timestamp()), reverse=True)
                winner_msg_id = scored[0].msg_id
                final_reason = "Auto-selected as highest scored question."
            else:
                # если нет оценок, fallback — последний по времени
                winner_msg_id = candidates[-1].msg_id
                final_reason = "Fallback to latest question."

        winner_content = None
        if winner_msg_id:
            for q in candidates:
                if q.msg_id == winner_msg_id:
                    winner_content = q.content
                    break

        # подготовим выдачу кандидатов
        candidates_response: List[CandidateItem] = []
        for idx, q in enumerate(candidates, start=1):
            candidates_response.append(
                CandidateItem(
                    rank=idx,
                    msg_id=q.msg_id,
                    chat_id=q.chat_id,
                    content=q.content,
                    answer=q.answer,
                    created_at_iso=q.created_at_utc.isoformat(),
                    is_question=q.is_question,
                    prescore=None,
                    model_score=(None if q.model_score is None else round(float(q.model_score), 2)),
                    model_reason=q.model_reason
                )
            )

        # можно сохранить обратно score/reason в файл (append как обновление записи)
        # Вариант простой: записать «апдейт» отдельной строкой (event sourcing).
        try:
            for q in candidates:
                if (q.model_score is not None) or (q.model_reason):
                    save_question_to_file(q)
        except Exception:
            logger.exception("Failed to append scored records to file")

        return PickBestResponse(
            winner_msg_id=winner_msg_id,
            winner_content=winner_content,
            model_reason=final_reason,
            raw_model_output=final_output_text or scoring_output,
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


def build_per_question_scoring_prompt(candidates: List[UserQuestion], criteria: str) -> str:
    """
    Просим модель вернуть JSON-массив объектов:
    [{"msg_id":"...", "score": 0..100, "reason": "..."}]
    """
    lines = []
    lines.append(
        "Оцени КАЖДЫЙ из приведённых ниже вопросов по шкале от 0 до 100.\n"
        "Критерии оценивания:\n"
        f"{criteria.strip()}\n\n"
        "Правила:\n"
        "- Оцени каждый вопрос независимо.\n"
        "- Строго верни JSON-массив с объектами вида "
        '{"msg_id":"<msg_id>", "score": <int>, "reason":"<краткое обоснование>"} '
        "в том же порядке, в каком даны вопросы.\n"
        "- Не добавляй ничего кроме JSON.\n\n"
        "Список вопросов:"
    )
    for i, q in enumerate(candidates, 1):
        ans = q.answer or ""
        lines.append(
            f'{i}) msg_id="{q.msg_id}" | chat_id="{q.chat_id}" | time_utc="{q.created_at_utc.isoformat()}"\n'
            f'   ВОПРОС: {q.content}\n'
            f'   ОТВЕТ:  {ans}'
        )
    return "\n".join(lines)


def build_final_selection_prompt(candidates: List[UserQuestion]) -> str:
    """
    Просим выбрать ровно один winner с краткой причиной; строгий JSON-объект.
    """
    lines = []
    lines.append(
        "Ниже — список вопросов. Выбери РОВНО ОДИН лучший с точки зрения интересности и актуальности.\n"
        'Верни строго JSON объект вида: {"winner_msg_id":"<msg_id>", "reason":"<краткое обоснование>"}\n'
        "Список вопросов:"
    )
    for i, q in enumerate(candidates, 1):
        sc = q.model_score
        rs = q.model_reason or ""
        lines.append(
            f'{i}) msg_id="{q.msg_id}" | score={sc if sc is not None else "NA"}\n'
            f'   ВОПРОС: {q.content}\n'
            f'   ОТВЕТ:  {q.answer or ""}\n'
            f'   ОБОСН.: {rs}'
        )
    return "\n".join(lines)


def extract_json(s: str):
    """достаём первый JSON-массив или объект из текста модели"""
    try:
        m = re.search(r"\[\s*\{.*\}\s*\]|\{\s*\".*", s, flags=re.DOTALL)
        if not m:
            return None
        return json.loads(m.group(0))
    except Exception:
        logger.exception("Failed to parse JSON from model output")
        return None
