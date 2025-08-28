import codecs
import hashlib
import json
import logging
import os
import re
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
import time
from pathlib import Path
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

_scores_cache: Dict[str, Dict[str, Optional[str]]] = {}


def _criteria_hash(criteria: str) -> str:
    return hashlib.sha256(criteria.strip().encode("utf-8")).hexdigest()


def load_scores_cache() -> None:
    global _scores_cache
    try:
        if os.path.exists(SCORES_CACHE_FILE):
            with open(SCORES_CACHE_FILE, "r", encoding="utf-8") as f:
                _scores_cache = json.load(f)
                if not isinstance(_scores_cache, dict):
                    _scores_cache = {}
        else:
            _scores_cache = {}
        logger.info(f"Scores cache loaded: {len(_scores_cache)} entries")
    except Exception:
        logger.exception("Failed to load scores cache")
        _scores_cache = {}


def save_scores_cache() -> None:
    try:
        tmp = SCORES_CACHE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_scores_cache, f, ensure_ascii=False)
        os.replace(tmp, SCORES_CACHE_FILE)
    except Exception:
        logger.exception("Failed to save scores cache")


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
    """
    Stream read NDJSON; фильтр по окну времени; collapse по msg_id (последняя запись побеждает).
    """
    if not QUESTIONS_FILE.exists():
        return []
    by_id: Dict[str, UserQuestion] = {}
    with QUESTIONS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                t = dtparser.parse(obj["created_at_utc"])
                if not (start_utc <= t < end_utc):
                    continue
                q = UserQuestion(
                    msg_id=obj["msg_id"],
                    chat_id=obj["chat_id"],
                    content=obj["content"],
                    created_at_utc=t,
                    tokens_est=obj.get("tokens_est", len(obj["content"])),
                    is_question=obj.get("is_question", True),
                    answer=obj.get("answer"),
                    model_score=obj.get("model_score"),
                    model_reason=obj.get("model_reason"),
                )
                # перезаписываем — так «последняя версия» останется
                by_id[q.msg_id] = q
            except Exception:
                logger.exception("Failed to parse questions.ndjson line")
    out = list(by_id.values())
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
    load_scores_cache()
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
    scoring_criteria: str = (
        "- Насколько интересен и полезен вопрос для студентов и абитуриентов Новосибирского Государственного Университета;\n"
        "- Сколько актуальных тем он затрагивает и насколько глубоко;\n"
        "- Ясность формулировки и конкретика;\n"
        "- Новизна по сравнению с типичными вопросами."
    )
    do_final_llm_selection: bool = True  # если False — winner=argmax(score)
    return_candidates_only: bool = False  # <-- НОВОЕ: вернуть список без LLM-оценки/выбора


class PickBestResponse(BaseModel):
    winner_msg_id: str | None
    winner_content: str | None
    model_reason: str | None
    raw_model_output: str | None
    candidates_count: int
    candidates: List[CandidateItem]  # <-- полный список кандидатов


class CandidatesOnlyRequest(BaseModel):
    start: str
    end: str
    tz: str = "Asia/Novosibirsk"
    candidate_limit: int = 200


class CandidatesOnlyResponse(BaseModel):
    candidates_count: int
    candidates: List[CandidateItem]


@app.post("/candidates_only", response_model=CandidatesOnlyResponse)
async def candidates_only(req: CandidatesOnlyRequest):
    try:
        start_utc, end_utc = parse_time_range(req.start, req.end, req.tz)
        window: List[UserQuestion] = load_questions_window(start_utc, end_utc)
        if not window:
            return CandidatesOnlyResponse(candidates_count=0, candidates=[])
        candidates = window[: max(1, req.candidate_limit)]
        resp: List[CandidateItem] = []
        for idx, q in enumerate(candidates, start=1):
            cinfo = _scores_cache.get(q.msg_id)
            ms = cinfo.get("score") if cinfo else q.model_score
            mr = cinfo.get("reason") if cinfo else q.model_reason
            resp.append(
                CandidateItem(
                    rank=idx, msg_id=q.msg_id, chat_id=q.chat_id,
                    content=q.content, answer=q.answer,
                    created_at_iso=q.created_at_utc.isoformat(),
                    is_question=q.is_question, prescore=None,
                    model_score=(None if ms is None else round(float(ms), 2)),
                    model_reason=mr
                )
            )
        return CandidatesOnlyResponse(candidates_count=len(resp), candidates=resp)
    except Exception:
        logger.exception("candidates_only failed")
        return CandidatesOnlyResponse(candidates_count=0, candidates=[])


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


def _snip(s: Optional[str], n: int = 2000) -> str:
    """Безопасное усечение строк для логов."""
    if s is None:
        return "None"
    if len(s) <= n:
        return s
    return f"{s[:n]}… [truncated {len(s) - n} chars]"


def _len(s: Optional[str]) -> int:
    return 0 if s is None else len(s)


@app.post("/pick_best_question", response_model=PickBestResponse)
async def pick_best_question(req: PickBestRequest):
    global rag_instance, _scores_cache
    req_id = uuid.uuid4().hex[:8]
    t0 = time.perf_counter()
    try:
        start_utc, end_utc = parse_time_range(req.start, req.end, req.tz)
        # 1) читаем кандидатов из файла
        window: List[UserQuestion] = load_questions_window(start_utc, end_utc)
        logger.info(f"[{req_id}] /pick_best_question: start")
        logger.info(
            f"[{req_id}] input: tz={req.tz!r}, start={req.start!r}, end={req.end!r}, "
            f"limit={req.candidate_limit}, return_candidates_only={req.return_candidates_only}, "
            f"do_final_llm_selection={req.do_final_llm_selection}"
        )
        logger.info(f"[{req_id}] time_range_utc: {start_utc.isoformat()} .. {end_utc.isoformat()}")
        if not window:
            logger.warning(f"[{req_id}] empty window -> return empty response")
            return PickBestResponse(
                winner_msg_id=None, winner_content=None,
                model_reason=None, raw_model_output=None,
                candidates_count=0, candidates=[],
            )

        # 2) сужаем (если надо)
        cap = max(1, req.candidate_limit)
        candidates = window[:cap]
        logger.info(f"[{req_id}] candidates clipped to: {len(candidates)} (limit={cap})")

        # 2a) режим "только кандидаты" — никаких оценок и финального выбора
        if req.return_candidates_only:
            logger.info(f"[{req_id}] return_candidates_only=True (no scoring/final)")
            candidates_response = []
            cache_hits = 0
            for idx, q in enumerate(candidates, start=1):
                # подтянем кеш-оценку «как есть», если вдруг уже есть
                cinfo = _scores_cache.get(q.msg_id)
                ms = cinfo.get("score") if cinfo else q.model_score
                mr = cinfo.get("reason") if cinfo else q.model_reason
                candidates_response.append(
                    CandidateItem(
                        rank=idx, msg_id=q.msg_id, chat_id=q.chat_id,
                        content=q.content, answer=q.answer,
                        created_at_iso=q.created_at_utc.isoformat(),
                        is_question=q.is_question, prescore=None,
                        model_score=(None if ms is None else round(float(ms), 2)),
                        model_reason=mr
                    )
                )
            logger.info(f"[{req_id}] return-only: cache_hits={cache_hits}, returning {len(candidates_response)}")
            return PickBestResponse(
                winner_msg_id=None, winner_content=None,
                model_reason=None, raw_model_output=None,
                candidates_count=len(candidates_response),
                candidates=candidates_response
            )

        # 3) оценка КАЖДОГО вопроса — только тех, кого нет в кеше по текущему criteria
        crit_hash = _criteria_hash(req.scoring_criteria)
        need_scoring: List[UserQuestion] = []
        cache_hits, cache_miss = 0, 0
        for q in candidates:
            entry = _scores_cache.get(q.msg_id)
            if not entry or entry.get("criteria_hash") != crit_hash:
                need_scoring.append(q)
                cache_miss += 1
            else:
                cache_hits += 1
        logger.info(
            f"[{req_id}] criteria_hash={crit_hash}, need_scoring={len(need_scoring)}, "
            f"cache_hits={cache_hits}, cache_miss={cache_miss}"
        )

        raw_scoring_output = None
        if need_scoring:
            scoring_user_prompt = build_per_question_scoring_prompt(need_scoring, req.scoring_criteria)
            scoring_system_prompt = "Ты строгий судья. Верни только JSON, без комментариев."
            logger.debug(
                f"[{req_id}] scoring: prompt_user_len={_len(scoring_user_prompt)}, "
                f"prompt_sys_len={_len(scoring_system_prompt)}"
            )

            t_llm = time.perf_counter()
            raw_scoring_output = await rag_instance.aquery(
                scoring_user_prompt,
                param=QueryParam(
                    mode=QUERY_MODE, top_k=0,
                    max_token_for_text_unit=QUERY_MAX_TOKENS,
                    max_token_for_global_context=QUERY_MAX_TOKENS,
                    max_token_for_local_context=QUERY_MAX_TOKENS,
                    history_turns=0,
                ),
                system_prompt=scoring_system_prompt
            )
            dt = time.perf_counter() - t_llm
            logger.info(f"[{req_id}] scoring LLM ok in {dt:.3f}s, out_len={_len(raw_scoring_output)}")
            logger.debug(f"[{req_id}] scoring LLM out (snip): {raw_scoring_output}")
            parsed_scores = extract_json(raw_scoring_output)
            if isinstance(parsed_scores, list):
                updated = 0
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
                    if mid is None:
                        continue
                    _scores_cache[mid] = {"score": sc, "reason": rs, "criteria_hash": crit_hash}
                    updated += 1
                if updated:
                    save_scores_cache()
                    logger.info(f"Scores cache updated: +{updated} entries")
            else:
                logger.warning("LLM scoring JSON not parsed; scores not updated.")

        # 4) смёрджим оценки из кеша в объекты кандидатов (в память)
        for q in candidates:
            cinfo = _scores_cache.get(q.msg_id)
            if cinfo:
                q.model_score = cinfo.get("score")
                q.model_reason = cinfo.get("reason")

        # 5) финальный выбор победителя (LLM или argmax)
        winner_msg_id = None
        final_reason = None
        final_output_text = None

        if req.do_final_llm_selection:
            final_user_prompt = build_final_selection_prompt(candidates)
            final_system_prompt = "Выбери наилучший вопрос. Верни только JSON-объект."

            final_output_text = await rag_instance.aquery(
                final_user_prompt,
                param=QueryParam(
                    mode=QUERY_MODE, top_k=0,
                    max_token_for_text_unit=QUERY_MAX_TOKENS,
                    max_token_for_global_context=QUERY_MAX_TOKENS,
                    max_token_for_local_context=QUERY_MAX_TOKENS,
                    history_turns=0,
                ),
                system_prompt=final_system_prompt
            )
            logger.info(f"LLM raw answer of scoring query: {final_output_text}")
            obj = extract_json(final_output_text)
            if isinstance(obj, dict):
                winner_msg_id = obj.get("winner_msg_id")
                final_reason = obj.get("reason")
            else:
                final_reason = final_output_text
        else:
            logger.info(f"[{req_id}] final selection via heuristic (argmax/fallback)")
            scored = [q for q in candidates if q.model_score is not None]
            if scored:
                scored.sort(key=lambda x: (x.model_score, x.tokens_est, -x.created_at_utc.timestamp()), reverse=True)
                winner_msg_id = scored[0].msg_id
                final_reason = "Auto-selected as highest scored question."
            else:
                winner_msg_id = candidates[-1].msg_id
                final_reason = "Fallback to latest question."
            logger.info(f"[{req_id}] heuristic winner: {winner_msg_id!r}")

        winner_content = None
        if winner_msg_id:
            for q in candidates:
                if q.msg_id == winner_msg_id:
                    winner_content = q.content
                    break
        logger.debug(f"[{req_id}] winner_content_len={_len(winner_content)}; reason_len={_len(final_reason)}")
        # 6) готовим ответ
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
        try:
            if need_scoring:
                updates = {}
                for q in need_scoring:
                    cinfo = _scores_cache.get(q.msg_id)
                    if not cinfo:
                        continue
                    updates[q.msg_id] = {
                        "model_score": cinfo.get("score"),
                        "model_reason": cinfo.get("reason"),
                    }
                if updates:
                    t_up = time.perf_counter()
                    n = upsert_scores_into_questions_file(updates)
                    logger.info(f"[{req_id}] questions.ndjson upsert: {n} records in {time.perf_counter()-t_up:.3f}s")
        except Exception:
            logger.exception(f"[{req_id}] upsert scores failed")
        # ВАЖНО: не пишем обратно «оценённые» q в NDJSON — чтобы не размножать дубликаты
        resp = PickBestResponse(
            winner_msg_id=winner_msg_id,
            winner_content=winner_content,
            model_reason=final_reason,
            raw_model_output=final_output_text or raw_scoring_output,
            candidates_count=len(candidates),
            candidates=candidates_response
        )
        total_dt = time.perf_counter() - t0
        logger.info(
            f"[{req_id}] done in {total_dt:.3f}s; "
            f"resp: candidates_count={resp.candidates_count}, "
            f"raw_model_output={resp.raw_model_output}"
        )
        return resp

    except Exception:
        logger.exception(f"[{req_id}] pick_best_question failed (outer)")
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
        "Ниже — список вопросов. Выбери РОВНО ОДИН лучший с точки зрения интересности и актуальности для абитуриентов и студентов Новосибирского Государственного Университета.\n"
        'Верни строго JSON объект вида: {"winner_msg_id":"<msg_id>", "reason":"<краткое обоснование>"}\n'
        "Список вопросов:"
    )
    for i, q in enumerate(candidates, 1):
        sc = q.model_score
        # rs = q.model_reason or ""
        lines.append(
            f'{i}) msg_id="{q.msg_id}" | score={sc if sc is not None else "NA"}\n'
            f'   ВОПРОС: {q.content}\n'
            # f'   ОТВЕТ:  {q.answer or ""}\n'
            # f'   ОБОСН.: {rs}'
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


def upsert_scores_into_questions_file(updates: Dict[str, Dict[str, Optional[float]]]) -> int:
    """
    Переписывает QUESTIONS_FILE атомарно, проставляя/обновляя model_score/model_reason
    для записей с msg_id, присутствующих в updates.
    updates: msg_id -> {"model_score": float|None, "model_reason": str|None}
    Возвращает количество обновлённых записей.
    """
    if not updates:
        return 0
    _ensure_questions_file()
    updated = 0
    tmp_path = str(QUESTIONS_FILE) + ".tmp"

    with QUESTIONS_FILE.open("r", encoding="utf-8") as src, open(tmp_path, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                dst.write(line)
                continue
            try:
                obj = json.loads(line)
                mid = obj.get("msg_id")
                if mid and mid in updates:
                    patch = updates[mid]
                    if patch.get("model_score") is not None:
                        obj["model_score"] = patch["model_score"]
                    if patch.get("model_reason") is not None:
                        obj["model_reason"] = patch["model_reason"]
                    updated += 1
                dst.write(json.dumps(obj, ensure_ascii=False) + "\n")
            except Exception:
                # пробрасываем как есть, чтобы не потерять строку
                dst.write(line)

    os.replace(tmp_path, QUESTIONS_FILE)
    return updated
