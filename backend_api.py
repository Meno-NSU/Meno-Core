import codecs
import json
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import List, Dict
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from config import settings
from lightrag import QueryParam, LightRAG
from rag_engine import initialize_rag, SYSTEM_PROMPT_FOR_MENO, QUERY_MAX_TOKENS, TOP_K, resolve_anaphora, \
    explain_abbreviations, URLS_FNAME, LOCAL_EMBEDDER_NAME
from reference_searcher import ReferenceSearcher

QUERY_MODE: Literal["local", "global", "hybrid", "naive", "mix"] = "naive"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# user_id -> [{"role": "user"/"assistant", "content": "..."}]
dialogue_histories: Dict[str, List[Dict[str, str]]] = defaultdict(list)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_instance, abbreviations, ref_searcher
    rag_instance = await initialize_rag()
    ref_searcher = ReferenceSearcher(URLS_FNAME, model_name=LOCAL_EMBEDDER_NAME, threshold=0.75)
    try:
        with codecs.open(settings.abbreviations_file, mode='r', encoding='utf-8') as fp:
            abbreviations = json.load(fp)
            logger.info(f"📚 Загружено сокращений: {len(abbreviations)}")
    except Exception as e:
        logger.exception("Не удалось загрузить файл сокращений")

    yield  # <-- здесь FastAPI продолжает работу
    # Здесь можно вызвать await rag_instance.cleanup(), если нужно


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
        logger.info(f"Query after resolving anaphora: {resolved_query}")

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
            system_prompt=SYSTEM_PROMPT_FOR_MENO
        )
        # answer = ref_searcher.replace_references(response_text)
        answe = response_text
        dialogue_histories[chat_id].append({"role": "user", "content": query})
        dialogue_histories[chat_id].append({"role": "assistant", "content": answer})
        logger.info(f"Ответ сформирован для {chat_id}: {answer}")

        return ChatResponse(chat_id=request.chat_id, response=answer)
    except Exception as e:
        logger.exception(f"Error while processing request from user {request.chat_id}")
        return ChatResponse(chat_id=request.chat_id, response="Произошла ошибка при обработке запроса.")


@app.post("/clear_history", response_model=ResetResponse)
async def reset_history(request: ResetRequest):
    chat_id = request.chat_id

    if chat_id in dialogue_histories:
        dialogue_histories.pop(chat_id)
        logger.info(f"История очищена для пользователя {chat_id}")
    else:
        logger.info(f"Попытка очистки истории: история для пользователя {chat_id} не найдена")

    return ResetResponse(chat_id=chat_id, status="ok")
