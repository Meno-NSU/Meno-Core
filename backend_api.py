import codecs
import json
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from config import settings
from lightrag.lightrag import QueryParam

from rag_engine import initialize_rag, SYSTEM_PROMPT_FOR_MENO, QUERY_MAX_TOKENS, TOP_K, resolve_anaphora, \
    explain_abbreviations
import logging

from collections import defaultdict
from typing import List, Dict

QUERY_MODE: Literal["local", "global", "hybrid", "naive", "mix"] = "hybrid"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# user_id -> [{"role": "user"/"assistant", "content": "..."}]
dialogue_histories: Dict[int, List[Dict[str, str]]] = defaultdict(list)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_instance, abbreviations
    rag_instance = await initialize_rag()
    try:
        with codecs.open(settings.abbreviations_file, mode='r', encoding='utf-8') as fp:
            abbreviations = json.load(fp)
            logger.info(f"📚 Загружено сокращений: {len(abbreviations)}")
    except Exception as e:
        logger.exception("Не удалось загрузить файл сокращений")

    yield  # <-- здесь FastAPI продолжает работу
    # Здесь можно вызвать await rag_instance.cleanup(), если нужно


app = FastAPI(lifespan=lifespan)
rag_instance = None
abbreviations = {}


class ChatRequest(BaseModel):
    chat_id: int
    message: str


class ChatResponse(BaseModel):
    chat_id: int
    response: str


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
        history = dialogue_histories[chat_id][-10:]

        expanded_query = await explain_abbreviations(query, abbreviations)
        logger.info(f"После expand_abbr: {expanded_query}")

        resolved_query = await resolve_anaphora(expanded_query, history)
        logger.info(f"После разрешения анафор: {resolved_query}")

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
        dialogue_histories[chat_id].append({"role": "user", "content": query})
        dialogue_histories[chat_id].append({"role": "assistant", "content": response_text})
        logger.info(f"Ответ сформирован для {chat_id}: {response_text}")

        return ChatResponse(chat_id=request.chat_id, response=response_text)
    except Exception as e:
        logger.exception(f"Error while processing request from user {request.chat_id}")
        return ChatResponse(chat_id=request.chat_id, response="Произошла ошибка при обработке запроса.")
