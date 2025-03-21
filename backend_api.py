from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel
from lightrag.lightrag import QueryParam

from rag_engine import initialize_rag, SYSTEM_PROMPT_FOR_MENO, QUERY_MAX_TOKENS, TOP_K
import logging

QUERY_MODE: Literal["local", "global", "hybrid", "naive", "mix"] = "hybrid"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_instance
    rag_instance = await initialize_rag()
    yield  # <-- здесь FastAPI продолжает работу
    # Здесь можно вызвать await rag_instance.cleanup(), если нужно


app = FastAPI(lifespan=lifespan)
rag_instance = None


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
        query = request.message
        logger.info(f"New request from user {request.chat_id}: {query}")
        response_text = await rag_instance.aquery(
            query,
            param=QueryParam(
                mode=QUERY_MODE,
                top_k=TOP_K,
                max_token_for_text_unit=QUERY_MAX_TOKENS,
                max_token_for_global_context=QUERY_MAX_TOKENS,
                max_token_for_local_context=QUERY_MAX_TOKENS
            ),
            system_prompt=SYSTEM_PROMPT_FOR_MENO
        )
        logger.info(f"Response generated successfully for user {request.chat_id}")

        return ChatResponse(chat_id=request.chat_id, response=response_text)
    except Exception as e:
        logger.exception(f"Error while processing request from user {request.chat_id}")
        return ChatResponse(chat_id=request.chat_id, response="Произошла ошибка при обработке запроса.")
