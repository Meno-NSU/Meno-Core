import openai
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc, ENCODER
from nltk import wordpunct_tokenize
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from config import settings

app = FastAPI()
TEMPERATURE = 0.3


class ChatRequest(BaseModel):
    chat_id: int
    message: str


class ChatResponse(BaseModel):
    chat_id: int
    response: str


openai_client = openai.AsyncOpenAI(
    api_key=settings.openai_api_key,
    base_url=settings.openai_base_url
)


async def llm_model_func(
        prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        settings.llm_model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=TEMPERATURE,
        **kwargs
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Можно будет использовать system_prompt или history_messages здесь
    response_text = await llm_model_func(
        prompt=request.message,
        system_prompt="Ты дружелюбный бот-помощник на русском языке.",
        history_messages=[],
    )
    return ChatResponse(chat_id=request.chat_id, response=response_text)
