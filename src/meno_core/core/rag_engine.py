import contextvars
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from logging import Logger
from pathlib import Path
from re import Match, Pattern
from typing import Any, Optional, List, Union

import numpy as np  # type: ignore[import]
# third-party imports without stubs - mark them to silence mypy import-untyped
import pytz  # type: ignore[import]
import torch
import torch.nn.functional as F  # type: ignore[import]
from lightrag import LightRAG  # type: ignore[import]
from lightrag.kg.shared_storage import initialize_pipeline_status, initialize_share_data  # type: ignore[import]
from lightrag.llm.openai import openai_complete_if_cache  # type: ignore[import]
from lightrag.utils import EmbeddingFunc  # type: ignore[import]
from nltk import wordpunct_tokenize  # type: ignore[import]
from nltk.stem.snowball import SnowballStemmer  # type: ignore[import]
from rank_bm25 import BM25Okapi  # type: ignore[import]
from torch import Tensor
from transformers import AutoModelForTokenClassification  # type: ignore[import]
from transformers import AutoTokenizer, AutoModel  # type: ignore[import]

from meno_core.config.settings import settings
from meno_core.core.gte_embedding import GTEEmbedding
from meno_core.core.lightrag_timing import get_current_rag_trace
from meno_core.core.lexical_normalizer import normalize_for_bm25, tokenize_for_bm25
from meno_core.core.prompts import (
    TEMPLATE_FOR_ABBREVIATION_EXPLAINING,
    SYSTEM_PROMPT_FOR_ANAPHORA_RESOLUTION,
    FEW_SHOTS_FOR_ANAPHORA,
)
from meno_core.core.rag.rerank.qwen_reranker import load_cached_qwen_reranker_backend, QwenRerankerBackend

_reranker_backend: QwenRerankerBackend | None = None

TEMPERATURE: float = settings.temperature
QUERY_MAX_TOKENS: int = settings.query_max_tokens
CHUNK_MAX_TOKENS: int = settings.chunk_max_tokens
ENTITY_MAX_TOKENS: int = settings.entity_max_tokens
RELATION_MAX_TOKENS: int = settings.relation_max_tokens
TOP_K: int = settings.top_k
CHUNK_TOP_K: int = settings.chunk_top_k
WORKING_DIR: Path = Path(settings.working_dir) if settings.working_dir else Path("./tmp_working_dir")
ABBREVIATIONS_PATH: Path | None = settings.abbreviations_path
URLS_PATH: str = str(settings.urls_path)
MULTILINGUAL_EMBEDDER_DIMENSION: int = settings.embedder_dim
MULTILINGUAL_EMBEDDER_MAX_TOKENS: int = settings.embedder_max_tokens
MULTILINGUAL_EMBEDDER_PATH: str = settings.multilingual_embedder_path
RERANKER_PATH: str = settings.reranker_path
DEFAULT_HALLUCINATION_THRESHOLD: float = 0.3

THINK_START_TOKEN = '<think>'
THINK_END_TOKEN = '</think>'

# ── Model override via contextvars ──
# Set by chat_completions handler so llm_model_func uses the model
# chosen by the user in the UI, not the hardcoded settings value.
_current_model_override: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    '_current_model_override', default=None
)
_current_base_url_override: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    '_current_base_url_override', default=None
)


logger: Logger = logging.getLogger(__name__)

_JSON_BLOCK_RE: Pattern[str] = re.compile(r"```json\s*([\s\S]*?)\s*```", re.IGNORECASE)
_JSON_OBJECT_RE: Pattern[str] = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


def _as_json_block(obj: dict) -> str:
    return "```json\n" + json.dumps(obj, ensure_ascii=False) + "\n```"


def _coerce_llm_response_to_json_block(text: str) -> str:
    """
    Возвращает STRING в формате код-блока ```json ... ``` с ключами:
      response, high_level_keywords, low_level_keywords.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    m: Optional[Match[str]] = _JSON_BLOCK_RE.search(text)
    if m:
        payload: str = m.group(1)
        try:
            obj = json.loads(payload)
            if "response" not in obj:
                for k in ("answer", "output", "content"):
                    if isinstance(obj.get(k), str):
                        obj["response"] = obj[k]
                        break
            obj.setdefault("high_level_keywords", obj.get("hl_keywords", []))
            obj.setdefault("low_level_keywords", obj.get("ll_keywords", []))
            obj.pop("hl_keywords", None)
            obj.pop("ll_keywords", None)
            return _as_json_block(obj)
        except Exception:
            logger.debug("Found ```json block``` but cannot parse; will wrap cleanly")

    m2: Optional[Match[str]] = _JSON_OBJECT_RE.search(text)
    if m2:
        try:
            obj = json.loads(m2.group(0))
            if "response" not in obj:
                for k in ("answer", "output", "content"):
                    if isinstance(obj.get(k), str):
                        obj["response"] = obj[k]
                        break
            obj.setdefault("high_level_keywords", obj.get("hl_keywords", []))
            obj.setdefault("low_level_keywords", obj.get("ll_keywords", []))
            obj.pop("hl_keywords", None)
            obj.pop("ll_keywords", None)
            return _as_json_block(obj)
        except Exception:
            logger.debug("Braces present but not valid JSON; wrapping plaintext")

    obj = {
        "response": text.strip(),
        "high_level_keywords": [],
        "low_level_keywords": [],
    }
    return _as_json_block(obj)


def resolve_llm_request(
    override_model: Optional[str] = None,
    override_base_url: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    return (
        override_model or _current_model_override.get() or settings.llm_model_name or "menon-1",
        override_base_url or _current_base_url_override.get() or settings.openai_base_url,
    )


@contextmanager
def llm_request_scope(
    override_model: Optional[str] = None,
    override_base_url: Optional[str] = None,
) -> Iterator[None]:
    model_token = _current_model_override.set(override_model)
    base_url_token = _current_base_url_override.set(override_base_url)
    try:
        yield
    finally:
        _current_base_url_override.reset(base_url_token)
        _current_model_override.reset(model_token)


def _strip_reasoning_prefix(text: str) -> str:
    """Strip <think>…</think> reasoning blocks from a non-streaming result.

    Used only for *internal* LLM calls (query rewriting, knowledge-graph
    extraction, etc.) where the frontend never sees the output.
    """
    thinking_end_position = text.find(THINK_END_TOKEN)
    if thinking_end_position >= 0:
        logger.debug(
            "Reasoning part was removed from 0 to %s position",
            thinking_end_position,
        )
        return text[thinking_end_position + len(THINK_END_TOKEN):]
    return text


async def _single_chunk_stream(text: str) -> AsyncIterator[str]:
    if text:
        yield text


async def _passthrough_streaming_llm_result(result: Any) -> AsyncIterator[str]:
    """Pass LLM output through as-is, preserving <think>…</think> tags.

    The frontend is responsible for parsing and rendering thinking blocks.
    """
    if isinstance(result, str):
        async for part in _single_chunk_stream(result.strip()):
            yield part
        return

    async for part in result:
        text = "" if part is None else str(part)
        if text:
            yield text


async def call_openai_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Any]] = None,
    *,
    stream: bool = False,
    enable_cot: bool = False,
    hashing_kv=None,
    override_model: Optional[str] = None,
    override_base_url: Optional[str] = None,
    preserve_thinking: bool = False,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    if history_messages is None:
        history_messages = []

    effective_model, effective_base_url = resolve_llm_request(
        override_model=override_model,
        override_base_url=override_base_url,
    )
    logger.debug(
        "Sending request to LLM (model=%s) with %s history messages, prompt_len=%s, stream=%s",
        effective_model,
        len(history_messages),
        len(prompt),
        stream,
    )

    trace = get_current_rag_trace()
    llm_started_at = time.perf_counter()
    result = await openai_complete_if_cache(
        model=effective_model,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        hashing_kv=hashing_kv,
        api_key=settings.openai_api_key,
        base_url=effective_base_url,
        stream=stream,
        **kwargs,
    )

    if trace is not None:
        trace.record_stage(
            "llm_open" if stream else "llm_nonstream",
            (time.perf_counter() - llm_started_at) * 1000,
            meta={
                "model": effective_model,
                "history_messages": len(history_messages),
                "prompt_len": len(prompt),
            },
        )
        if stream:
            trace.mark_llm_stream_open()

    if stream:
        return _passthrough_streaming_llm_result(result)

    if not isinstance(result, str):
        chunks: list[str] = []
        async for part in result:
            chunks.append(str(part))
        result = "".join(chunks)

    logger.debug("Received raw LLM answer, length=%s", len(result))
    if preserve_thinking:
        return result.strip()
    return _strip_reasoning_prefix(result).strip()


# ---------- LLM wrapper ----------
async def llm_model_func(prompt: str,
                         system_prompt: Optional[str] = None,
                         history_messages: Optional[List[Any]] = None,
                         *,
                         stream: bool = False,
                         enable_cot: bool = False,
                         hashing_kv=None,
                         **kwargs) -> Union[str, AsyncIterator[str]]:
    """
    Функция для взаимодействия с языковой моделью (LLM).

    Аргументы:
    - prompt (str): Пользовательский запрос.
    - system_prompt (str, optional): Системный запрос для настройки поведения модели.
    - history_messages (list, optional): История сообщений для контекста.
    - **kwargs: Дополнительные параметры для настройки запроса.

    Возвращает:
    - str: Ответ, сгенерированный языковой моделью.
    """
    effective_model = _current_model_override.get() or settings.llm_model_name
    try:
        override_model = kwargs.pop("override_model", None) or kwargs.pop("model", None)
        override_base_url = kwargs.pop("override_base_url", None) or kwargs.pop("base_url", None)
        return await call_openai_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            stream=stream,
            enable_cot=enable_cot,
            hashing_kv=hashing_kv,
            override_model=override_model,
            override_base_url=override_base_url,
            **kwargs,
        )

    except Exception as e:
        logger.error(
            "LLM call failed in llm_model_func (model=%s, prompt_len=%d): %s",
            effective_model,
            len(prompt) if prompt else 0,
            str(e),
            exc_info=True,
        )
        return "Извините, сейчас не удалось получить ответ от модели."


# make system_prompt Optional and avoid mutable default for history_messages
async def generate_with_llm(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Any]] = None,
        **kwargs: Any,
) -> str:
    override_model = kwargs.pop("override_model", None)
    override_base_url = kwargs.pop("override_base_url", None)

    kwargs.pop("stream", None)
    result = await call_openai_llm(
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        stream=False,
        enable_cot=False,
        override_model=override_model,
        override_base_url=override_base_url,
        **kwargs,
    )
    if isinstance(result, str):
        return result

    chunks: list[str] = []
    async for part in result:
        chunks.append(part)
    return "".join(chunks).strip()


async def explain_abbreviations(question: str, abbreviations: dict, override_model: Optional[str] = None, override_base_url: Optional[str] = None) -> str:
    """
    Обрабатывает вопрос пользователя, заменяя аббревиатуры на их расшифровки.

    Аргументы:
    - question (str): Вопрос пользователя.
    - abbreviations (dict): Словарь аббревиатур и их расшифровок.

    Возвращает:
    - str: Вопрос с заменёнными аббревиатурами или исходный вопрос, если аббревиатуры не найдены.
    """
    try:
        snow_stemmer: SnowballStemmer = SnowballStemmer(language='russian')
        filtered_abbreviations = dict()
        for cur_word in wordpunct_tokenize(question):
            if cur_word in abbreviations:
                filtered_abbreviations[cur_word] = abbreviations[cur_word]
            elif cur_word.lower() in abbreviations:
                filtered_abbreviations[cur_word] = abbreviations[cur_word.lower()]
            elif cur_word.upper() in abbreviations:
                filtered_abbreviations[cur_word] = abbreviations[cur_word.upper()]
            else:
                stem = snow_stemmer.stem(cur_word)
                if stem in abbreviations:
                    filtered_abbreviations[cur_word] = abbreviations[stem]
                elif stem.lower() in abbreviations:
                    filtered_abbreviations[cur_word] = abbreviations[stem.lower()]
                elif stem.upper() in abbreviations:
                    filtered_abbreviations[cur_word] = abbreviations[stem.upper()]
        del snow_stemmer
        if len(filtered_abbreviations) == 0:
            logger.debug("No abbreviations found in question")
            return question

        logger.debug(f"Found abbreviations: {filtered_abbreviations}")

        user_prompt: str = TEMPLATE_FOR_ABBREVIATION_EXPLAINING.format(
            abbreviations_dict=filtered_abbreviations,
            text_of_question=question
        )
        new_improved_question = await generate_with_llm(
            prompt=user_prompt,
            override_model=override_model,
            override_base_url=override_base_url
        )
        logger.debug(f"Improved question: {new_improved_question}")
        return new_improved_question
    except Exception as e:
        logger.error(f"Error in explain_abbreviations: {str(e)}", exc_info=True)
        return question


async def resolve_anaphora(question: str, history: list, override_model: Optional[str] = None, override_base_url: Optional[str] = None) -> str:
    """
    Обрабатывает вопрос пользователя, устраняя местоимённую анафору.

    Аргументы:
    - question (str): Вопрос пользователя.
    - history (list): История диалога в виде списка сообщений.

    Возвращает:
    - str: Вопрос с устранённой анафорой или исходный вопрос, если анафора не обнаружена.
    """
    try:
        logger.debug(f"Resolving anaphora for question: {question}")
        if (len(history) == 0) or (len(question.strip()) == 0):
            return question
        if (len(history) % 2) != 0:
            raise RuntimeError(f'The dialogue history length is wrong! Expected an even number, got {len(history)}.')
        expected_roles: list[str] = ['user', 'assistant']
        for _ in range((len(history) // 2) - 1):
            expected_roles += ['user', 'assistant']
        history_roles: list = [it['role'] for it in history]
        if history_roles != expected_roles:
            raise RuntimeError(f'The dialogue history roles are wrong! Expected {expected_roles}, got {history_roles}.')
        if len(history) > 6:
            history_ = history[-6:]
        else:
            history_ = history
        user_prompt: str = f'Человек: {" ".join(history_[0]["content"].split()).strip()}'
        user_prompt += f'\nБольшая языковая модель: : {" ".join(history_[1]["content"].split()).strip()}'
        for val in history_[2:]:
            if val['role'] == 'user':
                user_prompt += '\nЧеловек: '
            else:
                user_prompt += '\nБольшая языковая модель: '
            user_prompt += ' '.join(val['content'].split()).strip()
        del history_
        user_prompt += '\nЧеловек: ' + ' '.join(question.split()).strip()
        question_without_anaphora = await generate_with_llm(
            prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT_FOR_ANAPHORA_RESOLUTION,
            history_messages=FEW_SHOTS_FOR_ANAPHORA,
            override_model=override_model,
            override_base_url=override_base_url
        )
        logger.debug(f"Question after anaphora resolution: {question_without_anaphora}")
        return question_without_anaphora
    except Exception as e:
        logger.error(f"Error in resolve_anaphora: {str(e)}", exc_info=True)
        return question


# ---------- Embedding function ----------
# treat AutoModel and tokenizers as Any in signatures to avoid missing-stub attribute errors
async def gte_hf_embed(texts: List[str], tokenizer: Any, embed_model: Any) -> np.ndarray:
    started_at = time.perf_counter()
    try:
        device = next(embed_model.parameters()).device
        batch_dict = tokenizer(
            texts, return_tensors='pt',
            max_length=MULTILINGUAL_EMBEDDER_MAX_TOKENS, padding=True, truncation=True,
        ).to(device)
        with torch.no_grad():
            outputs = embed_model(**batch_dict)
            embeddings: Tensor = F.normalize(
                outputs.last_hidden_state[:, 0, :MULTILINGUAL_EMBEDDER_DIMENSION],
                p=2, dim=1
            )
        result: np.ndarray
        if embeddings.dtype == torch.bfloat16:
            result = embeddings.detach().to(torch.float32).cpu().numpy()
        else:
            result = embeddings.detach().cpu().numpy()

        logger.debug("Generated %s multilingual embeddings", len(texts))
        return result
    except Exception as e:
        logger.error(f"Error in gte_hf_embed: {str(e)}", exc_info=True)
        raise
    finally:
        trace = get_current_rag_trace()
        if trace is not None:
            text_count = len(texts)
            trace.increment_counter("embedding_calls", 1)
            trace.increment_counter("embedding_texts", text_count)
            trace.record_stage(
                "embedding_compute",
                (time.perf_counter() - started_at) * 1000,
                meta={"texts": text_count},
            )


async def qwen_hf_rerank(
        query: str,
        documents: List[str],
        reranker_backend: QwenRerankerBackend,
        top_n: Optional[int] = None,
) -> List[dict[str, Any]]:
    started_at = time.perf_counter()
    try:
        return reranker_backend.rerank_documents(query=query, documents=documents, top_n=top_n)
    finally:
        trace = get_current_rag_trace()
        if trace is not None:
            trace.increment_counter("reranker_calls", 1)
            trace.increment_counter("reranker_documents", len(documents))
            trace.record_stage(
                "reranker_model",
                (time.perf_counter() - started_at) * 1000,
                meta={
                    "documents": len(documents),
                    "top_n": top_n,
                },
            )


async def score_answer_relevance_to_prompt(
        prompt: str,
        answer: str,
        reranker_backend: QwenRerankerBackend,
        normalize: bool = True,
) -> float:
    """
    Оценивает, насколько ответ модели (answer) относится к исходному промпту (prompt)
    с помощью локального реранкера.

    :param prompt: исходный пользовательский запрос / промпт
    :param answer: ответ модели, который хотим проверить на релевантность
    :param reranker_backend: causal-lm реранкер с yes/no scoring
    :param normalize: сохранён для обратной совместимости; Qwen backend уже отдаёт значение в [0,1]
    :return: скор релевантности (float). Чем ближе к 1, тем ответ ближе к вопросу.
             Можно использовать пороги типа:
             - > 0.7 — ответ хорошо соответствует запросу
             - 0.4–0.7 — погранично
             - < 0.4 — высокая вероятность галлюцинаций/ухода от темы
    """
    try:
        rerank_result = await qwen_hf_rerank(
            query=answer,
            documents=[prompt],
            reranker_backend=reranker_backend,
            top_n=1,
        )
        raw_score: float = float(rerank_result[0]["relevance_score"])
        logger.debug("Raw reranker score for answer vs prompt: %s", raw_score)

        if not normalize:
            return raw_score
        return raw_score

    except Exception as e:
        logger.error(f"Error in score_answer_relevance_to_prompt: {str(e)}", exc_info=True)
        return 1.0


async def is_likely_hallucination(
        original_prompt: str,
        llm_answer: str,
        threshold: float = DEFAULT_HALLUCINATION_THRESHOLD,
) -> tuple[bool, float]:
    """
    Проверяет, насколько ответ модели связан с исходным промптом, и
    возвращает флаг "похоже на галлюцинацию" + сам скор.

    :param original_prompt: исходный запрос пользователя (желательно уже после
                            resolve_anaphora / explain_abbreviations)
    :param llm_answer: финальный текст ответа модели
    :param threshold: порог для флага галлюцинации.
                      По умолчанию 0.4 (score < threshold -> галлюцинация).
    :return: (is_hallucination, relevance_score)
             - is_hallucination: True, если ответ слабо связан с вопросом
             - relevance_score: нормализованный скор в [0,1]
    """
    if not original_prompt or not llm_answer:
        logger.warning("is_likely_hallucination: empty prompt or answer")
        return True, 0.0

    if _reranker_backend is None:
        logger.warning(
            "is_likely_hallucination: reranker is not initialized; "
            "returning (False, 0.0)"
        )
        return False, 0.0

    try:
        score = await score_answer_relevance_to_prompt(
            prompt=original_prompt,
            answer=llm_answer,
            reranker_backend=_reranker_backend,
            normalize=True,
        )
    except Exception as e:
        logger.error(f"Error in is_likely_hallucination: {str(e)}", exc_info=True)
        # В случае ошибки лучше не обвинять модель в галлюцинации, а просто вернуть "не знаем"
        return False, 0.0

    is_h = score < threshold
    logger.info(
        f"Hallucination check: score={score:.4f}, "
        f"threshold={threshold:.4f}, hallucination={is_h}"
    )
    return is_h, score


async def initialize_rag() -> tuple:  # returns (LightRAGEngine, GTEEmbedding, BM25Okapi, list)
    """
    Инициализирует объект LightRAG.

    Функция создаёт токенизатор и модель для эмбеддингов, а также настраивает объект LightRAG с необходимыми параметрами.

    Возвращает:
        LightRAG: Инициализированный объект LightRAG.
    """
    try:
        logger.info("Initializing RAG system...")
        embedder_path = MULTILINGUAL_EMBEDDER_PATH
        logger.info(f"Loading tokenizer and embedder model: {embedder_path}...")
        emb_tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            embedder_path,
        )
        emb_model = AutoModel.from_pretrained(
            embedder_path,
            trust_remote_code=True,
            # device_map='cuda:0'
            device_map='cpu',
        )
        emb_model.eval()
        logger.info(f"Model {embedder_path} loaded successfully")

        reranker_path = RERANKER_PATH
        logger.info(f"Loading tokenizer and reranker model: {reranker_path}...")
        reranker_backend = load_cached_qwen_reranker_backend(reranker_path)
        logger.info(f"Reranker {reranker_path} loaded successfully")

        global _reranker_backend
        _reranker_backend = reranker_backend

        logger.info("Initializing shared data and pipeline status...")
        initialize_share_data()
        await initialize_pipeline_status()

        token_cls = AutoModelForTokenClassification.from_pretrained(
            embedder_path, trust_remote_code=True, device_map='cuda' if torch.cuda.is_available() else 'cpu'
        )
        token_cls.eval()
        embedder: GTEEmbedding = GTEEmbedding(emb_tokenizer, token_cls, normalized=True)

        chunk_db, bm25 = await build_chunks_db_and_bm25(str(WORKING_DIR))
        logger2: Logger = logging.getLogger("links")
        logger2.debug("RAG init: chunks=%d", len(chunk_db))
        try:
            logger2.debug("RAG init: BM25 corpus size=%d", len(bm25.corpus))
        except Exception:
            pass

        logger.info("Creating LightRAG instance...")
        # Cosine thresholds: storage threshold (0.15) is stricter to avoid indexing
        # low-quality matches; retrieval threshold (0.05) is lenient to avoid
        # missing potentially relevant results during search.
        rag: LightRAG = LightRAG(
            working_dir=str(WORKING_DIR),
            kv_storage='JsonKVStorage',
            vector_db_storage_cls_kwargs={
                'cosine_better_than_threshold': 0.15
            },
            chunk_token_size=CHUNK_MAX_TOKENS,
            llm_model_func=llm_model_func,
            cosine_better_than_threshold=0.05,
            embedding_func=EmbeddingFunc(
                embedding_dim=MULTILINGUAL_EMBEDDER_DIMENSION,
                max_token_size=MULTILINGUAL_EMBEDDER_MAX_TOKENS,
                func=lambda texts: gte_hf_embed(
                    texts,
                    tokenizer=emb_tokenizer,
                    embed_model=emb_model
                )
            ),
            rerank_model_func=partial(qwen_hf_rerank, reranker_backend=reranker_backend),
            addon_params={'language': 'Russian'}
        )
        logger.info("Initializing RAG storages...")
        await rag.initialize_storages()
        logger.info("RAG system initialized successfully")

        from meno_core.core.lightrag_engine import LightRAGEngine
        engine = LightRAGEngine(
            rag_instance=rag,
            embedder=embedder,
            bm25=bm25,
            chunk_db=chunk_db
        )
        return engine, embedder, bm25, chunk_db
    except Exception as e:
        logger.error(f"Error initializing RAG: {str(e)}", exc_info=True)
        raise


# ---------- Date string generation ----------
async def get_current_period():
    today: datetime = datetime.now(pytz.timezone("Asia/Novosibirsk"))
    day: int = today.day
    month: int = today.month
    year: int = today.year

    month_names: dict[int, str] = {
        1: "января", 2: "февраля", 3: "марта", 4: "апреля",
        5: "мая", 6: "июня", 7: "июля", 8: "августа",
        9: "сентября", 10: "октября", 11: "ноября", 12: "декабря"
    }

    # Format day with proper suffix for Russian
    if 11 <= day <= 19:
        day_str = f"{day}ое"
    else:
        last_digit = day % 10
        if last_digit == 1:
            day_str = f"{day}ое"
        elif last_digit == 2:
            day_str = f"{day}ое"
        elif last_digit == 3:
            day_str = f"{day}ье"
        elif last_digit == 4:
            day_str = f"{day}ое"
        else:
            day_str = f"{day}ое"

    return f"{day_str} {month_names[month]} {year} года"


_snow = SnowballStemmer(language='russian')


async def tokenize_and_normalize(text: str) -> str:
    return normalize_for_bm25(text)


async def build_chunks_db_and_bm25(working_dir: Union[str, Path]):
    # coerce Path to str so callers can pass Path and code that expects str will work
    working_dir_str: str
    if isinstance(working_dir, Path):
        working_dir_str = str(working_dir)
    else:
        working_dir_str = working_dir

    chunks_path = Path(working_dir_str) / "kv_store_text_chunks.json"
    if not chunks_path.exists():
        logger.warning(f"Chunk database not found at {chunks_path}. Initializing empty.")
        return {}, BM25Okapi([["dummy"]])

    with chunks_path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    # [(content, full_doc_id)]
    chunk_db = [(raw[k]["content"], raw[k]["full_doc_id"]) for k in raw]
    norm_texts = [tokenize_for_bm25(content) for content, _doc_id in chunk_db]
    bm25 = BM25Okapi(norm_texts)
    return chunk_db, bm25
