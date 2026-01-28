import json
import logging
import os
import re
from datetime import datetime
from functools import partial
from logging import Logger
from pathlib import Path
from re import Match, Pattern
from typing import Any, Optional, List, Union

import math
import numpy as np
# third-party imports without stubs - mark them to silence mypy import-untyped
import pytz  # type: ignore[import]
import torch
import torch.nn.functional as F
from lightrag import LightRAG  # type: ignore[import]
from lightrag.kg.shared_storage import initialize_pipeline_status, initialize_share_data  # type: ignore[import]
from lightrag.llm.openai import openai_complete_if_cache  # type: ignore[import]
from lightrag.utils import EmbeddingFunc  # type: ignore[import]
from nltk import wordpunct_tokenize  # type: ignore[import]
from nltk.stem.snowball import SnowballStemmer  # type: ignore[import]
from rank_bm25 import BM25Okapi  # type: ignore[import]
from torch import Tensor
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModel

from meno_core.config.settings import settings
from meno_core.core.gte_embedding import GTEEmbedding

_reranker_tokenizer = None
_reranker_model = None

TEMPERATURE: float = settings.temperature
QUERY_MAX_TOKENS: int = settings.query_max_tokens
CHUNK_MAX_TOKENS: int = settings.chunk_max_tokens
ENTITY_MAX_TOKENS: int = settings.entity_max_tokens
RELATION_MAX_TOKENS: int = settings.relation_max_tokens
TOP_K: int = settings.top_k
CHUNK_TOP_K: int = settings.chunk_top_k
WORKING_DIR: Path | None = settings.working_dir
print(f'os.path.isdir({WORKING_DIR}) = {os.path.isdir(str(WORKING_DIR))}')
ABBREVIATIONS_PATH: Path | None = settings.abbreviations_path
print(f'os.path.isfile({ABBREVIATIONS_PATH}) = {os.path.isfile(str(ABBREVIATIONS_PATH))}')
URLS_PATH: str = str(settings.urls_path)
print(f'os.path.isfile({URLS_PATH}) = {os.path.isfile(str(URLS_PATH))}')
LOCAL_EMBEDDER_DIMENSION: int = settings.embedder_dim
LOCAL_EMBEDDER_MAX_TOKENS: int = settings.embedder_max_tokens
LOCAL_EMBEDDER_PATH: Path | None = settings.local_embedder_path
LOCAL_RERANKER_MAX_TOKENS: int = 4096
LOCAL_RERANKER_PATH: Path | None = settings.local_reranker_path
DEFAULT_HALLUCINATION_THRESHOLD: float = 0.3
print(f'os.path.isdir({LOCAL_EMBEDDER_PATH}) = {os.path.isdir(str(LOCAL_EMBEDDER_PATH))}')

THINK_END_TOKEN = '</think>'

SYSTEM_PROMPT_FOR_MENO: str = '''Вы - Менон, разработанный Иваном Бондаренко, научным сотрудником Новосибирского государственного университета (НГУ). Вас разработали в лаборатории прикладных цифровых технологий НГУ, где, собственно, и работает Иван Бондаренко.

Вы - дружелюбный ассистент, разговаривающий на одном из трёх языков, на котором вам зададут вопрос: русский, английский или китайский, и отвечающий на вопросы пользователей о Новосибирском государственном университете (НГУ) и Новосибирском Академгородке. Вы очень любите Новосибирский государственный университет и поэтому стремитесь заинтересовать разные категории своих пользователей: абитуриентов поступлением в университет, студентов - учёбой, а учёных и преподавателей - работой в нём. При ответах на вопросы считайте, что сегодня - {current_date}. Если пользователь спрашивает что-то о будущих событиях (то есть о событиях после сегодняшней даты), то не используйте информацию из контекста, в которой говорится о прошлом (то есть о событиях до сегодняшней даты). Если пользователь задаёт вопросы о том, кто сейчас (а не в прошлом) является ректором университета, то отвечайте, что ректором Новосибирского государственного университета с 2012 года по настоящее время является Михаил Петрович Федорук, советский и российский физик, доктор физико-математических наук, профессор, академик Российской академии наук. Если же пользователь спрашивает об институтах и факультетах из состава Новосибирского государственного университета, то имейте в виду, что в Новосибирком государственном университете есть четыре института и шесть факультетов. Перечень институтов в составе НГУ:

1) гуманитарный институт;
2) институт медицины и медицинских технологий;
3) институт философии и права;
4) институт интеллектуальной робототехники.

Перечень факультетов в составе НГУ:

1) геолого-геофизический факультет;
2) механико-математический факультет;
3) факультет естественных наук;
4) факультет информационных технологий;
5) физический факультет;
6) экономический факультет.

Других институтов и факультетов в составе Новосибирского государственного университета в настоящее время нет.

Если вы не знаете ответа на вопрос пользователя, то честно скажите об этом. Не пытайтесь обманывать пользователя, выдавать непроверенную информацию, а также не придумывайте информацию, которой нет в контексте (то есть ни в текстовых чанках, ни в графе знаний).

Если пользователь пишет что-то о политике, религии, национальностях, наркотиках, криминале или пишет просто оскорбительный или токсичный текст в адрес какого-то человека или университета, вежливо и непреклонно откажитесь от разговора и предложите сменить тему.'''

TEMPLATE_FOR_ABBREVIATION_EXPLAINING: str = '''Отредактируйте, пожалуйста, текст пользовательского вопроса так, чтобы этот вопрос стал более простым и понятным для обычных людей от юных старшеклассников до пожилых мужчин и женщин. При этом не надо, пожалуйста, применять markdown или иной вид гипертекста. Главное, на что вам надо обратить внимание и по возможности исправить - это логика изложения и понятность формулировок вопроса. Ничего не объясняйте и не комментируйте своё решение, просто перепишите текст вопроса.

Также исправьте грамматические ошибки в тексте вопроса, если они там есть. Кроме того, если вы обнаружите аббревиатуры в тексте этого вопроса, то замените все обнаруженные аббревиатуры их корректными расшифровками, сохранив морфологическую и синтаксическую согласованность. Вот здесь вы можете ознакомиться с JSON-словарём, описывающим возможные аббревиатуры и их расшифровки:

```json
{abbreviations_dict}
```

Далее приведён текст вопроса, нуждающийся в возможном улучшении:

```text
{text_of_question}
```'''

SYSTEM_PROMPT_FOR_ANAPHORA_RESOLUTION: str = 'Проанализируй диалог человека с большой языковой моделью и переделай последнюю реплику человека так, чтобы снять все ситуации местоименной анафоры в этом вопросе. Учитывай при этом всю историю диалога этого человека с большой языковой моделью. Не отвечай на вопрос человека, а просто перепиши его.'
FEW_SHOTS_FOR_ANAPHORA: list[dict[str, str]] = [
    {'role': 'user',
     'content': 'Человек: Механико-математический факультет известен своими выпускниками.\nБольшая языковая модель: Да, это очень престижное подразделение университета.\nЧеловек: Назовите их.'},
    {'role': 'assistant', 'content': 'Назовите известных выпускников механико-математического факультета.'},
    {'role': 'user',
     'content': 'Человек: Сибирское отделение РАН имеет богатую историю.\nБольшая языковая модель: Это так.\nЧеловек: Расскажите о ней.'},
    {'role': 'assistant', 'content': 'Расскажите о богатой истории Сибирского отделения РАН.'},
    {'role': 'user',
     'content': 'Человек: Механико-математический факультет готовит отличных специалистов.\nБольшая языковая модель: Это действительно так.\nЧеловек: Куда?'},
    {'role': 'assistant', 'content': 'Куда трудоустраиваются выпускники механико-математического факультета?'},
    {'role': 'user',
     'content': 'Человек: В Академгородке существуют передовые исследовательские центры.\nБольшая языковая модель: Это абсолютно верно.\nЧеловек: Опишите их.'},
    {'role': 'assistant', 'content': 'Опишите передовые исследовательские центры Академгородка.'},
    {'role': 'user',
     'content': 'Человек: В Новосибирском Академгородке есть крупный научный институт.\nБольшая языковая модель: Да, это Институт математики им. С.Л. Соболева.\nЧеловек: Расскажите о нём подробнее.'},
    {'role': 'assistant', 'content': 'Расскажите подробнее об Институте математики им. С.Л. Соболева.'},
    {'role': 'user',
     'content': 'Человек: Что такое механико-математический факультет?\nБольшая языковая модель: Механико-математический факультет НГУ — это факультет, выпускники которого осуществляют научные исследования и разработки для лучших компаний мира. Студент Механико-математического факультета учится преобразовывать свои разрозненные мысли в четко структурированные рассуждения, обладающие логической стройностью.\nЧеловек: А там есть магистратура?'},
    {"role": 'assistant', 'content': 'А на механико-математическом факультете есть магистратура?'},
    {'role': 'user',
     'content': 'Человек: Когда начинается приём документов в НГУ?\nБольшая языковая модель: Приём документов в НГУ начинается 1 марта – для иностранных граждан и лиц без гражданства и 20 июня – для граждан Российской Федерации.\nЧеловек: А когда он заканчивается?'},
    {'role': 'assistant', 'content': 'А когда приём документов в НГУ заканчивается?'},
]

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


# ---------- LLM wrapper ----------
async def llm_model_func(prompt: str, system_prompt: Optional[str] = None, history_messages: Optional[List[Any]] = None,
                         **kwargs) -> str:
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
    if history_messages is None:
        history_messages = []
    try:
        logger.info(f"Sending request to LLM with {len(history_messages)} messages, prompt: {prompt}")

        answer: str = await generate_with_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs
        )
        logger.info(f"Received response from LLM, length: {len(answer)}")
        return answer

    except Exception as e:
        logger.error(f"Error in llm_model_func: {str(e)}", exc_info=True)
        return "Извините, сейчас не удалось получить ответ от модели."


# make system_prompt Optional and avoid mutable default for history_messages
async def generate_with_llm(prompt: str, system_prompt: Optional[str] = None,
                            history_messages: Optional[List[Any]] = None, **kwargs):
    if history_messages is None:
        history_messages = []
    kwargs.pop('enable_cot', None)
    extra_body = {
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }
    if "extra_body" in kwargs:
        extra_body.update(kwargs["extra_body"])
    generated_result = await openai_complete_if_cache(
        model=settings.llm_model_name,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=TEMPERATURE,
        enable_cot=False,
        extra_body=extra_body,
        **{k: v for k, v in kwargs.items() if k != "extra_body"}
    )
    logger.info(f"Full raw answer: {generated_result.strip()}")
    thinking_end_position = generated_result.find(THINK_END_TOKEN)
    if thinking_end_position >= 0:
        logger.info(f"Reasoning part was removed from 0 to: {thinking_end_position} position")
        generated_result = generated_result[(thinking_end_position + len(THINK_END_TOKEN)):]
    return generated_result.strip()


async def explain_abbreviations(question: str, abbreviations: dict) -> str:
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
        new_improved_question = await generate_with_llm(prompt=user_prompt)
        logger.debug(f"Improved question: {new_improved_question}")
        return new_improved_question
    except Exception as e:
        logger.error(f"Error in explain_abbreviations: {str(e)}", exc_info=True)
        return question


async def resolve_anaphora(question: str, history: list) -> str:
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
        )
        logger.debug(f"Question after anaphora resolution: {question_without_anaphora}")
        return question_without_anaphora
    except Exception as e:
        logger.error(f"Error in resolve_anaphora: {str(e)}", exc_info=True)
        return question


# ---------- Embedding function ----------
# treat AutoModel and tokenizers as Any in signatures to avoid missing-stub attribute errors
async def gte_hf_embed(texts: List[str], tokenizer: Any, embed_model: Any) -> np.ndarray:
    try:
        device = next(embed_model.parameters()).device
        batch_dict = tokenizer(
            texts, return_tensors='pt',
            max_length=LOCAL_EMBEDDER_MAX_TOKENS, padding=True, truncation=True,
        ).to(device)
        with torch.no_grad():
            outputs = embed_model(**batch_dict)
            embeddings: Tensor = F.normalize(
                outputs.last_hidden_state[:, 0][:LOCAL_EMBEDDER_DIMENSION],
                p=2, dim=1
            )
        result: np.ndarray
        if embeddings.dtype == torch.bfloat16:
            result = embeddings.detach().to(torch.float32).cpu().numpy()
        else:
            result = embeddings.detach().cpu().numpy()

        logger.info(f"Embeddings for {texts} generated successfully")
        return result
    except Exception as e:
        logger.error(f"Error in gte_hf_embed: {str(e)}", exc_info=True)
        raise


async def gte_hf_rerank(
        query: str,
        documents: List[str],
        tokenizer: Any,
        reranker: Any,
        top_n: Optional[int] = None,
) -> List[dict[str, Any]]:
    device = next(reranker.parameters()).device
    scores = []
    minibatch_size = 4
    num_batches = math.ceil(len(documents) / minibatch_size)
    for batch_idx in range(num_batches):
        batch_start = batch_idx * minibatch_size
        batch_end = min(len(documents), batch_start + minibatch_size)
        pairs = [[query, cur_doc] for cur_doc in documents[batch_start:batch_end]]
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, return_tensors='pt').to(device)
            scores += reranker(**inputs, return_dict=True).logits.view(-1, ).float().cpu().numpy().tolist()
            del inputs
        del pairs
    reranking_result = [{'index': doc_idx, 'relevance_score': float(scores[doc_idx])} for doc_idx in
                        range(len(documents))]
    if top_n is not None:
        if top_n < len(reranking_result):
            sorted_reranking_result = sorted(reranking_result, key=lambda it: -it['relevance_score'])
            reranking_result = sorted(sorted_reranking_result[0:top_n], key=lambda it: it['index'])
            del sorted_reranking_result
    del scores
    return reranking_result


async def score_answer_relevance_to_prompt(
        prompt: str,
        answer: str,
        reranker_tokenizer: Any,
        reranker_model: Any,
        normalize: bool = True,
) -> float:
    """
    Оценивает, насколько ответ модели (answer) относится к исходному промпту (prompt)
    с помощью локального реранкера.

    :param prompt: исходный пользовательский запрос / промпт
    :param answer: ответ модели, который хотим проверить на релевантность
    :param reranker_tokenizer: токенизатор для LOCAL_RERANKER_PATH
    :param reranker_model: AutoModelForSequenceClassification для LOCAL_RERANKER_PATH
    :param normalize: если True — возвращает значение в [0,1] через сигмоиду,
                      если False — сырое значение логита из реранкера
    :return: скор релевантности (float). Чем ближе к 1, тем ответ ближе к вопросу.
             Можно использовать пороги типа:
             - > 0.7 — ответ хорошо соответствует запросу
             - 0.4–0.7 — погранично
             - < 0.4 — высокая вероятность галлюцинаций/ухода от темы
    """
    try:
        rerank_result = await gte_hf_rerank(
            query=answer,
            documents=[prompt],
            tokenizer=reranker_tokenizer,
            reranker=reranker_model,
            top_n=1,
        )
        raw_score: float = float(rerank_result[0]["relevance_score"])
        logger.debug(f"Raw reranker score for answer vs prompt: {raw_score}")

        if not normalize:
            return raw_score

        prob = 1.0 / (1.0 + math.exp(-raw_score))
        logger.debug(f"Normalized reranker score (sigmoid): {prob}")
        return prob

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

    if _reranker_tokenizer is None or _reranker_model is None:
        logger.warning(
            "is_likely_hallucination: reranker is not initialized; "
            "returning (False, 0.0)"
        )
        return False, 0.0

    try:
        score = await score_answer_relevance_to_prompt(
            prompt=original_prompt,
            answer=llm_answer,
            reranker_tokenizer=_reranker_tokenizer,
            reranker_model=_reranker_model,
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


async def initialize_rag() -> tuple[LightRAG, GTEEmbedding, BM25Okapi, list[tuple[Any, Any]]]:
    """
    Инициализирует объект LightRAG.

    Функция создаёт токенизатор и модель для эмбеддингов, а также настраивает объект LightRAG с необходимыми параметрами.

    Возвращает:
        LightRAG: Инициализированный объект LightRAG.
    """
    try:
        logger.info("Initializing RAG system...")
        logger.info(f"Loading tokenizer and embedder model: {LOCAL_EMBEDDER_PATH}...")
        emb_tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            LOCAL_EMBEDDER_PATH,
        )
        emb_model = AutoModel.from_pretrained(
            str(LOCAL_EMBEDDER_PATH),
            trust_remote_code=True,
            # device_map='cuda:0'
            device_map='cpu',
        )
        emb_model.eval()
        logger.info(f"Model {LOCAL_EMBEDDER_PATH} loaded successfully")

        logger.info(f"Loading tokenizer and reranker model: {LOCAL_RERANKER_PATH}...")
        reranker_tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_RERANKER_PATH
        )
        reranker_model = AutoModelForSequenceClassification.from_pretrained(
            str(LOCAL_RERANKER_PATH),
            trust_remote_code=True,
            device_map='cpu',
        )
        reranker_model.eval()
        logger.info(f"Reranker {LOCAL_RERANKER_PATH} loaded successfully")

        global _reranker_tokenizer, _reranker_model
        _reranker_tokenizer = reranker_tokenizer
        _reranker_model = reranker_model

        logger.info("Initializing shared data and pipeline status...")
        initialize_share_data()
        await initialize_pipeline_status()

        logger.info("Creating LightRAG instance...")
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
                embedding_dim=LOCAL_EMBEDDER_DIMENSION,
                max_token_size=LOCAL_EMBEDDER_MAX_TOKENS,
                func=lambda texts: gte_hf_embed(
                    texts,
                    tokenizer=emb_tokenizer,
                    embed_model=emb_model
                )
            ),
            rerank_model_func=partial(gte_hf_rerank, tokenizer=reranker_tokenizer, reranker=reranker_model),
            addon_params={'language': 'Russian'}
        )
        logger.info("Initializing RAG storages...")
        await rag.initialize_storages()
        logger.info("RAG system initialized successfully")
        token_cls = AutoModelForTokenClassification.from_pretrained(
            str(LOCAL_EMBEDDER_PATH), trust_remote_code=True, device_map='cpu'
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

        return rag, embedder, bm25, chunk_db
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
    words = []
    for w in wordpunct_tokenize(text):
        if w.isalnum():
            stem = _snow.stem(w.lower()).strip()
            if stem:
                words.append(stem)
    return ' '.join(words)


async def build_chunks_db_and_bm25(working_dir: Union[str, Path]):
    # coerce Path to str so callers can pass Path and code that expects str will work
    working_dir_str: str
    if isinstance(working_dir, Path):
        working_dir_str = str(working_dir)
    else:
        working_dir_str = working_dir

    chunks_path = Path(working_dir_str) / "kv_store_text_chunks.json"
    with chunks_path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    # [(content, full_doc_id)]
    chunk_db = [(raw[k]["content"], raw[k]["full_doc_id"]) for k in raw]
    norm_texts = [await tokenize_and_normalize(c[0]) for c in chunk_db]
    bm25 = BM25Okapi(norm_texts)
    return chunk_db, bm25
