import os
from typing import List

from nltk import SnowballStemmer, wordpunct_tokenize

from config import settings
from lightrag.lightrag.utils import EmbeddingFunc
from lightrag.lightrag import LightRAG, QueryParam
from lightrag.lightrag.llm.openai import openai_complete_if_cache
from lightrag.lightrag.kg.shared_storage import initialize_pipeline_status, initialize_share_data
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
import openai
import asyncio

TEMPERATURE = 0.3
QUERY_MAX_TOKENS = 4000
TOP_K = 30
WORKING_DIR = 'light_rag_db_v2'
print(f'os.path.isdir({WORKING_DIR}) = {os.path.isdir(WORKING_DIR)}')
ABBREVIATIONS_FNAME = 'full_abbreviations_updated.json'
print(f'os.path.isfile({ABBREVIATIONS_FNAME}) = {os.path.isfile(ABBREVIATIONS_FNAME)}')
LOCAL_EMBEDDER_DIMENSION = 768
LOCAL_EMBEDDER_MAX_TOKENS = 4096
LOCAL_EMBEDDER_NAME = 'gte-multilingual-base'
print(f'os.path.isdir({LOCAL_EMBEDDER_NAME}) = {os.path.isdir(LOCAL_EMBEDDER_NAME)}')
ENCODER = AutoTokenizer.from_pretrained(LOCAL_EMBEDDER_NAME)
SYSTEM_PROMPT_FOR_MENO = '''Ты - Менон, разработанный Иваном Бондаренко. Ты - дружелюбный ассистент, разговаривающий на русском языке и отвечающий на вопросы пользователей о Новосибирском государственном университете (НГУ) и Новосибирском Академгородке. Ты очень любишь Новосибирский государственный университет и поэтому стремишься заинтересовать разные категории своих пользователей: абитуриентов поступлением в университет, студентов - учёбой, а учёных и преподавателей - работой в нём. Твои ответы должны быть максимально точными, понятными и подробными. Если же ты понимаешь, что твоей базы знаний недостаточно для точного ответа, ты должен сообщить, что тебе не хватает знаний для компетентного ответа, и предложить уважаемому пользователю задать какой-либо другой вопрос. Если пользователь спрашивает информацию про этот год или этот месяц (в общем, про информацию, зависящую от времени), то отвечай так, как-будто сейчас март 2025 года.
---История общения с пользователем---
{history}

---База знаний---
{context_data}

---Правила ответа---

- Ожидаемый формат и длина: {response_type}
'''

TEMPLATE_FOR_ABBREVIATION_EXPLAINING = '''Отредактируйте, пожалуйста, текст пользовательского вопроса так, чтобы этот вопрос стал более простым и понятным для обычных людей от юных старшеклассников до пожилых мужчин и женщин. При этом не надо, пожалуйста, применять markdown или иной вид гипертекста. Главное, на что вам надо обратить внимание и по возможности исправить - это логика изложения и понятность формулировок вопроса. Ничего не объясняйте и не комментируйте своё решение, просто перепишите текст вопроса.

Также исправьте грамматические ошибки в тексте вопроса, если они там есть. Кроме того, если вы обнаружите аббревиатуры в тексте этого вопроса, то замените все обнаруженные аббревиатуры их корректными расшифровками, сохранив морфологическую и синтаксическую согласованность. Вот здесь вы можете ознакомиться с JSON-словарём, описывающим возможные аббревиатуры и их расшифровки:

```json
{abbreviations_dict}
```

Далее приведён текст вопроса, нуждающийся в возможном улучшении:

```text
{text_of_question}
```'''

SYSTEM_PROMPT_FOR_ANAPHORA_RESOLUTION = 'Проанализируй диалог человека с большой языковой моделью и переделай последнюю реплику человека так, чтобы снять все ситуации местоименной анафоры в этом вопросе. Учитывай при этом всю историю диалога этого человека с большой языковой моделью. Не отвечай на вопрос человека, а просто перепиши его.'
FEWSHOTS_FOR_ANAPHORA = [
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


# ---------- LLM wrapper ----------
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages += history_messages
    messages.append({"role": "user", "content": prompt})

    client = openai.AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url
    )

    completion = await client.chat.completions.create(
        model=settings.llm_model_name,
        messages=messages,
        temperature=TEMPERATURE,
    )
    return completion.choices[0].message.content


async def explain_abbreviations(question: str, abbreviations: dict) -> str:
    snow_stemmer = SnowballStemmer(language='russian')
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
        return question
    user_prompt = TEMPLATE_FOR_ABBREVIATION_EXPLAINING.format(
        abbreviations_dict=filtered_abbreviations,
        text_of_question=question
    )
    try:
        new_improved_question = await openai_complete_if_cache(
            settings.llm_model_name,
            user_prompt,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=TEMPERATURE
        )
    except:
        new_improved_question = question
    return new_improved_question


async def resolve_anaphora(question: str, history: list) -> str:
    if (len(history) == 0) or (len(question.strip()) == 0):
        return question
    if (len(history) % 2) != 0:
        raise RuntimeError(f'The dialogue history length is wrong! Expected an even number, got {len(history)}.')
    expected_roles = ['user', 'assistant']
    for _ in range((len(history) // 2) - 1):
        expected_roles += ['user', 'assistant']
    history_roles = [it['role'] for it in history]
    if history_roles != expected_roles:
        raise RuntimeError(f'The dialogue history roles are wrong! Expected {expected_roles}, got {history_roles}.')
    if len(history) > 6:
        history_ = history[-6:]
    else:
        history_ = history
    user_prompt = f'Человек: {" ".join(history_[0]["content"].split()).strip()}'
    user_prompt += f'\nБольшая языковая модель: : {" ".join(history_[1]["content"].split()).strip()}'
    for val in history_[2:]:
        if val['role'] == 'user':
            user_prompt += '\nЧеловек: '
        else:
            user_prompt += '\nБольшая языковая модель: '
        user_prompt += ' '.join(val['content'].split()).strip()
    del history_
    user_prompt += '\nЧеловек: ' + ' '.join(question.split()).strip()
    try:
        question_without_anaphora = await openai_complete_if_cache(
            settings.llm_model_name,
            user_prompt,
            system_prompt=SYSTEM_PROMPT_FOR_ANAPHORA_RESOLUTION,
            history_messages=FEWSHOTS_FOR_ANAPHORA,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=TEMPERATURE
        )
    except:
        question_without_anaphora = question
    return question_without_anaphora


# ---------- Embedding function ----------
async def gte_hf_embed(texts: List[str], tokenizer, embed_model) -> np.ndarray:
    device = next(embed_model.parameters()).device
    encoded_texts = tokenizer(
        texts, return_tensors='pt', padding=True, truncation=True
    ).to(device)
    batch_dict = tokenizer(
        texts, return_tensors='pt',
        max_length=LOCAL_EMBEDDER_MAX_TOKENS, padding=True, truncation=True,
    ).to(device)
    with torch.no_grad():
        outputs = embed_model(**batch_dict)
        embeddings = F.normalize(
            outputs.last_hidden_state[:, 0][:LOCAL_EMBEDDER_DIMENSION],
            p=2, dim=1
        )
    if embeddings.dtype == torch.bfloat16:
        return embeddings.detach().to(torch.float32).cpu().numpy()
    else:
        return embeddings.detach().cpu().numpy()


# ---------- RAG initialization ----------
async def initialize_rag():
    emb_tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_EMBEDDER_NAME
    )
    ENCODER = emb_tokenizer
    emb_model = AutoModel.from_pretrained(
        LOCAL_EMBEDDER_NAME,
        trust_remote_code=True,
        # device_map='cuda:0'
        device_map='cpu'
    )
    emb_model.eval()

    initialize_share_data()
    await initialize_pipeline_status()
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        cosine_better_than_threshold=0.1,
        embedding_func=EmbeddingFunc(
            embedding_dim=LOCAL_EMBEDDER_DIMENSION,
            max_token_size=LOCAL_EMBEDDER_MAX_TOKENS,
            func=lambda texts: gte_hf_embed(
                texts,
                tokenizer=emb_tokenizer,
                embed_model=emb_model
            )
        ),
        addon_params={'language': 'Russian'}
    )
    await rag.initialize_storages()

    return rag
