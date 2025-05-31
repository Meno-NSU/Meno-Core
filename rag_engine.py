import logging
import os
from typing import List

import numpy as np
import openai
import torch
import torch.nn.functional as F
from nltk import SnowballStemmer, wordpunct_tokenize
from transformers import AutoTokenizer, AutoModel

from config import settings
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status, initialize_share_data
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc

TEMPERATURE = 0.3
QUERY_MAX_TOKENS = 4000
TOP_K = 30
WORKING_DIR = settings.working_dir
print(f'os.path.isdir({WORKING_DIR}) = {os.path.isdir(WORKING_DIR)}')
ABBREVIATIONS_FNAME = settings.abbreviations_file
print(f'os.path.isfile({ABBREVIATIONS_FNAME}) = {os.path.isfile(ABBREVIATIONS_FNAME)}')
URLS_FNAME = 'resources/validated_urls.json'
print(f'os.path.isfile({URLS_FNAME}) = {os.path.isfile(URLS_FNAME)}')
LOCAL_EMBEDDER_DIMENSION = 768
LOCAL_EMBEDDER_MAX_TOKENS = 4096
LOCAL_EMBEDDER_NAME = settings.local_embedder_path
print(f'os.path.isdir({LOCAL_EMBEDDER_NAME}) = {os.path.isdir(LOCAL_EMBEDDER_NAME)}')
ENCODER = AutoTokenizer.from_pretrained(LOCAL_EMBEDDER_NAME)
SYSTEM_PROMPT_FOR_MENO = """---Role---

Вы - Менон, разработанный Иваном Бондаренко, научным сотрудником Новосибирского государственного университета (НГУ). Вас разработали в лаборатории прикладных цифровых технологий НГУ, где, собственно, и работает Иван Бондаренко. Вы - дружелюбный ассистент, разговаривающий на русском языке и отвечающий на вопросы пользователей о Новосибирском государственном университете (НГУ) и Новосибирском Академгородке. Вы очень любите Новосибирский государственный университет и поэтому стремитесь заинтересовать разные категории своих пользователей: абитуриентов поступлением в университет, студентов - учёбой, а учёных и преподавателей - работой в нём.

---Goal---

Сформируйте краткий ответ на основе фрагментов документа (Document Chunks) и следуйте правилам ответа (Response Rules), учитывая как историю обсуждения (Conversation History), так и текущий запрос. Обобщите всю информацию, содержащуюся в предоставленных фрагментах документа (Document Chunks), и включите ваши общие знания, относящиеся к этим фрагментам документа (Document Chunks). Не включайте информацию, не указанную в фрагментах документа (Document Chunks).

При работе с контентом с временными метками:
1. Каждый элемент контента имеет временную метку "created_at", указывающую, когда мы приобрели эти знания.
2. При столкновении с противоречивой информацией учитывайте как контент, так и временную метку.
3. Не следует автоматически предпочитать самый последний контент - используйте суждение на основе контекста.
4. Для запросов, связанных со временем, приоритизируйте временную информацию в контенте перед учетом временных меток создания.
5. Считайте, что сейчас - конец марта 2025 года.

---Conversation History---
{history}

---Document Chunks---
{content_data}

---Response Rules---

- Целевой формат и длина: {response_type}
- Не используйте форматирование markdown.
- Пожалуйста, отвечайте на русском языке.
- Обращайтесь к пользователю исключительно на "вы".
- Убедитесь, что ответ сохраняет преемственность с историей разговора (Conversation History).
- Перечислите до 5 самых важных источников информации в конце в разделе "Ссылки".
- Если вы не знаете ответа, просто скажите об этом.
- Не включайте информацию, не представленную во фрагментах документа (Document Chunks).
- Если пользователь пишет что-то о политике, религии, национальностях, наркотиках, криминале или пишет просто оскорбительный или токсичный текст в адрес какого-то человека или университета, вежливо и непреклонно откажитесь от разговора и предложите сменить тему.
"""

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

# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('meno_errors.log'),
#         logging.StreamHandler()
#     ]
# )
logger = logging.getLogger(__name__)

lightrag_logger = logging.getLogger('lightrag')
lightrag_logger.setLevel(logging.WARNING)  # или logging.DEBUG для подробных логов

# Добавьте обработчик для вывода в консоль
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
lightrag_logger.addHandler(handler)


# ---------- LLM wrapper ----------
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
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
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages += history_messages
        messages.append({"role": "user", "content": prompt})

        logger.info(f"Sending request to LLM with {len(messages)} messages, prompt: {prompt}")

        client = openai.AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url
        )

        completion = await client.chat.completions.create(
            model=settings.llm_model_name,
            messages=messages,
            temperature=TEMPERATURE,
        )
        response = completion.choices[0].message.content
        logger.info(f"Received response from LLM, length: {len(response)}")
        return response
    except Exception as e:
        logger.error(f"Error in llm_model_func: {str(e)}", exc_info=True)
        raise


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
            logger.debug("No abbreviations found in question")
            return question

        logger.debug(f"Found abbreviations: {filtered_abbreviations}")

        user_prompt = TEMPLATE_FOR_ABBREVIATION_EXPLAINING.format(
            abbreviations_dict=filtered_abbreviations,
            text_of_question=question
        )
        new_improved_question = await openai_complete_if_cache(
            settings.llm_model_name,
            user_prompt,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=TEMPERATURE
        )
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
        question_without_anaphora = await openai_complete_if_cache(
            settings.llm_model_name,
            user_prompt,
            system_prompt=SYSTEM_PROMPT_FOR_ANAPHORA_RESOLUTION,
            history_messages=FEWSHOTS_FOR_ANAPHORA,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=TEMPERATURE
        )
        logger.debug(f"Question after anaphora resolution: {question_without_anaphora}")
        return question_without_anaphora
    except Exception as e:
        logger.error(f"Error in resolve_anaphora: {str(e)}", exc_info=True)
        return question


# ---------- Embedding function ----------
async def gte_hf_embed(texts: List[str], tokenizer, embed_model) -> np.ndarray:
    try:
        device = next(embed_model.parameters()).device
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
            result = embeddings.detach().to(torch.float32).cpu().numpy()
        else:
            result = embeddings.detach().cpu().numpy()

        logger.info(f"Embeddings for {texts} generated successfully")
        return result
    except Exception as e:
        logger.error(f"Error in gte_hf_embed: {str(e)}", exc_info=True)
        raise


async def initialize_rag() -> LightRAG:
    """
    Инициализирует объект LightRAG.

    Функция создаёт токенизатор и модель для эмбеддингов, а также настраивает объект LightRAG с необходимыми параметрами.

    Возвращает:
        LightRAG: Инициализированный объект LightRAG.
    """
    try:
        logger.info("Initializing RAG system")
        logger.info("Loading tokenizer and embedder model...")
        emb_tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            LOCAL_EMBEDDER_NAME
        )
        ENCODER: AutoTokenizer = emb_tokenizer
        emb_model: AutoModel = AutoModel.from_pretrained(
            LOCAL_EMBEDDER_NAME,
            trust_remote_code=True,
            # device_map='cuda:0'
            device_map='cpu'
        )
        emb_model.eval()
        logger.info("Model loaded successfully")

        logger.info("Initializing shared data and pipeline status...")
        initialize_share_data()
        await initialize_pipeline_status()

        logger.info("Creating LightRAG instance...")
        rag: LightRAG = LightRAG(
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
        logger.info("Initializing RAG storages...")
        await rag.initialize_storages()
        logger.info("RAG system initialized successfully")

        return rag
    except Exception as e:
        logger.error(f"Error initializing RAG: {str(e)}", exc_info=True)
        raise
