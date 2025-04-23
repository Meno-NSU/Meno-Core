import json
from typing import List, Dict

import numpy as np
from nltk import wordpunct_tokenize
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class ReferenceSearcher:
    """
    Модуль для поиска релевантных URL по эмбеддингам без HTTP-запросов.
    """

    def __init__(
            self,
            urls_file: str,
            model_name: str = 'all-MiniLM-L6-v2',
            threshold: float = 0.85,
            max_links: int = 3,
            device: str = None
    ):
        self.threshold = threshold
        self.max_links = max_links
        self.model = SentenceTransformer(model_name, device=device) if device else SentenceTransformer(model_name)
        # Загрузка и нормализация ключей URL
        self._load_urls(urls_file)
        # Предварительный расчёт эмбеддингов всех ключей
        self._compute_embeddings()

    def _load_urls(self, urls_file: str):
        raw = json.load(open(urls_file, encoding='utf-8'))
        self.titles: List[str] = []  # нормализованные ключи
        self.urls_map: Dict[str, str] = {}  # key -> url
        for title, url in raw.items():
            key = ' '.join(
                filter(str.isalnum, wordpunct_tokenize(title.lower()))
            )
            self.titles.append(key)
            self.urls_map[key] = url
        logger.info(f"Загружено {len(self.titles)} ссылок из {urls_file}")

    def _compute_embeddings(self):
        # encode + normalize векторы: cosine(a,b)=dot(a,b)
        logger.info(f"Вычисление эмбеддингов для {len(self.titles)} заголовков ссылок...")
        embeds = self.model.encode(
            self.titles,
            convert_to_numpy=True,
            normalize_embeddings=True
        )  # shape (N, D)
        self.embeds = embeds
        logger.info(f"Вычислены эмбеддинги, размер: {self.embeds.shape}")

    def search(self, raw_titles: List[str]) -> List[List[str]]:
        """
        Для каждого заголовка из списка возвращает список URL,
        косинус >= self.threshold, отсортированных по убыванию сходства.
        """
        results: List[List[str]] = []
        for raw_title in raw_titles:
            logger.info(f"Поиск ссылок для заголовка: '{raw_title}'")
            # нормализуем так же, как и при загрузке
            key = ' '.join(
                filter(str.isalnum, wordpunct_tokenize(raw_title.lower()))
            )
            # эмбеддинг и нормализация
            q_emb = self.model.encode([key], convert_to_numpy=True)
            q_emb = q_emb / np.linalg.norm(q_emb, keepdims=True)
            # все косинусы сразу через матричное умножение
            sims = self.embeds @ q_emb[0]
            sorted_idxs = np.argsort(-sims)
            selected = []
            for idx in sorted_idxs:
                if sims[idx] < self.threshold:
                    break
                selected.append(self.urls_map[self.titles[idx]])
                if len(selected) >= self.max_links:
                    break
            logger.info(f"Найдено {len(selected)} ссылок для '{raw_title}': {selected}")
            results.append(selected)
        return results

    def replace_references(self, llm_answer: str) -> str:
        """
        Вырезает блок 'Ссылки:' из llm_answer и заменяет его
        реальными ссылками из ref_searcher; если не найдено —
        возвращает текст без блока ссылок.
        """
        logger.info(f"Замена ссылок в ответе: {llm_answer!r}")
        prefix = 'Ссылки:\n1. '
        idx = llm_answer.rfind(prefix)
        if idx >= 0:
            base = llm_answer[:idx].strip()
        else:
            logger.info("Нет блока 'Ссылки:'; возвращён оригинальный ответ модели.")
            return llm_answer.strip()

        raw_titles = extract_reference_titles(llm_answer)
        if not raw_titles:
            logger.info("Ни одной ссылки не было извлечено; возвращён оригинальный ответ модели.")
            return base

        urls_lists = self.search(raw_titles)
        flat = []
        for sub in urls_lists:
            for u in sub:
                if u not in flat:
                    flat.append(u)
        capped = flat[:self.max_links]
        if not capped:
            logger.info("Ни одной ссылки не было найдено; возвращён оригинальный ответ модели.")
            return base

        result = base + "\n\nПолезные ссылки:"
        for u in capped:
            result += f"\n- {u}"
        return result


def extract_reference_titles(llm_answer: str) -> List[str]:
    """
    Извлекает из текста секцию с префиксом 'Ссылки:\\n1. ' и возвращает список заголовков.
    Если блок не найден — возвращает [llm_answer.strip()].
    """
    prefix = 'Ссылки:\n1. '
    idx = llm_answer.rfind(prefix)
    if idx < 0:
        # не нашли блок ссылок — возвращаем весь ответ
        full = llm_answer.strip()
        return [full] if full else []

    # отрезаем всё до 'Ссылки:\n1. '
    tail = llm_answer[idx + len(prefix):].strip()
    titles: List[str] = []
    counter = 1

    while True:
        next_marker = f'\n{counter + 1}. '
        next_pos = tail.find(next_marker)
        if next_pos < 0:
            # последняя строка
            titles.append(tail.strip())
            break
        # вырезаем заголовок до маркера
        titles.append(tail[:next_pos].strip())
        # обрезаем уже прочитанный кусок вместе с номером
        tail = tail[next_pos + len(next_marker):]
        counter += 1

    return titles

# ===== Пример интеграции в backend_api.py =====
#
# from reference_search import ReferenceSearcher
#
# ref_searcher: ReferenceSearcher = None
#
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global ref_searcher
#     # ... предыдущая инициализация rag, аббревиатур и т.д.
#     ref_searcher = ReferenceSearcher(URLS_FNAME, model_name='all-MiniLM-L6-v2', threshold=0.75)
#
#     yield
#
# @app.post("/chat", response_model=ChatResponse)
# async def chat(request: ChatRequest):
#     # ... до запроса к RAG
#     response_text = await rag_instance.aquery(...)
#     # извлекаем заголовки ссылок из текста LLM
#     titles = extract_reference_titles(response_text)
#     # получаем URL-списки
#     urls_lists = ref_searcher.search(titles)
#     # дальше форматируем ответ, объединяя URL
#     # ...
