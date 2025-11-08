# import json
# import logging
# from typing import List, Dict, Tuple
# import re
#
# import numpy as np
# import torch
# from nltk import wordpunct_tokenize
# from transformers import AutoTokenizer, AutoModel
# import torch.nn.functional as F
#
# logger = logging.getLogger(__name__)
#
#
# # Pattern to match the title line, allowing for markdown variations
# _TITLE_PATTERN = re.compile(r'.*Ссылки.*:')
# # Pattern to match list items starting with -, *, +, or numbers (e.g., 1.)
# _MARKER_PATTERN = re.compile(r'^\s*([-*+]|\d+\.)\s*')
#
#
# class ReferenceSearcher:
#     """
#     Модуль для поиска релевантных URL по эмбеддингам без HTTP-запросов.
#     """
#
#     def __init__(
#             self,
#             urls_file: str,
#             model_name: str = 'all-MiniLM-L6-v2',
#             threshold: float = 0.85,
#             max_links: int = 3,
#             device: str = None
#     ):
#         self.threshold = threshold
#         self.max_links = max_links
#         self.device = device
#         logger.info(
#             f"Initializing AutoModel(model_name={model_name}, device={device}) for reference search...")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name,
#                                                        trust_remote_code=True,
#                                                        # device_map='cuda:0'
#                                                        device_map='cpu',
#                                                        local_files_only=True)
#         self.model = AutoModel.from_pretrained(model_name,
#                                                trust_remote_code=True,
#                                                # device_map='cuda:0'
#                                                device_map='cpu',
#                                                local_files_only=True).to(device)
#         self.model.eval()
#         # self.model = SentenceTransformer(model_name, device=device) if device else SentenceTransformer(model_name)
#         # Загрузка и нормализация ключей URL
#         self._load_urls(urls_file)
#         # Предварительный расчёт эмбеддингов всех ключей
#         self._compute_embeddings()
#
#     def _load_urls(self, urls_file: str):
#         raw = json.load(open(urls_file, encoding='utf-8'))
#         self.titles: List[str] = []  # нормализованные ключи
#         self.urls_map: Dict[str, str] = {}  # key -> url
#         for title, url in raw.items():
#             key = ' '.join(
#                 filter(str.isalnum, wordpunct_tokenize(title.lower()))
#             )
#             self.titles.append(key)
#             self.urls_map[key] = url
#         logger.info(f"Загружено {len(self.titles)} ссылок из {urls_file}")
#
#     def _compute_embeddings(self):
#         logger.info(
#             f"Вычисление эмбеддингов для {len(self.titles)} заголовков ссылок...")
#
#         # Токенизация и получение эмбеддингов
#         with torch.no_grad():
#             inputs = self.tokenizer(
#                 self.titles,
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt"
#             ).to(self.device)
#
#             outputs = self.model(**inputs)
#             # Используем mean pooling для получения эмбеддингов предложений
#             embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
#             # Нормализация
#             embeddings = F.normalize(embeddings, p=2, dim=1)
#
#         self.embeds = embeddings.cpu().numpy()
#         logger.info(f"Вычислены эмбеддинги, размер: {self.embeds.shape}")
#
#     def _mean_pooling(self, model_output, attention_mask):
#         token_embeddings = model_output.last_hidden_state
#         input_mask_expanded = (
#             attention_mask
#             .unsqueeze(-1)
#             .expand(token_embeddings.size())
#             .float()
#         )
#         return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
#             input_mask_expanded.sum(1), min=1e-9)
#
#     def _embed_query(self, raw_title: str) -> Tuple[np.ndarray, str]:
#         """Возвращает нормализованный эмбеддинг и ключ"""
#         key = ' '.join(
#             filter(str.isalnum, wordpunct_tokenize(raw_title.lower())))
#
#         with torch.no_grad():
#             inputs = self.tokenizer(
#                 [key],
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt"
#             ).to(self.device)
#
#             outputs = self.model(**inputs)
#             # Mean pooling и нормализация
#             q_emb = self._mean_pooling(outputs, inputs['attention_mask'])
#             q_emb = F.normalize(q_emb, p=2, dim=1)
#
#         return q_emb[0].cpu().numpy(), key
#
#     def get_top_links(self, raw_titles: List[str]) -> List[str]:
#         """
#         Для списка заголовков возвращает до self.max_links наиболее релевантных URL
#         по глобальному ранжированию косинусного сходства.
#         """
#         scored: List[Tuple[str, float]] = []
#         for raw_title in raw_titles:
#             logger.info(f"Scoring links for title: '{raw_title}'")
#             q_emb, key = self._embed_query(raw_title)
#             sims = self.embeds @ q_emb
#             # собираем только те, что выше порога
#             for idx, score in enumerate(sims):
#                 if score >= self.threshold:
#                     url = self.urls_map[self.titles[idx]]
#                     scored.append((url, float(score)))
#
#         if not scored:
#             return []
#
#         # сортируем по убыванию score, убираем дубликаты, сохраняя порядок
#         scored.sort(key=lambda x: -x[1])
#         seen = set()
#         top_urls = []
#         for url, score in scored:
#             if url not in seen:
#                 seen.add(url)
#                 top_urls.append(url)
#             if len(top_urls) >= self.max_links:
#                 break
#
#         logger.info(f"Top {self.max_links} URLs: {top_urls}")
#         return top_urls
#
#     def search(self, raw_titles: List[str]) -> List[List[str]]:
#         """
#         Для каждого заголовка из списка возвращает список URL,
#         косинус >= self.threshold, отсортированных по убыванию сходства.
#         """
#         results: List[List[str]] = []
#         for raw_title in raw_titles:
#             logger.info(f"Поиск ссылок для заголовка: '{raw_title}'")
#             # нормализуем так же, как и при загрузке
#             key = ' '.join(
#                 filter(str.isalnum, wordpunct_tokenize(raw_title.lower()))
#             )
#             # эмбеддинг и нормализация
#             q_emb = self.model.encode([key], convert_to_numpy=True)
#             q_emb /= np.linalg.norm(q_emb, keepdims=True)
#             # все косинусы сразу через матричное умножение
#             sims = self.embeds @ q_emb[0]
#             sorted_idxs = np.argsort(-sims)
#             selected = []
#             for idx in sorted_idxs:
#                 if sims[idx] < self.threshold:
#                     break
#                 selected.append(self.urls_map[self.titles[idx]])
#                 if len(selected) >= self.max_links:
#                     break
#             logger.info(
#                 f"Найдено {len(selected)} ссылок для '{raw_title}': {selected}")
#             results.append(selected)
#         return results
#
#     def replace_references(self, llm_answer: str) -> str:
#         """
#         Вырезает блок 'Ссылки:' из llm_answer и заменяет его
#         реальными top_links; если не найдено — возвращает текст без блока ссылок.
#         """
#         logger.info(f"replace_references called with answer: {llm_answer!r}")
#
#         raw_titles, cleaned_text = extract_and_clean(llm_answer)
#         logger.info(f"Extracted reference titles: {raw_titles}")
#         top_urls = self.get_top_links(raw_titles)
#         if not top_urls:
#             return cleaned_text
#
#         result = cleaned_text + "\n\nПолезные ссылки:"
#         for u in top_urls:
#             result += f"\n- {u}"
#         return result
#
#
# def extract_and_clean(text: str) -> Tuple[List[str], str]:
#     """
#     Достает все пункты из списка под секцией 'Ссылки:' и убирает всю секцию из текста. Учитывает спец. символы и пустые строки
#     """
#
#     # Split text into lines
#     lines = text.splitlines()
#
#     # Search for the last occurrence of the title from the end
#     for i in range(len(lines) - 1, -1, -1):
#         if _TITLE_PATTERN.search(lines[i]):
#             start = i
#             extracted_items = []
#             j = i + 1
#
#             # Extract list items following the title
#             is_empty_line = False
#             while j < len(lines) and (_MARKER_PATTERN.match(lines[j]) or (is_empty_line := lines[j].strip() == '')):
#                 if is_empty_line:
#                     j += 1
#                     continue
#                 is_empty_line = False
#                 match = _MARKER_PATTERN.match(lines[j])
#                 item_text = lines[j][match.end():].strip()
#                 extracted_items.append(item_text)
#                 j += 1
#
#             # Remove the section (title and list items)
#             cleaned_lines = lines[:start] + lines[j:]
#             cleaned_text = '\n'.join(cleaned_lines)
#             return extracted_items, cleaned_text
#
#     # If no title is found, return empty list and original text
#     return [], text
