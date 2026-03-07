import json
import logging
import uuid
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, Dict, Union

import torch
import torch.nn.functional as F
import zvec
from rank_bm25 import BM25Okapi

from meno_core.config.settings import settings
from meno_core.core.gte_embedding import GTEEmbedding
from meno_core.core.rag_engine import (
    generate_with_llm,
    _snow,
    _reranker_tokenizer,
    _reranker_model,
)

logger = logging.getLogger(__name__)

from meno_core.core.prompts import (
    SEARCH_SYSTEM_PROMPT,
    SEARCH_USER_PROMPT,
    MEANINGLESS_REQUEST_ANSWER,
    EXAMPLES_OF_SEARCH_QUERIES,
)

LEXICAL_DISTANCE_THRESHOLD: float = 0.05
MAX_NUMBER_OF_ANSWER_VARIANTS: int = 3

class ZvecRAGEngine:
    def __init__(
            self,
            working_dir: Union[str, Path],
            embedder: GTEEmbedding,
            bm25: BM25Okapi, 
            chunk_db: list[tuple[Any, Any]],
    ):
        self.working_dir = Path(working_dir)
        self.embedder = embedder
        self.bm25 = bm25
        self.chunk_db = chunk_db
        self.zvec_path = self.working_dir / "zvec_index"
        
        # We will initialize zvec collection inline here or load it.
        self._init_zvec_collection()
        
    def _init_zvec_collection(self):
        schema = zvec.CollectionSchema(
            name="meno_zvec",
            vectors=zvec.VectorSchema(
                "embedding", 
                zvec.DataType.VECTOR_FP32, 
                settings.embedder_dim
            ),
        )
        
        need_build = not (self.zvec_path.exists() and any(self.zvec_path.iterdir()))
        
        if need_build:
            logger.info(f"Building ZVEC index at {self.zvec_path}...")
            os.makedirs(self.zvec_path, exist_ok=True)
            self.collection = zvec.create_and_open(path=str(self.zvec_path), schema=schema)
            # Encode and insert chunks
            batch_size = 32
            for i in range(0, len(self.chunk_db), batch_size):
                batch_chunks = self.chunk_db[i:i+batch_size]
                texts = [c[0] for c in batch_chunks]
                embeddings_res = self.embedder.encode(texts, return_dense=True, return_sparse=False)
                embeddings = embeddings_res["dense_embeddings"].detach().cpu().numpy().tolist()
                
                docs = []
                for j, (text, full_doc_id) in enumerate(batch_chunks):
                    # We use i+j as our internal index so we can map back to chunk_db later.
                    doc_id = str(i + j)
                    docs.append(zvec.Doc(id=doc_id, vectors={"embedding": embeddings[j]}))
                
                self.collection.insert(docs)
                logger.debug(f"Inserted batch {i//batch_size} of {len(self.chunk_db)//batch_size}")
            logger.info("ZVEC index build complete.")
        else:
            logger.info(f"Loading existing ZVEC index from {self.zvec_path}")
            # Ensure it is created before opening
            temp_col = zvec.create_and_open(path=str(self.zvec_path), schema=schema)
            self.collection = temp_col

    async def aclear_cache(self):
        """Mock method to be compatible with main.py calls"""
        pass

    async def aquery(self, query: str, param=None, system_prompt: Optional[str] = None):
        """
        Mimics LightRAG pipeline using custom ZVEC workflow as described by the user.
        """
        # Step 1: Generate search queries
        logger.info(f"Expanding query: {query}")
        # Note: in a real implementation we might pass LLM tokenizer specifically, but here we can 
        # use generate_with_llm directly using standard text replacements just like in rag_engine.py
        
        prompt_for_search = self._prepare_search_prompt(query)
        search_queries_str = await generate_with_llm(prompt=prompt_for_search)
        
        list_of_search_queries = list(set(filter(
            lambda it2: len(it2) > 0,
            map(lambda it1: it1.strip(), search_queries_str.split('\n'))
        )))
        
        is_meaningless = False
        for cur_query in list_of_search_queries:
            dist = self._calculate_dist(MEANINGLESS_REQUEST_ANSWER, cur_query)
            if dist <= LEXICAL_DISTANCE_THRESHOLD:
                is_meaningless = True
                break
                
        if is_meaningless or not list_of_search_queries:
            return "К сожалению, не удалось сформулировать поисковые запросы для этого вопроса."
            
        logger.info(f"Generated search queries: {list_of_search_queries}")
        
        # Step 2: Retrieve chunks
        union_of_relevant_indices = {}
        top_k = getattr(param, 'chunk_top_k', settings.chunk_top_k) if param else settings.chunk_top_k
        
        for cur_search_query in list_of_search_queries + [query]:
            # ZVEC Search
            question_vector_res = self.embedder.encode([cur_search_query], return_dense=True, return_sparse=False)
            q_vec = question_vector_res["dense_embeddings"][0].detach().cpu().numpy().tolist()
            
            zvec_results = self.collection.query(
                zvec.VectorQuery("embedding", vector=q_vec), 
                topk=top_k
            )
            zvec_ids = [int(r['id']) for r in zvec_results]
            
            # BM25 Search
            bm25_q = self._prepare_text_for_bm25(cur_search_query)
            chunk_scores = self.bm25.get_scores(bm25_q)
            bm25_ids = list(range(len(chunk_scores)))
            ordered_bm25 = sorted(zip(bm25_ids, chunk_scores), key=lambda it: (-it[1], it[0]))[:top_k]
            bm25_ids = [it[0] for it in ordered_bm25]
            
            united_ids = set(zvec_ids) | set(bm25_ids)
            
            # Step 3: Rerank
            if united_ids:
                reranked_chunks = await self._rerank(cur_search_query, list(united_ids), top_k)
                for chunk_idx, score in reranked_chunks.items():
                    if chunk_idx in union_of_relevant_indices:
                        union_of_relevant_indices[chunk_idx] = max(union_of_relevant_indices[chunk_idx], score)
                    else:
                        union_of_relevant_indices[chunk_idx] = score

                        
        if not union_of_relevant_indices:
            return "К сожалению, в базе данных недостаточно информации для ответа на этот вопрос."
            
        # Step 4: Prep Answer Context
        # Select best overall chunks
        best_chunks_idx = sorted(union_of_relevant_indices.keys(), key=lambda k: -union_of_relevant_indices[k])[:settings.chunk_top_k]
        
        context_str = ""
        for i, idx in enumerate(best_chunks_idx):
            content, doc_id = self.chunk_db[idx]
            context_str += f"[Фрагмент {i+1} из {doc_id}]: {content}\n\n"
            
        qa_prompt = f"Контекст:\n{context_str}\n\nВопрос:\n{query}"
        
        # Step 5: Generation
        logger.info("Generating final answer with Zvec RAG...")
        stream = getattr(param, 'stream', False) if param else False
        history_messages = getattr(param, 'conversation_history', []) if param else []
        
        # Just use generation with LightRAG's wrapper utility
        answer = await generate_with_llm(
            prompt=qa_prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            stream=stream
        )
        
        return answer

    def _prepare_search_prompt(self, query: str) -> str:
        prompt = SEARCH_SYSTEM_PROMPT + "\n\n"
        for ex in EXAMPLES_OF_SEARCH_QUERIES:
            prompt += f"Пользователь: {ex['question']}\nАгент:\n" + "\n".join(ex['search_queries']) + "\n\n"
        prompt += f"Пользователь: {query}\nАгент:\n"
        return prompt

    def _calculate_dist(self, t1: str, t2: str) -> float:
        # Simple distance placeholder. In full implementation, import cer from jiwer.
        # But for Meno-Core we can reuse the lexical distance function if imported.
        # Here we just check equality to avoid external deps if not strictly needed.
        return 0.0 if t1.strip().lower() == t2.strip().lower() else 1.0
        
    def _prepare_text_for_bm25(self, text: str) -> List[str]:
        words = []
        for w in text.split():
            if w.isalnum():
                words.append(_snow.stem(w.lower()).strip())
        return words

    async def _rerank(self, query: str, chunk_ids: List[int], top_n: int) -> Dict[int, float]:
        if _reranker_model is None or _reranker_tokenizer is None:
            return {i: 1.0 for i in chunk_ids[:top_n]}
            
        docs = [self.chunk_db[i][0] for i in chunk_ids]
        
        import torch
        device = next(_reranker_model.parameters()).device
        scores = []
        minibatch_size = 4
        import math
        num_batches = math.ceil(len(docs) / minibatch_size)
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * minibatch_size
            batch_end = min(len(docs), batch_start + minibatch_size)
            pairs = [[query, d] for d in docs[batch_start:batch_end]]
            
            with torch.no_grad():
                inputs = _reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
                scores += _reranker_model(**inputs, return_dict=True).logits.view(-1, ).float().cpu().numpy().tolist()
                
        scored_docs = list(zip(chunk_ids, scores))
        scored_docs.sort(key=lambda x: -x[1])
        
        return {id_: score for id_, score in scored_docs[:top_n]}
