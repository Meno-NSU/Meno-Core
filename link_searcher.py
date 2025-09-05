from pathlib import Path
import json
import numpy as np
from typing import List, Tuple
from lightrag import LightRAG
from rag_engine import _tokenize_and_normalize
import logging

class LinkSearcher:
    def __init__(
        self,
        urls_path: Path | str,
        lightrag_instance: LightRAG,
        top_k: int,
        dist_threshold: float,  # больше не используется, но оставим для совместимости
        max_links: int = 5,
        embedder=None,
        bm25=None,
        chunk_db: List[Tuple[str, str]] | None = None,
        dense_weight: float = 1.0,
        sparse_weight: float = 0.1,
        hybrid_similarity_threshold: float = 3.5,
        per_chunk_top_k: int | None = None,
        logger=None

    ):
        """
        urls_path: JSON {header -> url}
        chunk_db: список (chunk_text, full_doc_id)
        """
        self.lightrag_instance = lightrag_instance
        self.top_k = top_k
        self.max_links = max_links
        self.embedder = embedder
        self.bm25 = bm25
        self.chunk_db = chunk_db or []
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.hybrid_threshold = hybrid_similarity_threshold
        self.per_chunk_top_k = per_chunk_top_k  # если None — подберём от top_k
        self.logger = logger or logging.getLogger("links")

        self.logger.debug(
            "LinkSearcher init: top_k=%s, max_links=%s, dense_weight=%.3f, sparse_weight=%.3f, "
            "hybrid_threshold=%.3f, per_chunk_top_k=%s, chunks=%d",
            self.top_k, self.max_links, self.dense_weight, self.sparse_weight,
            self.hybrid_threshold, self.per_chunk_top_k, len(self.chunk_db)
        )

        urls_path = Path(urls_path)
        with urls_path.open(mode='r', encoding='utf-8') as fp:
            self.header2url = json.load(fp)

        # doc_id -> url (строим по заголовкам чанков)
        self.docid2url = {}
        for content, full_doc_id in self.chunk_db:
            header = content.split("\n", 1)[0]
            url = self.header2url.get(header)
            if url and full_doc_id not in self.docid2url:
                self.docid2url[full_doc_id.split("_chunk")[0]] = url

    def _prepare_doc_id(self, full_doc_id: str) -> str:
        i = full_doc_id.find("_chunk")
        return full_doc_id[:i] if i > 0 else full_doc_id

    async def _bm25_candidates(self, text: str) -> List[int]:
        if not self.bm25 or not self.chunk_db:
            self.logger.warning("BM25 or chunk_db is not initialized")
            return []
        norm = await _tokenize_and_normalize(text)
        scores = self.bm25.get_scores(norm.split())
        k = min(max(20, self.top_k), len(scores))
        idxs = np.argsort(-scores)[:k].tolist()
    
        # логируем топ-10
        top_log = idxs[:10]
        preview = []
        for i in top_log:
            content, full_doc_id = self.chunk_db[i]
            header = content.split("\n", 1)[0][:120]
            preview.append({"idx": i, "bm25": float(scores[i]), "doc": full_doc_id, "header": header})
        self.logger.debug("BM25: picked %d candidates (k=%d). Top preview: %s", len(idxs), k, preview)
        return idxs


    async def _hybrid_rerank(self, source_text: str, candidate_idxs: List[int]) -> List[Tuple[int, float]]:
        if not candidate_idxs:
            self.logger.debug("Hybrid rerank: no candidates")
            return []
        pairs = [(source_text, self.chunk_db[i][0]) for i in candidate_idxs]
        scores = self.embedder.compute_scores(pairs, dense_weight=self.dense_weight, sparse_weight=self.sparse_weight)
        order = np.argsort(-np.array(scores))
        ranked = [(candidate_idxs[i], float(scores[i])) for i in order]
    
        # логируем топ-10
        preview = []
        for idx, sc in ranked[:10]:
            content, full_doc_id = self.chunk_db[idx]
            header = content.split("\n", 1)[0][:120]
            preview.append({"idx": idx, "hybrid": sc, "doc": full_doc_id, "header": header})
        self.logger.debug("Hybrid rerank: threshold=%.3f, top preview: %s", self.hybrid_threshold, preview)
        return ranked


    async def get_links_from_answer(self, answer_text: str) -> list[str]:
        try:
            if not answer_text or not self.chunk_db:
                self.logger.warning("No answer text (%s) or empty chunk_db=%d", bool(answer_text), len(self.chunk_db))
                return []
    
            self.logger.debug("Linking start: answer_len=%d", len(answer_text))
            parts = [p.strip() for p in answer_text.split("\n\n") if len(p.strip()) > 30]
            if not parts:
                parts = [answer_text]
            self.logger.debug("Answer split: parts=%d, lengths=%s", len(parts), [len(p) for p in parts])
    
            docid2best = {}
            per_chunk_k = self.per_chunk_top_k or max(5, self.top_k // 10)
    
            for pi, part in enumerate(parts):
                cands = await self._bm25_candidates(part)
                reranked = await self._hybrid_rerank(part, cands)
    
                kept = 0
                for idx, score in reranked[:per_chunk_k]:
                    content, full_doc_id = self.chunk_db[idx]
                    base_id = self._prepare_doc_id(full_doc_id)
                    if score < self.hybrid_threshold:
                        self.logger.debug(
                            "Reject: score %.3f < %.3f for doc=%s (idx=%d, header=%s...)",
                            score, self.hybrid_threshold, base_id, idx, content.split("\n", 1)[0][:80]
                        )
                        continue
                    prev = docid2best.get(base_id)
                    if (prev is None) or (score > prev):
                        docid2best[base_id] = score
                    kept += 1
                self.logger.debug("Part %d: kept %d docs after threshold (per_chunk_top_k=%d)", pi, kept, per_chunk_k)
    
            if not docid2best:
                self.logger.info("No docs passed threshold; no links will be added.")
                return []
    
            best_docs = sorted(docid2best.items(), key=lambda kv: -kv[1])
            urls, seen = [], set()
            for doc_id, sc in best_docs:
                url = self.docid2url.get(doc_id)
                if not url:
                    self.logger.debug("No URL mapping for doc %s (score=%.3f)", doc_id, sc)
                    continue
                if url in seen:
                    continue
                urls.append(url); seen.add(url)
                if len(urls) >= self.max_links:
                    break
    
            self.logger.info("Links chosen: %s", urls)
            return urls
        except Exception as e:
            self.logger.exception("Linking failed with exception: %s", e)
            return []

    async def get_formated_answer(self, answer: str) -> str:
        links = await self.get_links_from_answer(answer)
        if links:
            return f"{answer}\n\nСсылки, которые могут быть полезны:\n- " + "\n- ".join(links)
        return answer
