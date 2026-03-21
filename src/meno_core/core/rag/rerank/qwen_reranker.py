import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]

from meno_core.core.rag.debug_utils import build_retrieved_chunk_preview
from meno_core.core.rag.models import RetrievedChunk

logger = logging.getLogger(__name__)
retrieval_logger = logging.getLogger("chunk_rag.retrieval")

DEFAULT_RERANK_INSTRUCTION = "Given a user query, retrieve relevant passages that answer the query."
_BACKEND_CACHE: dict[str, "QwenRerankerBackend"] = {}


@dataclass(slots=True)
class QwenRerankResult:
    reranked_chunks: list[RetrievedChunk]
    preview: list[dict[str, Any]]


class QwenRerankerBackend:
    def __init__(
        self,
        tokenizer,
        model,
        model_path: str,
        *,
        batch_size: int = 4,
        max_length: int = 8192,
        instruction: str = DEFAULT_RERANK_INSTRUCTION,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.instruction = instruction

        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        if hasattr(self.tokenizer, "deprecation_warnings"):
            self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        self.prefix_tokens = self.tokenizer(
            "<|im_start|>system\nJudge whether the Document is relevant to the Query.\n<|im_end|>\n"
            "<|im_start|>user\n",
            add_special_tokens=False,
        )["input_ids"]
        self.suffix_tokens = self.tokenizer(
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            add_special_tokens=False,
        )["input_ids"]
        self.no_token_id = self.tokenizer.convert_tokens_to_ids("no")
        self.yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")

    @property
    def device(self) -> str:
        try:
            return str(next(self.model.parameters()).device)
        except Exception:
            device = getattr(self.model, "device", None)
            if device is not None:
                return str(device)
        return "unknown"

    @classmethod
    def from_pretrained(cls, model_path: str) -> "QwenRerankerBackend":
        logger.info("Loading Qwen reranker backend: %s", model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
        model.eval()
        backend = cls(tokenizer=tokenizer, model=model, model_path=model_path)
        logger.info("Qwen reranker backend loaded on device=%s", backend.device)
        return backend

    def format_pair(self, query: str, document: str, instruction: str | None = None) -> str:
        effective_instruction = instruction or self.instruction
        return (
            f"<Instruct>: {effective_instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )

    def process_inputs(self, pairs: Sequence[tuple[str, str]], instruction: str | None = None) -> dict[str, torch.Tensor]:
        texts = [self.format_pair(query, document, instruction=instruction) for query, document in pairs]
        encoded = self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=max(self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens), 32),
            return_attention_mask=False,
            add_special_tokens=False,
        )
        input_ids = [self.prefix_tokens + ids + self.suffix_tokens for ids in encoded["input_ids"]]
        padded = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt",
        )
        return {key: value.to(self.model.device) for key, value in padded.items()}

    @torch.no_grad()
    def score_pairs(self, pairs: Sequence[tuple[str, str]], instruction: str | None = None) -> list[float]:
        if not pairs:
            return []

        scores: list[float] = []
        num_batches = math.ceil(len(pairs) / self.batch_size)
        for batch_idx in range(num_batches):
            batch_start = batch_idx * self.batch_size
            batch_pairs = pairs[batch_start: batch_start + self.batch_size]
            inputs = self.process_inputs(batch_pairs, instruction=instruction)
            logits = self.model(**inputs, return_dict=True).logits[:, -1, :]
            batch_logits = torch.stack(
                [logits[:, self.no_token_id], logits[:, self.yes_token_id]],
                dim=1,
            )
            probs = torch.softmax(batch_logits, dim=1)[:, 1]
            scores.extend(float(score) for score in probs.detach().cpu().tolist())

        return scores

    def rerank_documents(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_n: int | None = None,
        instruction: str | None = None,
    ) -> list[dict[str, Any]]:
        scores = self.score_pairs([(query, document) for document in documents], instruction=instruction)
        ranked = [
            {"index": idx, "relevance_score": float(score)}
            for idx, score in enumerate(scores)
        ]
        ranked.sort(key=lambda item: -item["relevance_score"])
        if top_n is not None:
            ranked = ranked[:top_n]
        return ranked


class QwenCausalReranker:
    def __init__(
        self,
        backend: QwenRerankerBackend,
        *,
        filter_threshold: float = 0.0,
        preview_k: int = 5,
    ):
        self.backend = backend
        self.filter_threshold = filter_threshold
        self.preview_k = preview_k

    async def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> QwenRerankResult:
        if not chunks:
            return QwenRerankResult(reranked_chunks=[], preview=[])

        started_at = time.perf_counter()
        docs = [chunk_wrapper.chunk.text for chunk_wrapper in chunks]
        scores = self.backend.score_pairs([(query, doc) for doc in docs])

        reranked_chunks: list[RetrievedChunk] = []
        for chunk_wrapper, score in zip(chunks, scores):
            if score >= self.filter_threshold:
                chunk_wrapper.score = float(score)
                reranked_chunks.append(chunk_wrapper)

        reranked_chunks.sort(key=lambda chunk_wrapper: -chunk_wrapper.score)
        reranked_chunks = reranked_chunks[:top_n]
        preview = build_retrieved_chunk_preview(reranked_chunks, self.preview_k)
        retrieval_logger.info(
            "reranker=qwen model=%s device=%s candidates=%s kept=%s top_n=%s latency_ms=%.2f",
            self.backend.model_path,
            self.backend.device,
            len(chunks),
            len(reranked_chunks),
            top_n,
            (time.perf_counter() - started_at) * 1000,
        )
        return QwenRerankResult(reranked_chunks=reranked_chunks, preview=preview)


def load_cached_qwen_reranker_backend(model_path: str) -> QwenRerankerBackend:
    if model_path not in _BACKEND_CACHE:
        _BACKEND_CACHE[model_path] = QwenRerankerBackend.from_pretrained(model_path)
    return _BACKEND_CACHE[model_path]
