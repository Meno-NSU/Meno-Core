import logging
from dataclasses import dataclass
from typing import Protocol, Sequence

import torch
import torch.nn.functional as F  # type: ignore[import-untyped]
from transformers import AutoModel, AutoTokenizer  # type: ignore[import-untyped]

from meno_core.core.gte_embedding import GTEEmbedding

logger = logging.getLogger(__name__)


class DenseEmbedder(Protocol):
    name: str
    model_path: str
    dimension: int
    pooling_strategy: str
    query_prefix: str
    document_prefix: str

    def encode_queries(self, texts: Sequence[str]) -> torch.Tensor:
        ...

    def encode_documents(self, texts: Sequence[str]) -> torch.Tensor:
        ...


@dataclass(slots=True)
class MultilingualDenseEmbedder:
    base_embedder: GTEEmbedding
    model_path: str
    name: str = "multilingual_dense"
    pooling_strategy: str = "cls"
    query_prefix: str = ""
    document_prefix: str = ""

    @property
    def dimension(self) -> int:
        return int(self.base_embedder.model.config.hidden_size)

    def encode_queries(self, texts: Sequence[str]) -> torch.Tensor:
        return self._encode(texts)

    def encode_documents(self, texts: Sequence[str]) -> torch.Tensor:
        return self._encode(texts)

    def _encode(self, texts: Sequence[str]) -> torch.Tensor:
        result = self.base_embedder.encode(list(texts), return_dense=True, return_sparse=False)
        return result["dense_embeddings"]


class User2DenseEmbedder:
    name = "russian_dense"
    pooling_strategy = "mean"
    query_prefix = "search_query: "
    document_prefix = "search_document: "

    def __init__(
        self,
        tokenizer,
        model,
        model_path: str,
        *,
        max_length: int = 4096,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.model_path = model_path
        self.max_length = max_length
        self.dimension = int(self.model.config.hidden_size)

    @classmethod
    def from_pretrained(cls, model_path: str, *, max_length: int = 4096) -> "User2DenseEmbedder":
        logger.info("Loading russian dense embedder: %s", model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
        model.eval()
        return cls(tokenizer=tokenizer, model=model, model_path=model_path, max_length=max_length)

    def encode_queries(self, texts: Sequence[str]) -> torch.Tensor:
        return self._encode(texts, prefix=self.query_prefix)

    def encode_documents(self, texts: Sequence[str]) -> torch.Tensor:
        return self._encode(texts, prefix=self.document_prefix)

    @torch.no_grad()
    def _encode(self, texts: Sequence[str], *, prefix: str) -> torch.Tensor:
        prepared = [f"{prefix}{text}" for text in texts]
        tokens = self.tokenizer(
            prepared,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.model.device)
        outputs = self.model(**tokens, return_dict=True)
        pooled = self._mean_pool(outputs.last_hidden_state, tokens["attention_mask"])
        return F.normalize(pooled, p=2, dim=1)

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts
