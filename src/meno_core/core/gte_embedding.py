from collections import defaultdict
from typing import Any

import numpy as np
import torch


class GTEEmbedding(torch.nn.Module):
    def __init__(self, tokenizer, token_cls_model, normalized: bool = True):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = token_cls_model
        self.normalized = normalized
        self.vocab_size = self.model.config.vocab_size

    def _process_token_weights(
            self,
            token_weights: np.ndarray,
            input_ids: list[int],
    ) -> defaultdict[Any, float]:
        result: defaultdict[Any, float] = defaultdict(float)
        unused: set[Any] = {
            t_id
            for t_id in {
                self.tokenizer.cls_token_id,
                getattr(self.tokenizer, "eos_token_id", None),
                self.tokenizer.pad_token_id,
                self.tokenizer.unk_token_id,
            }
            if t_id is not None
        }
        for w, idx in zip(token_weights, input_ids):
            w_val = float(w)
            if idx not in unused and w_val > 0.0:
                token = self.tokenizer.decode([int(idx)])
                if w_val > result[token]:
                    result[token] = w_val
        return result

    @torch.no_grad()
    def encode(self, texts, dimension=None, max_length=4096, batch_size=16,
               return_dense=True, return_sparse=True):
        if isinstance(texts, str):
            texts = [texts]
        if dimension is None:
            dimension = self.model.config.hidden_size

        toks = self.tokenizer(texts, return_tensors='pt', padding=True,
                              truncation=True, max_length=max_length).to(self.model.device)
        out = self.model(**toks, return_dict=True)
        res = {}
        if return_dense:
            dense = out.last_hidden_state[:, 0, :dimension]
            if self.normalized:
                dense = torch.nn.functional.normalize(dense, dim=-1)
            res["dense_embeddings"] = dense
        if return_sparse:
            tw = torch.relu(out.logits).squeeze(-1)
            res["token_weights"] = list(map(
                self._process_token_weights,
                tw.detach().cpu().numpy().tolist(),
                toks['input_ids'].cpu().numpy().tolist()
            ))
        return res

    def compute_dense_scores(self, e1, e2):
        return torch.sum(e1 * e2, dim=-1).cpu().detach().numpy()

    def _sparse_dot(self, a, b):
        s = 0.0
        for t, w in a.items():
            if t in b:
                s += w * b[t]
        return s

    def compute_sparse_scores(self, s1, s2):
        return np.array([self._sparse_dot(a, b) for a, b in zip(s1, s2)])

    @torch.no_grad()
    def compute_scores(self, pairs, dimension=None, max_length=4096,
                       dense_weight=1.0, sparse_weight=0.1):
        t1 = [p[0] for p in pairs]
        t2 = [p[1] for p in pairs]
        e1 = self.encode(t1, dimension, max_length)
        e2 = self.encode(t2, dimension, max_length)
        ds = self.compute_dense_scores(e1["dense_embeddings"], e2["dense_embeddings"])
        ss = self.compute_sparse_scores(e1["token_weights"], e2["token_weights"])
        return (ds * dense_weight + ss * sparse_weight).tolist()
