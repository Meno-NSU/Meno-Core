from types import SimpleNamespace

import torch

from meno_core.core.rag.rerank.qwen_reranker import QwenRerankerBackend


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    def __init__(self):
        self.pad_token = None
        self.eos_token = "<eos>"
        self.padding_side = "right"
        self.last_formatted_texts = []
        self.vocab = {"no": 0, "yes": 1}

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            encoded = [3 if "unrelated" in texts.lower() else 7, 4]
            return {"input_ids": encoded}
        texts = list(texts)
        if kwargs.get("return_tensors") == "pt":
            max_len = max(len(text) for text in texts)
            rows = []
            for text in texts:
                row = [1 if "relevant" in text.lower() else 2]
                row += [0] * (max_len - 1)
                rows.append(row)
            return FakeBatch(
                {
                    "input_ids": torch.tensor(rows, dtype=torch.long),
                    "attention_mask": torch.ones(len(rows), max_len, dtype=torch.long),
                }
            )

        self.last_formatted_texts = texts
        encoded = []
        for text in texts:
            encoded.append([3 if "unrelated" in text.lower() else 7, 4])
        return {"input_ids": encoded}

    def pad(self, encoded, padding=True, return_tensors="pt"):
        input_ids = encoded["input_ids"]
        max_len = max(len(row) for row in input_ids)
        padded = [row + [0] * (max_len - len(row)) for row in input_ids]
        attention_mask = [[1] * len(row) + [0] * (max_len - len(row)) for row in input_ids]
        return FakeBatch(
            {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        )


class FakeModel:
    def __init__(self):
        self.device = torch.device("cpu")

    def __call__(self, **kwargs):
        input_ids = kwargs["input_ids"]
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros((batch_size, seq_len, 2), dtype=torch.float32)
        for row_idx in range(batch_size):
            relevance_marker = input_ids[row_idx, 2].item()
            if relevance_marker >= 7:
                logits[row_idx, -1, 1] = 5.0
                logits[row_idx, -1, 0] = 1.0
            else:
                logits[row_idx, -1, 1] = 1.0
                logits[row_idx, -1, 0] = 5.0
        return SimpleNamespace(logits=logits)


def test_qwen_backend_formats_prompt_and_extracts_yes_probability():
    backend = QwenRerankerBackend(
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        model_path="qwen-test",
    )

    scores = backend.score_pairs(
        [
            ("Where is NSU?", "Relevant passage about NSU"),
            ("Where is NSU?", "Unrelated text"),
        ]
    )

    assert "<Instruct>:" in backend.format_pair("q", "d")
    assert "<Query>: q" in backend.format_pair("q", "d")
    assert "<Document>: d" in backend.format_pair("q", "d")
    assert scores[0] > scores[1]
    assert scores[0] > 0.9
    assert scores[1] < 0.1
