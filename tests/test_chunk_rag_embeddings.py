from types import SimpleNamespace

import torch

from meno_core.core.rag.embeddings import User2DenseEmbedder


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    def __init__(self):
        self.seen_texts = []

    def __call__(self, texts, **kwargs):
        self.seen_texts.append(list(texts))
        batch_size = len(texts)
        return FakeBatch(
            {
                "input_ids": torch.tensor([[1, 2, 0]] * batch_size, dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 0]] * batch_size, dtype=torch.long),
            }
        )


class FakeModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(hidden_size=2)

    def __call__(self, **kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        last_hidden_state = torch.tensor(
            [[[1.0, 0.0], [3.0, 4.0], [100.0, 100.0]]] * batch_size,
            dtype=torch.float32,
        )
        return SimpleNamespace(last_hidden_state=last_hidden_state)


def test_user2_embedder_applies_prefixes_and_mean_pooling():
    tokenizer = FakeTokenizer()
    model = FakeModel()
    embedder = User2DenseEmbedder(tokenizer=tokenizer, model=model, model_path="user2-test")

    query_embeddings = embedder.encode_queries(["НГУ"])
    doc_embeddings = embedder.encode_documents(["Документ про НГУ"])

    assert tokenizer.seen_texts[0] == ["search_query: НГУ"]
    assert tokenizer.seen_texts[1] == ["search_document: Документ про НГУ"]
    assert query_embeddings.shape == (1, 2)
    assert doc_embeddings.shape == (1, 2)
    assert torch.allclose(query_embeddings[0], torch.tensor([0.7071, 0.7071]), atol=1e-4)
