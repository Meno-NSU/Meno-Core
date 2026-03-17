import asyncio
from types import SimpleNamespace

import numpy as np

from meno_core.core.rag.generation.context_assembler import ContextAssembler, estimate_tokens
from meno_core.core.rag.models import Chunk, ChunkMetadata, RetrievedChunk
from meno_core.core.rag.retrieval.bm25_retriever import BM25LexicalRetriever
from meno_core.core.zvec_rag import ZvecRAGEngine


def _chunk(chunk_id: str, text: str, *, document_id: str = "doc") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        text_for_dense=text,
        text_for_bm25=text,
        metadata=ChunkMetadata(
            document_id=document_id,
            document_title=document_id,
            chunk_index=0,
            source_url="https://example.com",
        ),
    )


class FakeBM25:
    def __init__(self, scores):
        self._scores = np.asarray(scores, dtype=float)
        self.corpus_size = len(scores)

    def get_scores(self, query_terms):
        return self._scores


def test_bm25_retriever_drops_non_positive_scores():
    retriever = BM25LexicalRetriever(
        bm25=FakeBM25([0.0, -0.5]),
        chunk_map={
            "0": _chunk("chunk-0", "zero"),
            "1": _chunk("chunk-1", "negative"),
        },
    )

    results = asyncio.run(retriever.retrieve_many(["missing term"], top_k=2))

    assert results == [[]]


def test_context_assembler_enforces_budget_on_formatted_context():
    first = RetrievedChunk(chunk=_chunk("chunk-0", "alpha beta gamma delta", document_id="doc-a"), score=1.0, source="hybrid")
    second = RetrievedChunk(
        chunk=_chunk("chunk-1", "epsilon zeta eta theta iota kappa", document_id="doc-b"),
        score=0.9,
        source="hybrid",
    )

    first_only_context, _ = ContextAssembler(token_budget=10_000).assemble([first])
    budget = estimate_tokens(first_only_context)
    assembler = ContextAssembler(token_budget=budget)

    context, sources = assembler.assemble([first, second])

    assert estimate_tokens(context) <= budget
    assert [source.document_id for source in sources] == ["doc-a"]
    assert "doc-b" not in context


def test_legacy_bm25_results_drop_non_positive_scores():
    engine = object.__new__(ZvecRAGEngine)
    engine.bm25 = FakeBM25([0.0, -0.1, 2.5])
    engine.legacy_chunks = [
        _chunk("legacy-0", "zero"),
        _chunk("legacy-1", "negative"),
        _chunk("legacy-2", "positive"),
    ]

    results = engine._bm25_results("query", top_k=3)

    assert [item.chunk.chunk_id for item in results] == ["legacy-2"]


def test_legacy_zvec_uses_embedder_dimension_not_settings(monkeypatch, tmp_path):
    recorded = {}

    def fake_vector_schema(name, data_type, dimension):
        recorded["dimension"] = dimension
        return {"name": name, "data_type": data_type, "dimension": dimension}

    class FakeCollection:
        def insert(self, docs):
            recorded.setdefault("insert_calls", 0)
            recorded["insert_calls"] += 1

    monkeypatch.setattr("meno_core.core.zvec_rag.zvec.VectorSchema", fake_vector_schema)
    monkeypatch.setattr("meno_core.core.zvec_rag.zvec.CollectionSchema", lambda name, vectors: {"name": name, "vectors": vectors})
    monkeypatch.setattr("meno_core.core.zvec_rag.zvec.create_and_open", lambda path, schema: FakeCollection())

    engine = object.__new__(ZvecRAGEngine)
    engine.embedder = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(hidden_size=1536)))
    engine.zvec_path = tmp_path / "legacy-zvec"
    engine.chunk_db = []

    engine._init_zvec_collection()

    assert recorded["dimension"] == 1536
