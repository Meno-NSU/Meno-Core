import pytest

from meno_core.core.rag_runtime import (
    CHUNK_RAG_KB_ID,
    RAG_ENGINE_CHUNK_RAG,
    RAG_ENGINE_LIGHTRAG,
    RagSelectionError,
    build_public_rag_backend_registry,
)


class _DummyBackend:
    async def answer(self, request, timings_sink=None):
        raise NotImplementedError


def _build_registry():
    return build_public_rag_backend_registry(
        lightrag_kb_id="graph-kb",
        lightrag_backend=_DummyBackend(),
        chunk_rag_backend=_DummyBackend(),
    )


def test_registry_defaults_to_lightrag_when_request_has_no_selection():
    registry = _build_registry()

    entry, route_reason = registry.resolve(None, None)

    assert entry.knowledge_base_id == "graph-kb"
    assert entry.rag_engine_id == RAG_ENGINE_LIGHTRAG
    assert "default selection" in route_reason


def test_registry_can_infer_engine_from_knowledge_base():
    registry = _build_registry()

    entry, route_reason = registry.resolve("graph-kb", None)

    assert entry.knowledge_base_id == "graph-kb"
    assert entry.rag_engine_id == RAG_ENGINE_LIGHTRAG
    assert "inferred rag_engine_id" in route_reason


def test_registry_can_infer_knowledge_base_from_engine():
    registry = _build_registry()

    entry, route_reason = registry.resolve(None, RAG_ENGINE_CHUNK_RAG)

    assert entry.knowledge_base_id == CHUNK_RAG_KB_ID
    assert entry.rag_engine_id == RAG_ENGINE_CHUNK_RAG
    assert "inferred knowledge_base_id" in route_reason


def test_registry_reports_supported_pairs_for_invalid_combination():
    registry = _build_registry()

    with pytest.raises(RagSelectionError) as error:
        registry.resolve("graph-kb", RAG_ENGINE_CHUNK_RAG)

    message = str(error.value)
    assert "Unsupported knowledge_base_id/rag_engine_id combination" in message
    assert "Supported combinations" in message


def test_registry_builder_fails_if_public_backend_is_missing():
    with pytest.raises(RuntimeError) as error:
        build_public_rag_backend_registry(
            lightrag_kb_id="graph-kb",
            lightrag_backend=_DummyBackend(),
            chunk_rag_backend=None,
        )

    assert "chunk_rag" in str(error.value)


def test_registry_lists_kbs_without_legacy_label():
    registry = _build_registry()

    knowledge_bases = registry.list_knowledge_bases()

    assert knowledge_bases == [
        {
            "id": "graph-kb",
            "name": "Граф знаний",
            "description": "Поиск по графу знаний и связанным чанкам документов",
            "supported_rag_engines": [RAG_ENGINE_LIGHTRAG],
        },
        {
            "id": CHUNK_RAG_KB_ID,
            "name": "Векторный поиск",
            "description": "Dense и BM25 поиск по текстовым чанкам с fusion и rerank",
            "supported_rag_engines": [RAG_ENGINE_CHUNK_RAG],
        },
    ]
