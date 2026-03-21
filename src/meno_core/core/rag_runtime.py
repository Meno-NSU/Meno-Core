from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol


RAG_ENGINE_LIGHTRAG = "lightrag"
RAG_ENGINE_CHUNK_RAG = "chunk_rag"
CHUNK_RAG_KB_ID = "chunk-rag-kb"


class RagSelectionError(ValueError):
    pass


class RagChatBackend(Protocol):
    async def answer(
        self,
        request: "RagChatRequest",
        timings_sink: dict[str, float] | None = None,
    ) -> str | Any:
        ...


@dataclass(frozen=True, slots=True)
class RagChatRequest:
    question: str
    history: list[dict[str, str]]
    system_prompt: str
    stream: bool
    session_id: str
    request_id: str
    model: str
    knowledge_base_id: str
    rag_engine_id: str
    route_reason: str
    base_url: str | None = None


@dataclass(frozen=True, slots=True)
class RagBackendEntry:
    knowledge_base_id: str
    knowledge_base_name: str
    description: str
    rag_engine_id: str
    backend: RagChatBackend


class RagBackendRegistry:
    def __init__(
        self,
        entries: list[RagBackendEntry],
        *,
        default_selection: tuple[str, str],
    ):
        if not entries:
            raise RuntimeError("RAG backend registry cannot be empty.")
        self._entries = entries
        self._entries_by_pair = {
            (entry.knowledge_base_id, entry.rag_engine_id): entry
            for entry in entries
        }
        self._entries_by_kb: dict[str, list[RagBackendEntry]] = {}
        self._entries_by_engine: dict[str, list[RagBackendEntry]] = {}
        for entry in entries:
            self._entries_by_kb.setdefault(entry.knowledge_base_id, []).append(entry)
            self._entries_by_engine.setdefault(entry.rag_engine_id, []).append(entry)

        if default_selection not in self._entries_by_pair:
            raise RuntimeError(f"Unknown default RAG selection: {default_selection!r}")
        self.default_selection = default_selection

    def supported_pairs(self) -> list[dict[str, str]]:
        return [
            {
                "knowledge_base_id": entry.knowledge_base_id,
                "rag_engine_id": entry.rag_engine_id,
            }
            for entry in self._entries
        ]

    def resolve(
        self,
        knowledge_base_id: str | None,
        rag_engine_id: str | None,
    ) -> tuple[RagBackendEntry, str]:
        normalized_kb = knowledge_base_id.strip() if isinstance(knowledge_base_id, str) and knowledge_base_id.strip() else None
        normalized_engine = rag_engine_id.strip() if isinstance(rag_engine_id, str) and rag_engine_id.strip() else None

        if normalized_kb is None and normalized_engine is None:
            return self._entries_by_pair[self.default_selection], (
                f"default selection {self.default_selection[0]!r}/{self.default_selection[1]!r}"
            )

        if normalized_kb is not None and normalized_engine is not None:
            key = (normalized_kb, normalized_engine)
            entry = self._entries_by_pair.get(key)
            if entry is None:
                raise RagSelectionError(
                    "Unsupported knowledge_base_id/rag_engine_id combination. "
                    f"Got knowledge_base_id={normalized_kb!r}, rag_engine_id={normalized_engine!r}. "
                    f"Supported combinations={self.supported_pairs()}."
                )
            return entry, "explicit knowledge_base_id and rag_engine_id"

        if normalized_kb is not None:
            entries = self._entries_by_kb.get(normalized_kb)
            if not entries:
                raise RagSelectionError(
                    f"Unknown knowledge_base_id={normalized_kb!r}. "
                    f"Supported knowledge_base_ids={sorted(self._entries_by_kb)}."
                )
            if len(entries) != 1:
                supported = sorted(entry.rag_engine_id for entry in entries)
                raise RagSelectionError(
                    "knowledge_base_id requires explicit rag_engine_id because multiple engines are supported. "
                    f"knowledge_base_id={normalized_kb!r}, supported_rag_engine_ids={supported}."
                )
            entry = entries[0]
            return entry, f"inferred rag_engine_id={entry.rag_engine_id!r} from knowledge_base_id={normalized_kb!r}"

        assert normalized_engine is not None
        entries = self._entries_by_engine.get(normalized_engine)
        if not entries:
            raise RagSelectionError(
                f"Unknown rag_engine_id={normalized_engine!r}. "
                f"Supported rag_engine_ids={sorted(self._entries_by_engine)}."
            )
        if len(entries) != 1:
            supported = sorted(entry.knowledge_base_id for entry in entries)
            raise RagSelectionError(
                "rag_engine_id requires explicit knowledge_base_id because multiple knowledge bases are supported. "
                f"rag_engine_id={normalized_engine!r}, supported_knowledge_base_ids={supported}."
            )
        entry = entries[0]
        return entry, f"inferred knowledge_base_id={entry.knowledge_base_id!r} from rag_engine_id={normalized_engine!r}"

    def list_knowledge_bases(self) -> list[dict[str, Any]]:
        knowledge_bases: list[dict[str, Any]] = []
        seen_kb_ids: set[str] = set()
        for entry in self._entries:
            if entry.knowledge_base_id in seen_kb_ids:
                continue
            seen_kb_ids.add(entry.knowledge_base_id)
            supported_entries = self._entries_by_kb[entry.knowledge_base_id]
            knowledge_bases.append(
                {
                    "id": entry.knowledge_base_id,
                    "name": entry.knowledge_base_name,
                    "description": entry.description,
                    "supported_rag_engines": [supported_entry.rag_engine_id for supported_entry in supported_entries],
                }
            )
        return knowledge_bases

    def default_payload(self) -> dict[str, str]:
        knowledge_base_id, rag_engine_id = self.default_selection
        return {
            "knowledge_base_id": knowledge_base_id,
            "rag_engine_id": rag_engine_id,
        }


def build_public_rag_backend_registry(
    *,
    lightrag_kb_id: str,
    lightrag_backend: RagChatBackend | None,
    chunk_rag_backend: RagChatBackend | None,
) -> RagBackendRegistry:
    missing_backends: list[str] = []
    if lightrag_backend is None:
        missing_backends.append(RAG_ENGINE_LIGHTRAG)
    if chunk_rag_backend is None:
        missing_backends.append(RAG_ENGINE_CHUNK_RAG)
    if missing_backends:
        raise RuntimeError(
            "Public RAG backend initialization failed for: " + ", ".join(sorted(missing_backends))
        )

    assert lightrag_backend is not None
    assert chunk_rag_backend is not None
    entries = [
        RagBackendEntry(
            knowledge_base_id=lightrag_kb_id,
            knowledge_base_name="Граф знаний",
            description="Поиск по графу знаний и связанным чанкам документов",
            rag_engine_id=RAG_ENGINE_LIGHTRAG,
            backend=lightrag_backend,
        ),
        RagBackendEntry(
            knowledge_base_id=CHUNK_RAG_KB_ID,
            knowledge_base_name="Векторный поиск",
            description="Dense и BM25 поиск по текстовым чанкам с fusion и rerank",
            rag_engine_id=RAG_ENGINE_CHUNK_RAG,
            backend=chunk_rag_backend,
        ),
    ]
    return RagBackendRegistry(
        entries,
        default_selection=(lightrag_kb_id, RAG_ENGINE_LIGHTRAG),
    )


class ChunkRagChatBackend:
    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator

    async def answer(
        self,
        request: RagChatRequest,
        timings_sink: dict[str, float] | None = None,
    ) -> str | AsyncIterator[str | dict]:
        from meno_core.core.lightrag_timing import reset_rag_request_trace, start_rag_request_trace
        from meno_core.core.rag.models import RagMessage, RagRequest

        trace, token = start_rag_request_trace(
            request_id=request.request_id,
            session_id=request.session_id,
            knowledge_base_id=request.knowledge_base_id,
            rag_engine_id=request.rag_engine_id,
            model=request.model,
            stream=request.stream,
            route_reason=request.route_reason,
        )

        rag_request = RagRequest(
            question=request.question,
            history=[RagMessage(role=item["role"], text=item["content"]) for item in request.history],  # type: ignore[arg-type]
            mode=request.rag_engine_id,
            session_id=request.session_id,
            request_id=request.request_id,
            model=request.model,
            base_url=request.base_url,
        )

        if request.stream and hasattr(self.orchestrator, "answer_stream"):
            async def _stream_wrapper() -> AsyncIterator[str | dict]:
                try:
                    async for piece in self.orchestrator.answer_stream(rag_request):
                        yield piece
                    trace.finalize(timings_sink=timings_sink)
                except Exception as error:
                    trace.finalize(timings_sink=timings_sink, error=error)
                    raise
                finally:
                    reset_rag_request_trace(token)

            return _stream_wrapper()

        try:
            response = await self.orchestrator.answer(rag_request)
            trace.finalize(timings_sink=timings_sink)
            return response.answer
        except Exception as error:
            trace.finalize(timings_sink=timings_sink, error=error)
            raise
        finally:
            reset_rag_request_trace(token)
