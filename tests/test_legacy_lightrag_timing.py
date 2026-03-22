import asyncio
import logging

from meno_core.core.lightrag_engine import LightRAGEngine
from meno_core.core.lightrag_timing import get_current_rag_trace


class _FakeRagNonStream:
    async def aquery(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return "final answer"


class _FakeRagStream:
    async def aquery(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        trace = get_current_rag_trace()
        assert trace is not None
        trace.mark_llm_stream_open()

        async def _stream():
            yield "part-1"
            yield "part-2"

        return _stream()


class _FakeParam:
    def __init__(self, stream: bool):
        self.stream = stream


def test_lightrag_timing_non_stream(caplog):
    rag = _FakeRagNonStream()
    engine = LightRAGEngine(rag_instance=rag, embedder=None, bm25=None, chunk_db=[])
    timings: dict[str, float] = {}

    with caplog.at_level(logging.INFO, logger="meno_core.request"):
        result = asyncio.run(
            engine.aquery(
                "question",
                param=_FakeParam(stream=False),
                request_id="req-1",
                session_id="session-1",
                timings_sink=timings,
            )
        )

    assert result == "final answer"
    assert "request_id" not in rag.kwargs
    assert "session_id" not in rag.kwargs
    assert "timings_sink" not in rag.kwargs
    assert timings["request_total"] >= 0
    assert any("rag-summary request_id=req-1" in record.message for record in caplog.records)


def test_lightrag_timing_stream(caplog):
    rag = _FakeRagStream()
    engine = LightRAGEngine(rag_instance=rag, embedder=None, bm25=None, chunk_db=[])
    timings: dict[str, float] = {}

    async def _run() -> tuple[str, list[dict]]:
        result = await engine.aquery(
            "question",
            param=_FakeParam(stream=True),
            request_id="req-2",
            session_id="session-2",
            timings_sink=timings,
        )
        parts: list[str] = []
        stage_events: list[dict] = []
        async for piece in result:
            if isinstance(piece, dict):
                stage_events.append(piece)
            else:
                parts.append(piece)
        return "".join(parts), stage_events

    with caplog.at_level(logging.INFO, logger="meno_core.request"):
        combined, stages = asyncio.run(_run())

    assert combined == "part-1part-2"
    # Verify a retrieval stage and generation stages are emitted
    stage_names = [s["_stage"] for s in stages]
    assert "retrieval" in stage_names
    assert "generation" in stage_names
    assert timings["llm_first_chunk"] >= 0
    assert timings["llm_stream"] >= 0
    assert timings["request_total"] >= 0
    assert any("rag-summary request_id=req-2" in record.message for record in caplog.records)
