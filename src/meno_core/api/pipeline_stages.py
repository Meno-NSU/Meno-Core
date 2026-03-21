"""Observable pipeline stages and latency tracking for the RAG SSE stream."""

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from logging import Logger
from typing import Any, Optional


class StageName(str, Enum):
    ABBREVIATION_EXPANSION = "abbreviation_expansion"
    ANAPHORA_RESOLUTION = "anaphora_resolution"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    RETRIEVAL_AND_GENERATION = "retrieval_and_generation"
    LINK_ADDITION = "link_addition"
    LINK_CORRECTION = "link_correction"


class StageStatus(str, Enum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageEvent:
    stage: str
    status: str
    ts: float = field(default_factory=time.time)
    duration_ms: Optional[float] = None
    detail: Optional[dict[str, Any]] = None

    def to_sse(self) -> str:
        payload = {k: v for k, v in asdict(self).items() if v is not None}
        return f"event: stage\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@dataclass
class StageSummary:
    total_ms: float
    stages: dict[str, float]

    def to_sse(self) -> str:
        return f"event: summary\ndata: {json.dumps(asdict(self), ensure_ascii=False)}\n\n"


class StageTracker:
    """Tracks timing for a single pipeline stage."""

    def __init__(self, stage: StageName, logger: Logger):
        self.stage = stage
        self.logger = logger
        self._start: float = 0.0
        self.duration_ms: float = 0.0
        self.detail: dict[str, Any] = {}
        self.start_event: Optional[StageEvent] = None
        self.end_event: Optional[StageEvent] = None

    async def __aenter__(self) -> "StageTracker":
        self._start = time.monotonic()
        self.start_event = StageEvent(
            stage=self.stage.value,
            status=StageStatus.STARTED.value,
        )
        self.logger.info("Stage %s started", self.stage.value)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.monotonic() - self._start
        self.duration_ms = round(elapsed * 1000, 1)
        status = StageStatus.FAILED.value if exc_type else StageStatus.COMPLETED.value
        self.end_event = StageEvent(
            stage=self.stage.value,
            status=status,
            duration_ms=self.duration_ms,
            detail=self.detail or None,
        )
        self.logger.info(
            "Stage %s %s in %.1f ms", self.stage.value, status, self.duration_ms
        )
        return False


class PipelineTimer:
    """Collects stage durations and produces a summary."""

    def __init__(self):
        self.stages: dict[str, float] = {}
        self._pipeline_start: float = time.monotonic()

    def record(self, tracker: StageTracker) -> None:
        self.stages[tracker.stage.value] = tracker.duration_ms

    def summary(self) -> StageSummary:
        total = round((time.monotonic() - self._pipeline_start) * 1000, 1)
        return StageSummary(total_ms=total, stages=self.stages)


class ThinkingTokenSplitter:
    """Stateful splitter that separates <think>...</think> from content in a stream.

    Handles the case where tags are split across chunk boundaries by buffering
    the last few characters (length of the longest tag).
    """

    THINK_START = "<think>"
    THINK_END = "</think>"
    _MAX_TAG_LEN = len(THINK_END)  # 8 chars — longest tag

    def __init__(self):
        self.in_thinking: bool = False
        self._buffer: str = ""

    def feed(self, text: str) -> list[tuple[str, bool]]:
        """Process a chunk and return list of (text, is_thinking) segments.

        Returns segments in order. Empty segments are omitted.
        """
        combined = self._buffer + text
        self._buffer = ""
        segments: list[tuple[str, bool]] = []

        while combined:
            if self.in_thinking:
                end_idx = combined.find(self.THINK_END)
                if end_idx == -1:
                    # Check if buffer might contain a partial </think>
                    if len(combined) > self._MAX_TAG_LEN:
                        safe = combined[: -self._MAX_TAG_LEN]
                        self._buffer = combined[-self._MAX_TAG_LEN:]
                        if safe:
                            segments.append((safe, True))
                    else:
                        self._buffer = combined
                    break
                else:
                    thinking_text = combined[:end_idx]
                    if thinking_text:
                        segments.append((thinking_text, True))
                    combined = combined[end_idx + len(self.THINK_END):]
                    self.in_thinking = False
            else:
                start_idx = combined.find(self.THINK_START)
                if start_idx == -1:
                    # Check if buffer might contain a partial <think>
                    if len(combined) > self._MAX_TAG_LEN:
                        safe = combined[: -self._MAX_TAG_LEN]
                        self._buffer = combined[-self._MAX_TAG_LEN:]
                        if safe:
                            segments.append((safe, False))
                    else:
                        self._buffer = combined
                    break
                else:
                    content_text = combined[:start_idx]
                    if content_text:
                        segments.append((content_text, False))
                    combined = combined[start_idx + len(self.THINK_START):]
                    self.in_thinking = True

        return segments

    def flush(self) -> list[tuple[str, bool]]:
        """Flush any remaining buffered text at the end of stream."""
        if self._buffer:
            result = [(self._buffer, self.in_thinking)]
            self._buffer = ""
            return result
        return []
