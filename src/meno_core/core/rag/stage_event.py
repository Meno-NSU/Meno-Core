"""Helpers for streaming ``<stage>`` events through the SSE response.

``<stage>`` tags carry per-step telemetry visible to the frontend while
``<think>`` remains reserved for native LLM chain-of-thought output.
"""

from __future__ import annotations

import re

__all__ = [
    "stage_event",
    "stage_started",
    "strip_stage_tags",
    "STAGE_TAG_RE",
]

STAGE_TAG_RE: re.Pattern[str] = re.compile(r"<stage\b[^>]*>.*?</stage>", re.DOTALL)


def stage_event(name: str, duration_ms: float, summary: str) -> str:
    """Return a completed ``<stage>`` tag with *duration_ms* and a short *summary*."""
    return f'<stage name="{name}" status="complete" duration_ms="{duration_ms:.1f}">{summary}</stage>'


def stage_started(name: str, summary: str) -> str:
    """Return a ``<stage>`` tag indicating a step has just started (no duration yet)."""
    return f'<stage name="{name}" status="started">{summary}</stage>'


def strip_stage_tags(text: str) -> str:
    """Remove all ``<stage …>…</stage>`` blocks from *text*."""
    return STAGE_TAG_RE.sub("", text).lstrip()
