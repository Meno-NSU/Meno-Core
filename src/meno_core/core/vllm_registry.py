"""
vLLM Model Registry — discovers available models across multiple vLLM endpoints.

Each configured endpoint is queried for its ``/v1/models`` list.
Results are cached in-memory and can be refreshed on demand.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Type alias for a single model record exposed by the registry.
ModelRecord = Dict[str, Any]


class VLLMRegistry:
    """Aggregates model lists from one or more vLLM-compatible servers."""

    def __init__(
            self,
            endpoints: List[str],
            *,
            timeout: float = 5.0,
            cache_ttl: float = 300.0,
    ) -> None:
        """
        Args:
            endpoints: base URLs of vLLM servers, e.g. ``["http://127.0.0.1:9020"]``.
            timeout:   per-request timeout in seconds.
            cache_ttl: how many seconds until the cached model list is considered stale.
        """
        # Normalise: strip trailing slashes
        self._endpoints: List[str] = [ep.rstrip("/") for ep in endpoints]
        self._timeout: float = timeout
        self._cache_ttl: float = cache_ttl

        self._cache: List[ModelRecord] = []
        self._cache_ts: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def discover(self) -> List[ModelRecord]:
        """Query every endpoint and rebuild the cache."""
        models: List[ModelRecord] = []
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for base_url in self._endpoints:
                url = f"{base_url}/v1/models"
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    body = resp.json()
                    for m in body.get("data", []):
                        record: ModelRecord = {
                            "id": m.get("id", "unknown"),
                            "object": "model",
                            "created": m.get("created", int(time.time())),
                            "owned_by": m.get("owned_by", "vllm"),
                            "endpoint": base_url,
                        }
                        models.append(record)
                    logger.info(
                        "Discovered %d model(s) on %s", len(body.get("data", [])), base_url
                    )
                except httpx.HTTPStatusError as exc:
                    logger.warning(
                        "vLLM endpoint %s returned HTTP %s: %s",
                        base_url, exc.response.status_code, exc.response.text[:200],
                    )
                except httpx.ConnectError:
                    logger.warning("Cannot connect to vLLM endpoint %s", base_url)
                except Exception:
                    logger.exception("Unexpected error querying vLLM endpoint %s", base_url)

        self._cache = models
        self._cache_ts = time.monotonic()
        return models

    async def list_models(self) -> List[ModelRecord]:
        """Return cached models, refreshing automatically if stale."""
        if not self._cache or (time.monotonic() - self._cache_ts) > self._cache_ttl:
            return await self.discover()
        return self._cache

    async def refresh(self) -> List[ModelRecord]:
        """Force-refresh the cache and return fresh data."""
        return await self.discover()

    async def is_valid_model(self, model_id: str) -> bool:
        """Return True only if *model_id* is present in the known vLLM model list.

        Uses the auto-refreshing cache so no extra network round-trip is made
        when the cache is still fresh.
        """
        models = await self.list_models()
        return any(m["id"] == model_id for m in models)

    def lookup_endpoint(self, model_id: str) -> Optional[str]:
        """Return the base URL for *model_id*, or ``None`` if not found."""
        for m in self._cache:
            if m["id"] == model_id:
                return m["endpoint"]
        return None
