from typing import Protocol, runtime_checkable

from meno_core.core.rag.models import RagRequest, RagResponse


@runtime_checkable
class RagMode(Protocol):
    """
    Standard interface for all RAG modes (e.g. graph-rag, chunk-rag).
    """

    async def answer(self, request: RagRequest) -> RagResponse:
        """
        Process a generic RagRequest and return a standardized RagResponse.
        """
        ...
