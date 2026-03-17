from abc import ABC, abstractmethod
from typing import List, Sequence

from meno_core.core.rag.models import RetrievedChunk


class BaseRetriever(ABC):
    """
    Abstract base class for all RAG retrievers.
    """

    async def retrieve(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """
        Retrieve the top_k most relevant chunks for the given query.
        """
        results = await self.retrieve_many([query], top_k)
        return results[0] if results else []

    @abstractmethod
    async def retrieve_many(self, queries: Sequence[str], top_k: int) -> List[List[RetrievedChunk]]:
        """
        Retrieve top_k chunks for each query in the same input order.
        """
        pass
