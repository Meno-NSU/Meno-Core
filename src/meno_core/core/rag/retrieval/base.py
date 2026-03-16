from abc import ABC, abstractmethod
from typing import List

from meno_core.core.rag.models import RetrievedChunk


class BaseRetriever(ABC):
    """
    Abstract base class for all RAG retrievers.
    """

    @abstractmethod
    async def retrieve(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """
        Retrieve the top_k most relevant chunks for the given query.
        """
        pass
