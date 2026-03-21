import logging
from collections.abc import AsyncIterator
from typing import Any, List, Optional, Union

from meno_core.core.rag_engine import call_openai_llm

logger = logging.getLogger(__name__)


async def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Any]] = None,
    stream: bool = False,
    override_model: Optional[str] = None,
    override_base_url: Optional[str] = None,
    preserve_thinking: bool = False,
    **kwargs: Any
) -> Union[str, AsyncIterator[str]]:
    """
    Dedicated LLM client helper for the Chunk RAG module.
    Delegates to the shared low-level LLM wrapper used by the runtime.
    """
    try:
        return await call_openai_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            stream=stream,
            enable_cot=False,
            override_model=override_model,
            override_base_url=override_base_url,
            preserve_thinking=preserve_thinking,
            **kwargs,
        )
    except Exception as e:
        logger.error(f"Error in chunk RAG LLM call: {e}", exc_info=True)
        return "Извините, сейчас не удалось получить ответ от модели."
