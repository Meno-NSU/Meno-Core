import logging
from collections.abc import AsyncIterator
from typing import Any, List, Optional, Union

from meno_core.config.settings import settings
from meno_core.core.rag_engine import _current_model_override
from lightrag.llm.openai import openai_complete_if_cache # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


async def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Any]] = None,
    stream: bool = False,
    override_model: Optional[str] = None,
    **kwargs: Any
) -> Union[str, AsyncIterator[str]]:
    """
    Dedicated LLM client helper for the Chunk RAG module.
    Wraps around the shared openai_complete_if_cache for vLLM usage.
    """
    if history_messages is None:
        history_messages = []
        
    effective_model = override_model or _current_model_override.get() or settings.llm_model_name
    
    try:
        # Strip chain-of-thought elements handled mostly if model output format requires it.
        # This mirrors rag_engine's main complete function behavior.
        result = await openai_complete_if_cache(
            model=effective_model,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            enable_cot=False,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            stream=stream,
            **kwargs,
        )
        
        if stream:
            if isinstance(result, str):
                async def _to_stream() -> AsyncIterator[str]:
                    yield result
                return _to_stream()
            return result
            
        else:
            if not isinstance(result, str):
                chunks: list[str] = []
                async for part in result:
                    chunks.append(part)
                result = "".join(chunks)
                
            # Filter <think> blocks common with Qwen reasoning models 
            thinking_end_position = result.find('</think>')
            if thinking_end_position >= 0:
                result = result[thinking_end_position + len('</think>'):]
                
            return result.strip()
            
    except Exception as e:
        logger.error(f"Error in chunk RAG LLM call: {e}", exc_info=True)
        return "Извините, сейчас не удалось получить ответ от модели."
