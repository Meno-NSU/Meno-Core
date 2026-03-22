import logging
import random
import asyncio
from collections.abc import AsyncIterator
from typing import List, Tuple, Optional, Union

from meno_core.core.rag.models import RagMessage, RagSource
from meno_core.core.rag.prompts import RAG_ANSWER_SYSTEM_PROMPT, FALLBACK_AGGREGATION_PROMPT
from meno_core.core.rag.llm_client import call_llm
from meno_core.core.rag_engine import is_likely_hallucination

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """
    Generates the final answer using strictly structured prompt instructions.
    Includes an optional fallback reliability mode (multi-answer aggregation).
    """

    def __init__(
        self,
        reliability_mode_enabled: bool = False,
        hallucination_threshold: float = 0.4
    ):
        self.reliability_mode_enabled = reliability_mode_enabled
        self.hallucination_threshold = hallucination_threshold

    async def generate_answer(
        self,
        question: str,
        context: str,
        sources: List[RagSource],
        history: List[RagMessage],
        stream: bool = False,
        override_model: Optional[str] = None,
        override_base_url: Optional[str] = None,
    ) -> Union[Tuple[str, bool], Tuple[AsyncIterator[str], bool]]:
        """Generate answer based on question and context.

        When *stream* is ``False`` returns ``(answer_text, insufficient_information)``.
        When *stream* is ``True`` returns ``(async_iterator, False)`` — hallucination
        check is skipped because the full text is not available yet.
        """
        if not context.strip():
            if stream:
                async def _empty_stream() -> AsyncIterator[str]:
                    yield "К сожалению, в базе данных недостаточно информации для ответа на этот вопрос."
                return _empty_stream(), True
            return "К сожалению, в базе данных недостаточно информации для ответа на этот вопрос.", True

        history_msgs = [{"role": m.role, "content": m.text} for m in history]

        if self.reliability_mode_enabled and not stream:
            return await self._generate_with_reliability_fallback(question, context, history_msgs, override_model=override_model, override_base_url=override_base_url)

        # Standard generation
        prompt = RAG_ANSWER_SYSTEM_PROMPT.format(context=context, question=question)

        result = await call_llm(
            prompt=prompt,
            history_messages=history_msgs,
            stream=stream,
            override_model=override_model,
            override_base_url=override_base_url,
            preserve_thinking=True,
        )

        # Streaming: wrap the iterator to run a post-stream hallucination check.
        # After all tokens are yielded, we check the accumulated text and yield
        # a warning suffix if hallucination is detected.
        if stream:
            return self._wrap_stream_with_hallucination_check(
                result,  # type: ignore[arg-type]
                question,
            ), False

        answer = str(result) if not isinstance(result, str) else result

        # Strip <think> for hallucination check but keep it in the answer
        from meno_core.core.rag_engine import _strip_reasoning_prefix
        answer_for_check = _strip_reasoning_prefix(answer)

        # Check hallucination or insufficient info
        insuff_info = "недостаточно информации" in answer_for_check.lower()
        if not insuff_info:
            is_hallucinating, _ = await is_likely_hallucination(question, answer_for_check, self.hallucination_threshold)
            if is_hallucinating:
                return "Ответ извлечен, но может быть неточным или недостаточно обоснован в контексте.", True

        return answer, insuff_info

    async def _wrap_stream_with_hallucination_check(
        self,
        token_iter: AsyncIterator[str],
        question: str,
    ) -> AsyncIterator[str]:
        """Yield all tokens from the LLM stream, then run hallucination check on
        the accumulated answer. If hallucination is detected, yield a warning."""
        from meno_core.core.rag_engine import _strip_reasoning_prefix

        collected: list[str] = []
        async for token in token_iter:
            collected.append(str(token))
            yield str(token)

        full_answer = "".join(collected)
        answer_for_check = _strip_reasoning_prefix(full_answer)

        if "недостаточно информации" in answer_for_check.lower():
            return

        try:
            is_hallucinating, _ = await is_likely_hallucination(
                question, answer_for_check, self.hallucination_threshold
            )
            if is_hallucinating:
                yield "\n\n⚠️ *Внимание: ответ может быть неточным или недостаточно обоснован в контексте.*"
        except Exception:
            logger.warning("Post-stream hallucination check failed", exc_info=True)

    async def _generate_with_reliability_fallback(
        self,
        question: str,
        context: str,
        history_msgs: list,
        override_model: Optional[str] = None,
        override_base_url: Optional[str] = None
    ) -> Tuple[str, bool]:
        """
        Implements fallback by shuffling context and generating multiple candidates,
        then aggregating them.
        """
        logger.info("Executing Fallback Reliability Generation Mode...")

        context_blocks = context.split("\n\n---\n\n")

        async def _gen(blocks: List[str]) -> str:
            shuffled_context = "\n\n---\n\n".join(blocks)
            prompt = RAG_ANSWER_SYSTEM_PROMPT.format(context=shuffled_context, question=question)
            result = await call_llm(prompt=prompt, history_messages=history_msgs, stream=False, override_model=override_model, override_base_url=override_base_url)
            return str(result) if not isinstance(result, str) else result

        # Keep original order for Candidate 1
        tasks = [_gen(context_blocks)]

        # Shuffle for Candidate 2 and 3
        for _ in range(2):
            shuffled = context_blocks.copy()
            random.shuffle(shuffled)
            tasks.append(_gen(shuffled))

        candidates = await asyncio.gather(*tasks)

        all_insufficient = all("недостаточно информации" in c.lower() for c in candidates)
        if all_insufficient:
            return "К сожалению, в базе данных недостаточно информации для ответа на этот вопрос.", True

        formatted_candidates = "\n\n".join([f"Candidate {i+1}:\n{c}" for i, c in enumerate(candidates)])
        agg_prompt = FALLBACK_AGGREGATION_PROMPT.format(question=question, candidate_answers=formatted_candidates)

        final_result = await call_llm(prompt=agg_prompt, stream=False, override_model=override_model, override_base_url=override_base_url, preserve_thinking=True)
        final_answer = str(final_result) if not isinstance(final_result, str) else final_result
        return final_answer, False
