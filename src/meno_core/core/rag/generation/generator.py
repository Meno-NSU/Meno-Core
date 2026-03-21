import logging
import random
import asyncio
from typing import List, Tuple, Optional

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
        override_base_url: Optional[str] = None
    ) -> Tuple[str, bool]:
        """
        Generate answer based on question and context.
        Returns:
            answer_text: generated string
            insufficient_information: boolean flag
        """
        if not context.strip():
            return "К сожалению, в базе данных недостаточно информации для ответа на этот вопрос.", True

        history_msgs = [{"role": m.role, "content": m.text} for m in history]

        if self.reliability_mode_enabled and not stream:
            return await self._generate_with_reliability_fallback(question, context, history_msgs, override_model=override_model, override_base_url=override_base_url)

        # Standard generation
        prompt = RAG_ANSWER_SYSTEM_PROMPT.format(context=context, question=question)

        answer = await call_llm(
            prompt=prompt,
            history_messages=history_msgs,
            stream=stream,
            override_model=override_model,
            override_base_url=override_base_url
        )

        # Check hallucination or insufficient info
        insuff_info = "недостаточно информации" in answer.lower()
        if not insuff_info:
            is_hallucinating, _ = await is_likely_hallucination(question, answer, self.hallucination_threshold)
            if is_hallucinating:
                return "Ответ извлечен, но может быть неточным или недостаточно обоснован в контексте.", True

        return answer, insuff_info

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
            return await call_llm(prompt=prompt, history_messages=history_msgs, stream=False, override_model=override_model, override_base_url=override_base_url)

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

        final_answer = await call_llm(prompt=agg_prompt, stream=False, override_model=override_model, override_base_url=override_base_url)
        return final_answer, False
