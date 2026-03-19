import json
import logging
import re
from typing import List

from meno_core.core.rag.models import QueryRepresentations, RagMessage
from meno_core.core.rag.prompts import QUERY_REWRITE_SYSTEM_PROMPT
from meno_core.core.rag_engine import generate_with_llm

logger = logging.getLogger(__name__)


def _parse_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return default


def _parse_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        normalized = item.strip()
        if normalized:
            result.append(normalized)
    return result


def _safe_text(value: object, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def extract_json_from_text(text: str) -> dict:
    """
    Attempts to extract and parse a JSON object from a potentially messy LLM output string.
    """
    if not isinstance(text, str) or not text.strip():
        return {}

    # Try finding markdown code block
    json_block_re = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
    match = json_block_re.search(text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    for start in (idx for idx, char in enumerate(text) if char == "{"):
        try:
            parsed, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    # Fallback to empty if nothing works
    logger.warning("Could not parse valid JSON from LLM output (length=%s)", len(text))
    return {}


class QueryProcessor:
    """
    Handles rewriting the initial user query into an expanded JSON object
    containing abbreviations, coreferences, search queries, and a hypothetical doc.
    """
    
    async def process_query(self, query: str, history: List[RagMessage], override_model: Optional[str] = None) -> QueryRepresentations:
        """
        Takes the user query and dialogue history and calls the LLM to get a structured rewriting.
        """
        # Build prompt from conversation history and new query
        history_text = ""
        for msg in history[-6:]:  # limit to last 6 messages
            history_text += f"{msg.role}: {msg.text}\n"
            
        user_prompt = f"### Conversation History:\n{history_text}\n" if history_text else ""
        user_prompt += f"### User Query: {query}\n"
        
        try:
            # We use the existing shared llm func from rag_engine
            # Alternatively we could use `llm_client.py` if we wanted to isolate it further.
            response_text = await generate_with_llm(
                prompt=user_prompt,
                system_prompt=QUERY_REWRITE_SYSTEM_PROMPT,
                override_model=override_model
            )
            
            # Parse the structured JSON output
            json_response = extract_json_from_text(response_text)
            
            # Construct the Pydantic representations object, mapping missing keys safely
            rewritten_query = _safe_text(json_response.get("rewritten_query"), query)
            resolved_coreferences = _safe_text(json_response.get("resolved_coreferences"), rewritten_query)
            search_queries = _parse_string_list(json_response.get("search_queries"))
            if not search_queries:
                search_queries = [rewritten_query]

            representations = QueryRepresentations(
                original_query=query,
                rewritten_query=rewritten_query,
                resolved_coreferences=resolved_coreferences,
                expanded_abbreviations=_parse_string_list(json_response.get("expanded_abbreviations")),
                search_queries=search_queries,
                hypothetical_document=_safe_text(json_response.get("hypothetical_document"), ""),
                is_meaningful=_parse_bool(json_response.get("is_meaningful"), True)
            )
            return representations
            
        except Exception as e:
            logger.error(f"Error during query processing: {e}", exc_info=True)
            # Safe Fallback
            return QueryRepresentations(
                original_query=query,
                rewritten_query=query,
                resolved_coreferences=query,
                search_queries=[query],
                is_meaningful=True
            )
