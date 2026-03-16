import json
import logging
import re
from typing import Optional, List

from meno_core.core.rag.models import QueryRepresentations, RagMessage
from meno_core.core.rag.prompts import QUERY_REWRITE_SYSTEM_PROMPT
from meno_core.core.rag_engine import generate_with_llm

logger = logging.getLogger(__name__)


def extract_json_from_text(text: str) -> dict:
    """
    Attempts to extract and parse a JSON object from a potentially messy LLM output string.
    """
    # Try finding markdown code block
    json_block_re = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
    match = json_block_re.search(text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
            
    # Try finding highest level balanced curly braces
    json_obj_re = re.compile(r"\{[\s\S]*\}", re.MULTILINE)
    match = json_obj_re.search(text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
            
    # Fallback to empty if nothing works
    logger.warning(f"Could not parse valid JSON from LLM output: {text}")
    return {}


class QueryProcessor:
    """
    Handles rewriting the initial user query into an expanded JSON object
    containing abbreviations, coreferences, search queries, and a hypothetical doc.
    """
    
    async def process_query(self, query: str, history: List[RagMessage]) -> QueryRepresentations:
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
                system_prompt=QUERY_REWRITE_SYSTEM_PROMPT
            )
            
            # Parse the structured JSON output
            json_response = extract_json_from_text(response_text)
            
            # Construct the Pydantic representations object, mapping missing keys safely
            representations = QueryRepresentations(
                original_query=query,
                rewritten_query=json_response.get("rewritten_query", query),
                resolved_coreferences=json_response.get("resolved_coreferences", query),
                expanded_abbreviations=json_response.get("expanded_abbreviations", []),
                search_queries=json_response.get("search_queries", [query]),
                hypothetical_document=json_response.get("hypothetical_document", ""),
                is_meaningful=json_response.get("is_meaningful", True)
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
