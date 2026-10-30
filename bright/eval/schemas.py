"""Tool schemas for the ranking agent."""

from __future__ import annotations

from typing import Any

from session import MAX_RANKED_DOCUMENTS

SEARCH_TOOL_DESCRIPTION = """
Semantic search over passage text. Returns up to 20 chunks, most relevant first.
Write a question or a sentence, not a list of keywords.
""".strip()

SEARCH_TEXT_DESCRIPTION = "A question or a sentence describing the content you need."

GET_DOCUMENT_TOOL_DESCRIPTION = """
Read the full text of a passage already returned by search. Search results can
show only part of a long passage.
""".strip()

SUBMIT_RANKING_TOOL_DESCRIPTION = """
Submit a ranked list of retrieved passages.

Use only shown document ids. Ranking is by relevance_score from 0 to 1, not array order.
Up to ten passages, the most relevant first.
""".strip()

RETRIEVAL_TOOL_NAMES = ("search", "get_document")


def tool_schemas(max_ranked_docids: int = MAX_RANKED_DOCUMENTS) -> list[dict[str, Any]]:
    ranked_cap = max(1, int(max_ranked_docids or MAX_RANKED_DOCUMENTS))
    return [
        {
            "name": "search",
            "description": SEARCH_TOOL_DESCRIPTION,
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text"],
                "properties": {
                    "text": {"type": "string", "description": SEARCH_TEXT_DESCRIPTION},
                },
            },
        },
        {
            "name": "get_document",
            "description": GET_DOCUMENT_TOOL_DESCRIPTION,
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["docid"],
                "properties": {
                    "docid": {
                        "type": "string",
                        "description": "Document id copied from a search result.",
                    },
                },
            },
        },
        {
            "name": "submit_ranking",
            "description": SUBMIT_RANKING_TOOL_DESCRIPTION,
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["ranking_strategy", "documents"],
                "properties": {
                    "ranking_strategy": {
                        "type": "string",
                        "description": "Why these documents are relevant.",
                    },
                    "documents": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": ranked_cap,
                        "description": f"Up to {ranked_cap} document ids, most relevant first.",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["docid", "relevance_score"],
                            "properties": {
                                "docid": {
                                    "type": "string",
                                    "description": "Document id copied from a previous result.",
                                },
                                "relevance_score": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                    "description": "Relevance from 0 to 1. Higher ranks first.",
                                },
                            },
                        },
                    },
                },
            },
        },
    ]


def search_phase_tools(*, can_rank: bool) -> list[str]:
    """Tools offered while retrieving. Ranking opens once something was shown."""
    names = list(RETRIEVAL_TOOL_NAMES)
    if can_rank:
        names.append("submit_ranking")
    return names


def bedrock_tool_config(
    names: list[str],
    *,
    tool_choice: str | None = None,
    max_ranked_docids: int = MAX_RANKED_DOCUMENTS,
) -> dict[str, Any]:
    """Bedrock Converse toolConfig for the named tools."""
    by_name = {tool["name"]: tool for tool in tool_schemas(max_ranked_docids)}
    config: dict[str, Any] = {
        "tools": [
            {
                "toolSpec": {
                    "name": by_name[name]["name"],
                    "description": by_name[name]["description"],
                    "inputSchema": {"json": by_name[name]["inputSchema"]},
                }
            }
            for name in names
        ]
    }
    if tool_choice:
        config["toolChoice"] = {"tool": {"name": tool_choice}}
    else:
        config["toolChoice"] = {"auto": {}}
    return config
