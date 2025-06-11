from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import add_messages
from typing_extensions import Annotated


import operator
from dataclasses import dataclass, field
from typing_extensions import Annotated


class OverallState(TypedDict):
    messages: Annotated[list, add_messages]
    search_query: Annotated[list, operator.add]
    web_research_result: Annotated[list, operator.add]
    sources_gathered: Annotated[list, operator.add]
    initial_search_query_count: int
    max_research_loops: int
    research_loop_count: int
    reasoning_model: str
    research_mode: str  # 'web' or 'rag'
    research_result: Annotated[list, operator.add]
    id: Annotated[list, operator.add]
    # RAG attribution fields
    rag_attributions: Annotated[list, operator.add]
    rag_retrieval_contents: Annotated[list, operator.add]
    rag_message_id: Annotated[list, operator.add]
    rag_agent_id: Annotated[list, operator.add]


class ReflectionState(TypedDict):
    is_sufficient: bool
    knowledge_gap: str
    follow_up_queries: Annotated[list, operator.add]
    research_loop_count: int
    number_of_ran_queries: int
    research_mode: str  # Add this field to preserve research_mode


class Query(TypedDict):
    query: str
    rationale: str


class QueryGenerationState(TypedDict):
    query_list: list[Query]
    research_mode: str  # Add research_mode field so fan_out_queries can access it


class WebSearchState(TypedDict):
    search_query: Annotated[list, operator.add]
    id: Annotated[list, operator.add]


@dataclass(kw_only=True)
class SearchStateOutput:
    running_summary: str = field(default=None)  # Final report
