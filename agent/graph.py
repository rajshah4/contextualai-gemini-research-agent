from dotenv import load_dotenv
load_dotenv()

import os
import re
import operator
from typing import Annotated, TypedDict

from agent.tools_and_schemas import SearchQueryList, Reflection
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from google.genai import Client

from contextual import ContextualAI

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
    query_rag_v3,
    tool_selection,
)


if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")

# Used for Google Search API
genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))


# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries and determines research mode per query."""
    print(f"Debug: Generate Query - Topic: {get_research_topic(state['messages'])}")
    configurable = Configuration.from_runnable_config(config)

    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    llm = ChatGoogleGenerativeAI(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    result = structured_llm.invoke(formatted_prompt)
    print(f"Debug: Generated {len(result.query)} queries: {result.query}")

    # Per-query tool selection
    query_mode_list = []
    for q in result.query:
        # Use the same tool_selection logic, but for each query
        tool_sel_state = {"messages": [HumanMessage(content=q)]}
        tool_sel_result = tool_selection(tool_sel_state, config)
        research_mode = tool_sel_result.get("research_mode", "web")
        print(f"Debug: Tool selection for query '{q}': {research_mode}")
        query_mode_list.append({"query": q, "research_mode": research_mode})

    return {"query_mode_list": query_mode_list}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    # Add debug statement to track state in continue_to_web_research
    print(f"Debug: Continue to Web Research - {len(state['query_list'])} queries")
    
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["query_list"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """Performs web research and returns the raw text and citation data."""
    search_query = state["search_query"][0] if isinstance(state["search_query"], list) else state["search_query"]
    print(f"Debug: Web Research - Query: {search_query}")
    configurable = Configuration.from_runnable_config(config)
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=search_query,
    )
    response = genai_client.models.generate_content(
        model=configurable.query_generator_model,
        contents=formatted_prompt,
        config={
            "tools": [{"google_search": {}}],
            "temperature": 0,
        },
    )
    id_value = state["id"][0] if isinstance(state["id"], list) and state["id"] else 0
    resolved_urls = resolve_urls(
        response.candidates[0].grounding_metadata.grounding_chunks, id_value
    )
    
    # Get the raw citation data, but do NOT insert markers here
    citations = get_citations(response, resolved_urls)
    
    # The research result is now a dictionary containing the raw data
    research_data = {
        "text": response.text,
        "citations": citations, # Pass the raw citation data
        "source_type": "web"
    }
    
    sources_gathered = [item for citation in citations for item in citation["segments"]]
    print(f"Debug: Web Research completed - {len(sources_gathered)} sources found.")
    return {
        "sources_gathered": state.get("sources_gathered", []) + sources_gathered,
        "research_result": state.get("research_result", []) + [research_data],
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format. It then determines the best research mode for each
    follow-up query.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including a list of follow-up queries with their
        determined research modes.
    """
    # Debug: Print message types and values
    messages = state.get("messages", [])
    print(f"Debug: reflection called with {len(messages)} messages.")
    for idx, msg in enumerate(messages):
        print(f"  Message {idx}: type={type(msg)}, value={msg}")
    try:
        configurable = Configuration.from_runnable_config(config)
        # Increment the research loop count and get the reasoning model
        state["research_loop_count"] = state.get("research_loop_count", 0) + 1
        reasoning_model = state.get("reasoning_model") or configurable.reasoning_model

        print(f"Debug: Reflection - Loop {state['research_loop_count']}")

        # Format the prompt
        current_date = get_current_date()
        research_summary_text = "\n\n---\n\n".join([item['text'] for item in state["research_result"]])
        formatted_prompt = reflection_instructions.format(
            current_date=current_date,
            research_topic=get_research_topic(state["messages"]),
            summaries=research_summary_text,
        )
        # init Reasoning Model
        llm = ChatGoogleGenerativeAI(
            model=reasoning_model,
            temperature=1.0,
            max_retries=2,
            api_key=os.getenv("GEMINI_API_KEY"),
        )
        result = llm.with_structured_output(Reflection).invoke(formatted_prompt)

        print(f"Debug: Reflection Result - Sufficient: {result.is_sufficient}, Follow-ups: {len(result.follow_up_queries)}")
        if result.knowledge_gap:
            print(f"Debug: Knowledge Gap: {result.knowledge_gap[:100]}..." if len(result.knowledge_gap) > 100 else f"Debug: Knowledge Gap: {result.knowledge_gap}")

        # Determine research mode for each follow-up query
        follow_up_query_mode_list = []
        for q in result.follow_up_queries:
            tool_sel_state = {"messages": [HumanMessage(content=q)]}
            tool_sel_result = tool_selection(tool_sel_state, config)
            research_mode = tool_sel_result.get("research_mode", "web")
            print(f"Debug: Tool selection for follow-up query '{q}': {research_mode}")
            follow_up_query_mode_list.append({"query": q, "research_mode": research_mode})

        return {
            "is_sufficient": result.is_sufficient,
            "knowledge_gap": result.knowledge_gap,
            "follow_up_query_mode_list": follow_up_query_mode_list,
            "research_loop_count": state["research_loop_count"],
            "number_of_ran_queries": len(state.get("search_query", [])),
        }
    except Exception as e:
        print(f"Debug: Error in reflection: {e}")
        import traceback
        traceback.print_exc()
        # Return a safe fallback - mark research as sufficient to end the loop
        return {
            "is_sufficient": True,
            "knowledge_gap": "Error occurred during reflection",
            "follow_up_query_mode_list": [],
            "research_loop_count": state.get("research_loop_count", 0) + 1,
            "number_of_ran_queries": len(state.get("search_query", [])),
        }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> list[Send]:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary. It now routes each follow-up query to the
    appropriate research node individually. If no follow-up queries are generated,
    it proceeds to finalize the answer.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        A list of Send objects, each directing a query to a specific research node,
        or a string literal "prepare_final_answer" to end the research loop.
    """
    try:
        configurable = Configuration.from_runnable_config(config)
        max_research_loops = (
            state.get("max_research_loops")
            if state.get("max_research_loops") is not None
            else configurable.max_research_loops
        )
        follow_up_query_mode_list = state.get("follow_up_query_mode_list", [])
        
        print(f"Debug: Evaluate Research - Loop {state['research_loop_count']}/{max_research_loops}, Sufficient: {state['is_sufficient']}, Follow-ups: {len(follow_up_query_mode_list)}")
        
        if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops or not follow_up_query_mode_list:
            print("Debug: Routing to prepare_final_answer")
            return "prepare_final_answer"
        else:
            sends = []
            print(f"Debug: Routing {len(follow_up_query_mode_list)} follow-up queries to appropriate research nodes.")
            for idx, item in enumerate(follow_up_query_mode_list):
                query = item["query"]
                research_mode = item["research_mode"]
                target_node = "rag_research" if research_mode == "rag" else "web_research"
                
                print(f"Debug: Routing query '{query}' to {target_node}")
                sends.append(
                    Send(
                        target_node,
                        {
                            "search_query": [query],
                            "id": [state["number_of_ran_queries"] + int(idx)],
                        },
                    )
                )
            return sends
    except Exception as e:
        print(f"Debug: Error in evaluate_research: {e}")
        import traceback
        traceback.print_exc()
        # Return a safe fallback
        return "prepare_final_answer"


def prepare_final_answer(state: OverallState, config: RunnableConfig):
    """Prepares the final answer by unifying sources and inserting all inline citations."""
    try:
        print(f"Debug: Preparing final answer for topic: {get_research_topic(state['messages'])}")

        research_items = state.get("research_result", [])
        
        # --- Unified Source and Citation Processing ---
        final_text_parts = []
        final_sources = {}
        citation_counter = 1

        # Create a unified map of all possible sources
        rag_contents_map = {c['content_id']: c for c in state.get('rag_retrieval_contents', [])}
        web_sources_map = {s['value']: s for s in state.get('sources_gathered', [])}

        for item in research_items:
            text = item['text']
            source_type = item['source_type']
            
            citations_to_process = []
            if source_type == 'rag':
                citations_to_process = item.get('attributions', [])
            elif source_type == 'web':
                citations_to_process = item.get('citations', [])

            # Sort citations by end_index descending to insert markers from the end
            sorted_citations = sorted(citations_to_process, key=lambda x: x.get('end_idx', x.get('end_index', 0)), reverse=True)

            for cit in sorted_citations:
                # Determine the unique identifier(s) for the source(s) in this citation
                content_ids = []
                if source_type == 'rag':
                    content_ids = cit.get('content_ids', [])
                elif source_type == 'web':
                    content_ids = [segment['value'] for segment in cit.get('segments', [])]
                
                if not content_ids:
                    continue

                markers_to_insert = ""
                for content_id in sorted(list(set(content_ids))): # Sort for consistent marker order
                    if content_id not in final_sources:
                        final_sources[content_id] = citation_counter
                        citation_counter += 1
                    
                    marker_num = final_sources[content_id]
                    if f"[{marker_num}]" not in markers_to_insert:
                        markers_to_insert += f"[{marker_num}]"
                
                end_index = cit.get('end_idx', cit.get('end_index', 0))
                text = text[:end_index] + markers_to_insert + text[end_index:]
            
            # --- Cleanup Step ---
            # Remove any old citation markers that might be left over
            text = re.sub(r'\[\d+\]\(\)', '', text) # Removes markers like [1]()
            text = re.sub(r'\[\d+\]', '', text) # Removes markers like [1]
            
            final_text_parts.append(text)

        # --- Format the final source list ---
        formatted_sources = []
        sorted_final_sources = sorted(final_sources.items(), key=lambda item: item[1])

        for identifier, number in sorted_final_sources:
            if identifier in rag_contents_map:
                data = rag_contents_map[identifier]
                formatted_sources.append(f"[{number}]: RAG Source - Document: {data.get('doc_name', 'N/A')}, Page: {data.get('page', 'N/A')}, Content ID: {data.get('content_id', 'N/A')}\n")
            elif identifier in web_sources_map:
                data = web_sources_map[identifier]
                formatted_sources.append(f"[{number}]: Web Source: {data.get('value', 'N/A')}\n")

        # --- Message and Kwargs Preparation ---
        final_message = AIMessage(content="\n\n".join(final_text_parts))
        final_message.additional_kwargs = {
            "formatted_sources": "\n".join(formatted_sources),
            "rag_message_id": state.get("rag_message_id", []),
            "rag_agent_id": state.get("rag_agent_id", [])
        }
        
        print(f"Debug: Answer prepared with {len(final_sources)} unique, re-numbered sources.")
        
        return {"messages": [final_message]}
    except Exception as e:
        print(f"Debug: Error in prepare_final_answer: {e}")
        import traceback
        traceback.print_exc()
        return {"messages": [AIMessage(content="I encountered an error while preparing the answer.")]}


def synthesize_final_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that synthesizes the final answer and appends sources.

    This node takes the user's original question and all the gathered research,
    synthesizes a direct, final answer, and then appends a formatted list of
    sources to the end of the answer.

    Args:
        state: Current graph state containing the user's question and the prepared research

    Returns:
        Dictionary with the final, synthesized answer message including sources.
    """
    try:
        print(f"Debug: Synthesizing final answer for topic: {get_research_topic(state['messages'])}")
        configurable = Configuration.from_runnable_config(config)
        synthesis_model = state.get("synthesis_model") or configurable.synthesis_model

        # The message from the previous step contains the combined research and all kwargs
        prepared_message = state["messages"][-1]
        
        # Format the prompt
        current_date = get_current_date()
        formatted_prompt = answer_instructions.format(
            current_date=current_date,
            research_topic=get_research_topic(state["messages"]),
            summaries=prepared_message.content,
            formatted_sources=prepared_message.additional_kwargs.get("formatted_sources", "")
        )
        # init Synthesis Model
        llm = ChatGoogleGenerativeAI(
            model=synthesis_model,
            temperature=0.7,
            max_retries=2,
            api_key=os.getenv("GEMINI_API_KEY"),
        )
        result = llm.invoke(formatted_prompt)

        # Append the formatted sources to the final answer
        synthesized_answer = result.content
        formatted_sources = prepared_message.additional_kwargs.get("formatted_sources", "")
        if formatted_sources:
            final_content = f"{synthesized_answer}\n\n**Sources:**\n{formatted_sources}"
        else:
            final_content = synthesized_answer

        # Update the content of the message, preserving the kwargs
        final_answer_message = prepared_message
        final_answer_message.content = final_content
        
        print(f"Debug: Final answer synthesized and sources appended.")
        
        return {"messages": [final_answer_message]}
    except Exception as e:
        print(f"Debug: Error in synthesize_final_answer: {e}")
        import traceback
        traceback.print_exc()
        # Return a safe fallback message
        fallback_message = AIMessage(content="I encountered an error while synthesizing the final answer. Please try again.")
        return {"messages": [fallback_message]}



# RAG research node
def rag_research(state: WebSearchState, config: RunnableConfig) -> dict:
    """LangGraph node that performs RAG research and places the raw results into the state."""
    prompt = state["search_query"][0] if isinstance(state["search_query"], list) else state["search_query"]
    print(f"Debug: RAG Research - Query: {prompt}")
    
    try:
        rag_result = query_rag_v3(prompt)
        
        # The research_result now contains a dictionary with the text and its attributions
        research_data = {
            "text": rag_result.get("content", ""),
            "attributions": rag_result.get("attributions", []),
            "source_type": "rag"
        }
        
        return {
            "research_result": state.get("research_result", []) + [research_data],
            "rag_retrieval_contents": state.get("rag_retrieval_contents", []) + rag_result.get("retrieval_contents", []),
            "rag_message_id": state.get("rag_message_id", []) + ([rag_result.get("message_id")] if rag_result.get("message_id") else []),
            "rag_agent_id": state.get("rag_agent_id", []) + ([rag_result.get("agent_id")] if rag_result.get("agent_id") else []),
        }
    except Exception as e:
        print(f"Debug: Error in rag_research: {e}")
        import traceback
        traceback.print_exc()
        # Return an error message in the research result
        return {"research_result": [{"text": f"Error during RAG research: {e}", "attributions": [], "source_type": "rag"}]}


# Fan-out conditional edge function to create individual research tasks for each query
def fan_out_queries(state: QueryGenerationState, config: RunnableConfig):
    """Takes the list of queries and emits a list of Send objects, each with a single search_query and id, routed per-query to the correct research node."""
    query_mode_list = state.get("query_mode_list", [])
    print(f"Debug: Fan-out queries - {len(query_mode_list)} queries with per-query research_mode")
    sends = []
    for idx, item in enumerate(query_mode_list):
        query = item["query"]
        research_mode = item["research_mode"]
        if research_mode == "web":
            target_node = "web_research"
        elif research_mode == "rag":
            target_node = "rag_research"
        else:
            target_node = "rag_research"  # fallback
        print(f"Debug: Routing query '{query}' to {target_node}")
        sends.append(Send(target_node, {"search_query": [query], "id": [idx]}))
    return sends


def clear_state(state: OverallState) -> OverallState:
    """Clears the research-related fields from the state for a new run."""
    print("Debug: Clearing state for new run.")
    state['research_result'] = []
    state['sources_gathered'] = []
    state['rag_attributions'] = []
    state['rag_retrieval_contents'] = []
    state['rag_message_id'] = []
    state['rag_agent_id'] = []
    state['knowledge_gap'] = ""
    state['follow_up_query_mode_list'] = []
    return state

# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Register nodes
builder.add_node("clear_state", clear_state)
builder.add_node("generate_query", generate_query)
builder.add_node("tool_selection", tool_selection)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("prepare_final_answer", prepare_final_answer)
builder.add_node("synthesize_final_answer", synthesize_final_answer)
builder.add_node("rag_research", rag_research)

# Set the entrypoint as `clear_state`
builder.add_edge(START, "clear_state")
builder.add_edge("clear_state", "generate_query")

# After generate_query, run tool_selection to set research_mode
builder.add_edge("generate_query", "tool_selection")
# After tool_selection, fan out directly to research nodes based on research_mode
builder.add_conditional_edges(
    "tool_selection", fan_out_queries, ["web_research", "rag_research"]
)
# Both research nodes go to reflection
builder.add_edge("web_research", "reflection")
builder.add_edge("rag_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "rag_research", "prepare_final_answer"]
)
# Prepare the final answer
builder.add_edge("prepare_final_answer", "synthesize_final_answer")
# Synthesize the final answer
builder.add_edge("synthesize_final_answer", END)

graph = builder.compile(name="pro-search-agent")