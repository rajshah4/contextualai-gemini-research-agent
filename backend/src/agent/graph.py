import os
import operator
from typing import Annotated, TypedDict

from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
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

load_dotenv()

if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")

# Used for Google Search API
genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))


# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates a search queries based on the User's question.

    Uses Gemini 2.0 Flash to create an optimized search query for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated query
    """
    # Add debug statement to track state in generate_query
    print(f"Debug: Generate Query - Topic: {get_research_topic(state['messages'])}")
    
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # init Gemini 2.0 Flash
    llm = ChatGoogleGenerativeAI(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)
    print(f"Debug: Generated {len(result.query)} queries: {result.query}")
    return {"query_list": result.query}


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
    """LangGraph node that performs web research using the native Google Search API tool.

    Executes a web search using the native Google Search API tool in combination with Gemini 2.0 Flash.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and research_result
    """
    # Extract the search query from the list
    search_query = state["search_query"][0] if isinstance(state["search_query"], list) else state["search_query"]
    print(f"Debug: Web Research - Query: {search_query}")
    
    # Configure
    configurable = Configuration.from_runnable_config(config)
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=search_query,
    )

    # Uses the google genai client as the langchain client doesn't return grounding metadata
    response = genai_client.models.generate_content(
        model=configurable.query_generator_model,
        contents=formatted_prompt,
        config={
            "tools": [{"google_search": {}}],
            "temperature": 0,
        },
    )
    # resolve the urls to short urls for saving tokens and time
    # Handle id as a list, take the first element
    id_value = state["id"][0] if isinstance(state["id"], list) and state["id"] else 0
    resolved_urls = resolve_urls(
        response.candidates[0].grounding_metadata.grounding_chunks, id_value
    )
    # Gets the citations and adds them to the generated text
    citations = get_citations(response, resolved_urls)
    modified_text = insert_citation_markers(response.text, citations)
    sources_gathered = [item for citation in citations for item in citation["segments"]]

    print(f"Debug: Web Research completed - {len(sources_gathered)} sources found")
    
    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "research_result": [modified_text],
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    try:
        configurable = Configuration.from_runnable_config(config)
        # Increment the research loop count and get the reasoning model
        state["research_loop_count"] = state.get("research_loop_count", 0) + 1
        reasoning_model = state.get("reasoning_model") or configurable.reasoning_model

        print(f"Debug: Reflection - Loop {state['research_loop_count']}, Mode: {state.get('research_mode', 'unknown')}")

        # Format the prompt
        current_date = get_current_date()
        formatted_prompt = reflection_instructions.format(
            current_date=current_date,
            research_topic=get_research_topic(state["messages"]),
            summaries="\n\n---\n\n".join(state["research_result"]),
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

        return {
            "is_sufficient": result.is_sufficient,
            "knowledge_gap": result.knowledge_gap,
            "follow_up_queries": result.follow_up_queries,
            "research_loop_count": state["research_loop_count"],
            "number_of_ran_queries": len(state["search_query"]),
            "research_mode": state.get("research_mode", "web"),
        }
    except Exception as e:
        print(f"Debug: Error in reflection: {e}")
        import traceback
        traceback.print_exc()
        # Return a safe fallback - mark research as sufficient to end the loop
        return {
            "is_sufficient": True,
            "knowledge_gap": "Error occurred during reflection",
            "follow_up_queries": [],
            "research_loop_count": state.get("research_loop_count", 0) + 1,
            "number_of_ran_queries": len(state.get("search_query", [])),
            "research_mode": state.get("research_mode", "web"),
        }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    try:
        configurable = Configuration.from_runnable_config(config)
        max_research_loops = (
            state.get("max_research_loops")
            if state.get("max_research_loops") is not None
            else configurable.max_research_loops
        )
        
        print(f"Debug: Evaluate Research - Loop {state['research_loop_count']}/{max_research_loops}, Sufficient: {state['is_sufficient']}")
        
        if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
            print("Debug: Routing to finalize_answer")
            return "finalize_answer"
        else:
            if state.get("research_mode") == "web":
                print(f"Debug: Routing to web_research with {len(state['follow_up_queries'])} follow-up queries")
                return [
                    Send(
                        "web_research",
                        {
                            "search_query": [follow_up_query],
                            "id": [state["number_of_ran_queries"] + int(idx)],
                        },
                    )
                    for idx, follow_up_query in enumerate(state["follow_up_queries"])
                ]
            elif state.get("research_mode") == "rag":
                print(f"Debug: Routing to rag_research with {len(state['follow_up_queries'])} follow-up queries")
                return [
                    Send(
                        "rag_research",
                        {
                            "search_query": [follow_up_query],
                            "id": [state["number_of_ran_queries"] + int(idx)],
                        },
                    )
                    for idx, follow_up_query in enumerate(state["follow_up_queries"])
                ]
            else:
                print(f"Debug: Unknown research_mode: {state.get('research_mode')}, routing to finalize_answer")
                return "finalize_answer"
    except Exception as e:
        print(f"Debug: Error in evaluate_research: {e}")
        import traceback
        traceback.print_exc()
        # Return a safe fallback
        return "finalize_answer"


def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    try:
        print(f"Debug: Finalizing answer for topic: {get_research_topic(state['messages'])}")
        
        configurable = Configuration.from_runnable_config(config)
        reasoning_model = state.get("reasoning_model") or configurable.reasoning_model

        # Format the prompt
        current_date = get_current_date()
        formatted_prompt = answer_instructions.format(
            current_date=current_date,
            research_topic=get_research_topic(state["messages"]),
            summaries="\n---\n\n".join(state["research_result"]),
        )

        # init Reasoning Model, default to Gemini 2.5 Flash
        llm = ChatGoogleGenerativeAI(
            model=reasoning_model,
            temperature=0,
            max_retries=2,
            api_key=os.getenv("GEMINI_API_KEY"),
        )
        result = llm.invoke(formatted_prompt)

        # Replace the short urls with the original urls and add all used urls to the sources_gathered
        unique_sources = []
        for source in state["sources_gathered"]:
            if source["short_url"] in result.content:
                result.content = result.content.replace(
                    source["short_url"], source["value"]
                )
                unique_sources.append(source)

        # Create the final AI message with RAG attribution data attached
        final_message = AIMessage(content=result.content)
        
        # Attach RAG attribution data to the message if it exists
        raw_rag_attributions = state.get("rag_attributions", [])
        raw_rag_retrieval_contents = state.get("rag_retrieval_contents", [])
        rag_message_id = next((mid for mid in state.get("rag_message_id", []) if mid), None)
        rag_agent_id = next((aid for aid in state.get("rag_agent_id", []) if aid), None)
        
        # Deduplicate retrieval contents by content_id to handle multiple queries
        seen_content_ids = set()
        deduplicated_retrieval_contents = []
        for rc in raw_rag_retrieval_contents:
            if rc.get("content_id") and rc["content_id"] not in seen_content_ids:
                seen_content_ids.add(rc["content_id"])
                deduplicated_retrieval_contents.append(rc)
        
        # Filter attributions to only include those that have corresponding retrieval_contents
        valid_attributions = [attr for attr in raw_rag_attributions if attr in seen_content_ids]
        
        print(f"Debug: RAG data processed - {len(raw_rag_attributions)} raw attributions -> {len(valid_attributions)} valid attributions, {len(raw_rag_retrieval_contents)} raw contents -> {len(deduplicated_retrieval_contents)} deduplicated contents")
        
        if valid_attributions or deduplicated_retrieval_contents:
            # Add RAG attribution data as additional kwargs for the frontend
            final_message.additional_kwargs = {
                "rag_attributions": valid_attributions,
                "rag_retrieval_contents": deduplicated_retrieval_contents,
                "rag_message_id": rag_message_id,
                "rag_agent_id": rag_agent_id,
            }
            print(f"Debug: Added RAG attribution data - {len(valid_attributions)} attributions, {len(deduplicated_retrieval_contents)} sources")

        print(f"Debug: Answer finalized with {len(unique_sources)} web sources")
        
        return {
            "messages": [final_message],
            "sources_gathered": unique_sources,
        }
    except Exception as e:
        print(f"Debug: Error in finalize_answer: {e}")
        import traceback
        traceback.print_exc()
        # Return a safe fallback message
        fallback_message = AIMessage(content="I encountered an error while finalizing the answer. Please try again.")
        return {
            "messages": [fallback_message],
            "sources_gathered": [],
        }


# RAG research node
def rag_research(state: WebSearchState, config: RunnableConfig) -> dict:
    """LangGraph node that performs RAG research using the query_rag utility."""
    # Use the search_query as the prompt
    prompt = state["search_query"][0] if isinstance(state["search_query"], list) else state["search_query"]
    print(f"Debug: RAG Research - Query: {prompt}")
    
    try:
        rag_result = query_rag_v3(prompt)
    except Exception as e:
        print(f"Debug: Error in RAG query: {e}")
        import traceback
        traceback.print_exc()
        raise e
    
    try:
        # Extract content and attribution data
        content = rag_result.get("content", "") if isinstance(rag_result, dict) else str(rag_result)
        attributions = rag_result.get("attributions", []) if isinstance(rag_result, dict) else []
        retrieval_contents = rag_result.get("retrieval_contents", []) if isinstance(rag_result, dict) else []
        message_id = rag_result.get("message_id") if isinstance(rag_result, dict) else None
        agent_id = rag_result.get("agent_id") if isinstance(rag_result, dict) else None
        
        # Return in a format compatible with downstream nodes
        return {
            "sources_gathered": [],  # We'll handle RAG sources differently
            "search_query": [state["search_query"]],
            "research_result": [content],
            # Add RAG-specific attribution data
            "rag_attributions": attributions,
            "rag_retrieval_contents": retrieval_contents,
            "rag_message_id": [message_id] if message_id else [],
            "rag_agent_id": [agent_id] if agent_id else [],
        }
    except Exception as e:
        print(f"Debug: Error extracting data from RAG result: {e}")
        import traceback
        traceback.print_exc()
        raise e


# Fan-out conditional edge function to create individual research tasks for each query
def fan_out_queries(state: QueryGenerationState, config: RunnableConfig):
    """Takes the list of queries and emits a list of Send objects, each with a single search_query and id."""
    # Get research_mode from state, defaulting to 'rag'
    research_mode = state.get("research_mode", "rag")
    
    if research_mode == "web":
        target_node = "web_research"
    elif research_mode == "rag":
        target_node = "rag_research"
    else:
        target_node = "rag_research"  # Default fallback
    
    return [
        Send(target_node, {"search_query": [query], "id": [idx]})
        for idx, query in enumerate(state["query_list"])
    ]


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Register nodes (do NOT register fan_out_queries as a node)
builder.add_node("generate_query", generate_query)
builder.add_node("tool_selection", tool_selection)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)
builder.add_node("rag_research", rag_research)

# Set the entrypoint as `generate_query`
builder.add_edge(START, "generate_query")
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
    "reflection", evaluate_research, ["web_research", "rag_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")
