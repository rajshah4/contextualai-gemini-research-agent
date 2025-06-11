import os
import operator
from typing import Annotated, TypedDict

from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
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

load_dotenv()

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
    """LangGraph node that performs web research using the native Google Search API tool.

    Executes a web search using the native Google Search API tool in combination with Gemini 2.0 Flash.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and research_result
    """
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

        # Instead of generating a new response, use the research results directly
        # This preserves the citations that were already inserted
        combined_research = "\n\n".join(state["research_result"])
        
        print(f"Debug: Using combined research results directly (preserves citations)")
        print(f"Debug: Combined research preview: {combined_research[:200]}...")
        
        # Replace the short urls with the original urls and add all used urls to the sources_gathered
        unique_sources = []
        # Remove debug statements for URL replacement
        # print(f"Debug: Available sources_gathered: {len(state['sources_gathered'])} sources")
        # print(f"Debug: Checking source - short_url: {source['short_url']}, in content: {source['short_url'] in modified_content}")
        # print(f"Debug: Replacing {source['short_url']} with {source['value']}")
        # print(f"Debug: After URL replacement - content preview: {modified_content[:200]}...")
        # print(f"Debug: Unique sources used: {len(unique_sources)}")

        modified_content = combined_research
        for source in state["sources_gathered"]:
            if source["short_url"] in modified_content:
                modified_content = modified_content.replace(
                    source["short_url"], source["value"]
                )
                unique_sources.append(source)
        
        # Create the final AI message with the preserved citations
        final_message = AIMessage(content=modified_content)
        
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
        
        # Heuristic: classify source type
        def classify_source(url):
            rag_domains = [
                "macrotrends.net", "companiesmarketcap.com", "nvidia.com", "tesla.com",
                "stockanalysis.com", "investing.com", "wallstreetzen.com", "visiblealpha.com",
                "stocktitan.net", "visualcapitalist.com", "analyzify.com", "globenewswire.com"
            ]
            if any(domain in url for domain in rag_domains):
                return "rag"
            return "web"

        rag_count = 0
        web_count = 0
        for source in unique_sources:
            url = source.get("value", "") or source.get("short_url", "")
            if classify_source(url) == "rag":
                rag_count += 1
            else:
                web_count += 1

        if rag_count > web_count:
            answer_type = "rag"
        elif web_count > rag_count:
            answer_type = "web"
        else:
            answer_type = "mixed"

        final_message.additional_kwargs["answer_type"] = answer_type

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
