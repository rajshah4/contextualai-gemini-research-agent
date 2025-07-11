from typing import Any, Dict, List
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.configuration import Configuration
import requests

# RAG query utility
API_KEY = os.getenv("CONTEXTUAL_AI_API_KEY")
AGENT = os.getenv("CONTEXTUAL_AI_AGENT_ID")

if not API_KEY:
    raise ValueError("CONTEXTUAL_AI_API_KEY environment variable is not set")
if not AGENT:
    raise ValueError("CONTEXTUAL_AI_AGENT_ID environment variable is not set")

def query_rag_v3(prompt: str) -> dict:
    """Search tool that queries the RAG system and returns the raw response data."""
    try:
        from contextual import ContextualAI
        client = ContextualAI(api_key=API_KEY)
        
        query_result = client.agents.query.create(
            agent_id=AGENT,
            messages=[{"content": prompt, "role": "user"}]
        )
        
        content = query_result.message.content
        message_id = query_result.message_id
        
        retrieval_contents = []
        if hasattr(query_result, 'retrieval_contents') and query_result.retrieval_contents:
            for rc in query_result.retrieval_contents:
                retrieval_contents.append({
                    "content_id": rc.content_id, "number": rc.number,
                    "doc_name": rc.doc_name, "page": rc.page,
                    "score": rc.score, "title": rc.doc_name,
                    "message_id": message_id
                })
        
        raw_attributions = []
        if hasattr(query_result, 'attributions') and query_result.attributions:
            for attr in query_result.attributions:
                raw_attributions.append({
                    "content_ids": attr.content_ids,
                    "start_idx": attr.start_idx,
                    "end_idx": attr.end_idx
                })

        print(f"Debug: RAG query completed - {len(raw_attributions)} attributions captured from {len(retrieval_contents)} sources.")
        
        return {
            "content": content,
            "message_id": message_id,
            "retrieval_contents": retrieval_contents,
            "attributions": raw_attributions,
            "agent_id": AGENT
        }
        
    except Exception as e:
        print(f"Debug: RAG query error: {e}")
        return {
            "content": f"Error: {e}", "message_id": None,
            "retrieval_contents": [], "attributions": [], "agent_id": AGENT
        }


def get_research_topic(messages: List[AnyMessage]) -> str:
    """Extracts the content of the most recent human message."""
    print(f"Debug: get_research_topic called with {len(messages)} messages.")
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            print(f"Debug: Found last human message: {msg.content}")
            return msg.content
    if messages:
        last_msg = messages[-1]
        if hasattr(last_msg, 'content'):
            return last_msg.content
        elif isinstance(last_msg, dict) and "content" in last_msg:
            return last_msg["content"]
    return "No research topic found"


def resolve_urls(urls_to_resolve: List[Any], id: int) -> Dict[str, str]:
    """
    Create a map of the vertex ai search urls (very long) to a short url with a unique id for each url.
    Ensures each original URL gets a consistent shortened form while maintaining uniqueness.
    """
    prefix = f"https://vertexaisearch.cloud.google.com/id/"
    urls = [site.web.uri for site in urls_to_resolve]

    resolved_map = {}
    for idx, url in enumerate(urls):
        if url not in resolved_map:
            resolved_map[url] = f"{prefix}{id}-{idx}"

    return resolved_map


def insert_citation_markers(text, citations_list):
    """
    Inserts citation markers into a text string based on start and end indices.
    """
    sorted_citations = sorted(
        citations_list, key=lambda c: (c["end_index"], c["start_index"]), reverse=True
    )
    modified_text = text
    for i, citation_info in enumerate(sorted_citations):
        end_idx = citation_info["end_index"]
        marker_to_insert = ""
        segments = citation_info.get("segments", [])
        for j, segment in enumerate(segments):
            label = segment.get('label', 'N/A')
            short_url = segment.get('short_url', 'N/A')
            marker = f" [{label}]({short_url})"
            marker_to_insert += marker
        modified_text = (
            modified_text[:end_idx] + marker_to_insert + modified_text[end_idx:]
        )
    return modified_text


def get_citations(response, resolved_urls_map):
    """
    Extracts and formats citation information from a Gemini model's response.
    """
    citations = []
    if not response or not response.candidates:
        return citations

    candidate = response.candidates[0]
    if (
        not hasattr(candidate, "grounding_metadata")
        or not candidate.grounding_metadata
        or not hasattr(candidate.grounding_metadata, "grounding_supports")
    ):
        return citations

    for support in candidate.grounding_metadata.grounding_supports:
        citation = {}
        if not hasattr(support, "segment") or support.segment is None:
            continue
        start_index = (
            support.segment.start_index
            if support.segment.start_index is not None
            else 0
        )
        if support.segment.end_index is None:
            continue
        citation["start_index"] = start_index
        citation["end_index"] = support.segment.end_index
        citation["segments"] = []
        if (
            hasattr(support, "grounding_chunk_indices")
            and support.grounding_chunk_indices
        ):
            for ind in support.grounding_chunk_indices:
                try:
                    chunk = candidate.grounding_metadata.grounding_chunks[ind]
                    resolved_url = resolved_urls_map.get(chunk.web.uri, None)
                    title = getattr(chunk.web, 'title', '') if hasattr(chunk, 'web') else ''
                    if title:
                        label = title.replace('.html', '').replace('.pdf', '').replace('.txt', '')
                        if len(label) > 30:
                            label = label[:30] + "..."
                    else:
                        label = f"{ind + 1}"
                    citation["segments"].append(
                        {
                            "label": label,
                            "short_url": resolved_url,
                            "value": chunk.web.uri,
                        }
                    )
                except (IndexError, AttributeError, NameError):
                    pass
        citations.append(citation)
    return citations


def tool_selection(state, config):
    configurable = Configuration.from_runnable_config(config)
    messages = state.get("messages", [])
    
    last_human_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human_message = msg.content
            break
    
    query = last_human_message
    
    print(f"Debug: Tool Selection - Processing query: '{query}'")
    
    prompt = f"""
You are an expert research assistant. Given a user query, decide whether it should be answered using a static knowledge base on fortune 500 companies (RAG) or requires up-to-date information from the web.

Use RAG for:
- Financial information, earnings, revenue data for major companies
- Technical documentation or product specifications from large companies
- Company-specific information (e.g., NVIDIA, Tesla, Microsoft, etc.)

Use WEB for:
- Weather information and forecasts
- Current news, breaking news, recent events
- Stock prices, market data, real-time information
- Sports scores, schedules, current events
- Any query asking for "today", "current", "latest", "recent", or "now"

Respond with only "RAG" or "WEB".

User Query: {query}
"""
    llm = ChatGoogleGenerativeAI(
        model=configurable.reasoning_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    result = llm.invoke(prompt)
    research_mode = result.content.strip().lower()
    
    print(f"Debug: Tool Selection - LLM decision: '{result.content}' -> research_mode: '{research_mode}'")
    
    if research_mode not in ["web", "rag"]:
        print(f"Debug: Tool Selection - Invalid mode '{research_mode}', defaulting to 'web'")
        research_mode = "web"
    
    state["research_mode"] = research_mode
    print(f"Debug: Tool Selection - Final research_mode set to: '{research_mode}'")
    return state