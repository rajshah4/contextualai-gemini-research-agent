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
    """Search tool for NVIDIA that returns full response with attributions"""
    
    try:
        from contextual import ContextualAI
        client = ContextualAI(api_key=API_KEY)
        
        query_result = client.agents.query.create(
            agent_id=AGENT,
            messages=[{
                "content": prompt,
                "role": "user"
            }]
        )
        
        # Extract the content first
        try:
            content = query_result.message.content
            message_id = query_result.message_id
        except Exception as e:
            print(f"Debug: Error extracting basic data: {e}")
            return {
                "content": str(query_result),
                "message_id": None,
                "attributions": [],
                "retrieval_contents": [],
                "agent_id": AGENT
            }
        
        # Extract retrieval_contents and attributions
        retrieval_contents = []
        attributions = []
        
        if hasattr(query_result, 'retrieval_contents') and query_result.retrieval_contents:
            for rc in query_result.retrieval_contents:
                retrieval_contents.append({
                    "content_id": rc.content_id,
                    "number": rc.number,
                    "doc_name": rc.doc_name,
                    "page": rc.page,
                    "score": rc.score,
                    "title": rc.doc_name,
                    "message_id": message_id
                })
            
        if hasattr(query_result, 'attributions') and query_result.attributions:
            for attr in query_result.attributions:
                if attr.content_ids:
                    attributions.extend(attr.content_ids)
            attributions = list(set(attributions))
        
        print(f"Debug: RAG query completed - {len(attributions)} attributions from {len(retrieval_contents)} sources")
        
        return {
            "content": content,
            "message_id": message_id,
            "attributions": attributions,
            "retrieval_contents": retrieval_contents,
            "agent_id": AGENT
        }
        
    except Exception as e:
        print(f"Debug: RAG query error: {e}")
        return {
            "content": f"Error: {e}",
            "message_id": None,
            "attributions": [],
            "retrieval_contents": [],
            "agent_id": AGENT
        }


def get_research_topic(messages: List[AnyMessage]) -> str:
    """
    Get the research topic from the messages.
    """
    # check if request has a history and combine the messages into a single string
    if len(messages) == 1:
        research_topic = messages[-1].content
    else:
        research_topic = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                research_topic += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                research_topic += f"Assistant: {message.content}\n"
    return research_topic


def resolve_urls(urls_to_resolve: List[Any], id: int) -> Dict[str, str]:
    """
    Create a map of the vertex ai search urls (very long) to a short url with a unique id for each url.
    Ensures each original URL gets a consistent shortened form while maintaining uniqueness.
    """
    prefix = f"https://vertexaisearch.cloud.google.com/id/"
    urls = [site.web.uri for site in urls_to_resolve]

    # Create a dictionary that maps each unique URL to its first occurrence index
    resolved_map = {}
    for idx, url in enumerate(urls):
        if url not in resolved_map:
            resolved_map[url] = f"{prefix}{id}-{idx}"

    return resolved_map


def insert_citation_markers(text, citations_list):
    """
    Inserts citation markers into a text string based on start and end indices.

    Args:
        text (str): The original text string.
        citations_list (list): A list of dictionaries, where each dictionary
                               contains 'start_index', 'end_index', and
                               'segment_string' (the marker to insert).
                               Indices are assumed to be for the original text.

    Returns:
        str: The text with citation markers inserted.
    """
    print(f"Debug: insert_citation_markers - Processing {len(citations_list)} citations")
    print(f"Debug: Original text preview: {text[:200]}...")
    
    # Sort citations by end_index in descending order.
    # If end_index is the same, secondary sort by start_index descending.
    # This ensures that insertions at the end of the string don't affect
    # the indices of earlier parts of the string that still need to be processed.
    sorted_citations = sorted(
        citations_list, key=lambda c: (c["end_index"], c["start_index"]), reverse=True
    )

    modified_text = text
    for i, citation_info in enumerate(sorted_citations):
        # These indices refer to positions in the *original* text,
        # but since we iterate from the end, they remain valid for insertion
        # relative to the parts of the string already processed.
        end_idx = citation_info["end_index"]
        marker_to_insert = ""
        segments = citation_info.get("segments", [])
        print(f"Debug: Citation {i}: {len(segments)} segments, end_idx={end_idx}")
        
        for j, segment in enumerate(segments):
            label = segment.get('label', 'N/A')
            short_url = segment.get('short_url', 'N/A')
            marker = f" [{label}]({short_url})"
            marker_to_insert += marker
            print(f"Debug: Segment {j}: label='{label}', short_url='{short_url}', marker='{marker}'")
        
        print(f"Debug: Citation {i}: inserting '{marker_to_insert}' at position {end_idx}")
        print(f"Debug: Text around position {end_idx}: '{modified_text[max(0, end_idx-20):end_idx+20]}'")
        
        # Insert the citation marker at the original end_idx position
        modified_text = (
            modified_text[:end_idx] + marker_to_insert + modified_text[end_idx:]
        )

    print(f"Debug: Final modified text preview: {modified_text[:300]}...")
    return modified_text


def get_citations(response, resolved_urls_map):
    """
    Extracts and formats citation information from a Gemini model's response.

    This function processes the grounding metadata provided in the response to
    construct a list of citation objects. Each citation object includes the
    start and end indices of the text segment it refers to, and a string
    containing formatted markdown links to the supporting web chunks.

    Args:
        response: The response object from the Gemini model, expected to have
                  a structure including `candidates[0].grounding_metadata`.
                  It also relies on a `resolved_map` being available in its
                  scope to map chunk URIs to resolved URLs.

    Returns:
        list: A list of dictionaries, where each dictionary represents a citation
              and has the following keys:
              - "start_index" (int): The starting character index of the cited
                                     segment in the original text. Defaults to 0
                                     if not specified.
              - "end_index" (int): The character index immediately after the
                                   end of the cited segment (exclusive).
              - "segments" (list[str]): A list of individual markdown-formatted
                                        links for each grounding chunk.
              - "segment_string" (str): A concatenated string of all markdown-
                                        formatted links for the citation.
              Returns an empty list if no valid candidates or grounding supports
              are found, or if essential data is missing.
    """
    citations = []

    # Ensure response and necessary nested structures are present
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

        # Ensure segment information is present
        if not hasattr(support, "segment") or support.segment is None:
            continue  # Skip this support if segment info is missing

        start_index = (
            support.segment.start_index
            if support.segment.start_index is not None
            else 0
        )

        # Ensure end_index is present to form a valid segment
        if support.segment.end_index is None:
            continue  # Skip if end_index is missing, as it's crucial

        # Add 1 to end_index to make it an exclusive end for slicing/range purposes
        # (assuming the API provides an inclusive end_index)
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
                    
                    # Improve label generation - use title or fallback to a number
                    title = getattr(chunk.web, 'title', '') if hasattr(chunk, 'web') else ''
                    if title:
                        # Clean up the title - remove file extensions and clean up
                        label = title.replace('.html', '').replace('.pdf', '').replace('.txt', '')
                        if len(label) > 30:  # Truncate very long titles
                            label = label[:30] + "..."
                    else:
                        label = f"{ind + 1}"  # Fallback to number
                    
                    print(f"Debug: get_citations - chunk {ind}: title='{title}', label='{label}', uri='{chunk.web.uri}', short_url='{resolved_url}'")
                    
                    citation["segments"].append(
                        {
                            "label": label,
                            "short_url": resolved_url,
                            "value": chunk.web.uri,
                        }
                    )
                except (IndexError, AttributeError, NameError):
                    # Handle cases where chunk, web, uri, or resolved_map might be problematic
                    # For simplicity, we'll just skip adding this particular segment link
                    # In a production system, you might want to log this.
                    pass
        citations.append(citation)
    return citations


def tool_selection(state, config):
    configurable = Configuration.from_runnable_config(config)
    messages = state.get("messages", [])
    query = messages[-1].content if messages else ""
    
    # Add debug logging to see what query is being processed
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

User query: {query}
"""
    # Use your reasoning model (e.g., Gemini, OpenAI, etc.)
    llm = ChatGoogleGenerativeAI(
        model=configurable.reasoning_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    result = llm.invoke(prompt)
    research_mode = result.content.strip().lower()
    
    # Add debug logging to see the LLM's decision
    print(f"Debug: Tool Selection - LLM decision: '{result.content}' -> research_mode: '{research_mode}'")
    
    # Normalize the response - accept both "web" and "rag" regardless of case
    if research_mode in ["web", "rag"]:
        pass  # Valid mode
    else:
        print(f"Debug: Tool Selection - Invalid mode '{research_mode}', defaulting to 'web'")
        research_mode = "web"  # fallback
    
    state["research_mode"] = research_mode
    print(f"Debug: Tool Selection - Final research_mode set to: '{research_mode}'")
    return state
