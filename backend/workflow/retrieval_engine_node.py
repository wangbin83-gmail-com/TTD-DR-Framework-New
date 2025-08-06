"""
Retrieval Engine Node for TTD-DR LangGraph workflow.
Integrates Google Search API through the Dynamic Retrieval Engine.
"""

import logging
from typing import Dict, Any
import asyncio

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.core import TTDRState, RetrievedInfo, Source
from services.dynamic_retrieval_engine import DynamicRetrievalEngine, GoogleSearchError
from services.google_search_client import GoogleSearchClient

logger = logging.getLogger(__name__)

def retrieval_engine_node(state: TTDRState) -> TTDRState:
    """
    Retrieve information for identified gaps using Google Search API
    
    Args:
        state: Current workflow state containing information gaps
        
    Returns:
        Updated state with retrieved information
    """
    logger.info("Executing retrieval_engine_node")
    
    # Check if we have information gaps to process
    if not state.get("information_gaps"):
        logger.warning("No information gaps found for retrieval")
        return {
            **state,
            "retrieved_info": [],
            "error_log": state.get("error_log", []) + ["No information gaps available for retrieval"]
        }
    
    try:
        # Initialize retrieval engine
        retrieval_engine = DynamicRetrievalEngine()
        
        # Determine max results per gap based on requirements
        requirements = state.get("requirements")
        if requirements:
            max_sources = getattr(requirements, 'max_sources', 20)
            max_results_per_gap = max(1, max_sources // max(len(state["information_gaps"]), 1))
        else:
            max_results_per_gap = 5
        
        # Run async retrieval operation
        retrieved_info = _run_async_retrieval(
            retrieval_engine, 
            state["information_gaps"], 
            max_results_per_gap
        )
        
        logger.info(f"Successfully retrieved {len(retrieved_info)} information items")
        
        # Filter results based on quality thresholds
        filtered_info = _filter_retrieved_info(retrieved_info, state)
        
        logger.info(f"Filtered to {len(filtered_info)} high-quality information items")
        
        return {
            **state,
            "retrieved_info": filtered_info
        }
        
    except GoogleSearchError as e:
        logger.error(f"Google Search API error: {e}")
        
        # Handle specific error types
        if e.error_type == "quota_exceeded":
            error_msg = "Google Search API daily quota exceeded"
        elif e.error_type == "invalid_key":
            error_msg = "Invalid Google Search API key"
        elif e.error_type == "configuration":
            error_msg = "Google Search API not configured"
        else:
            error_msg = f"Google Search API error: {e.message}"
        
        # Fallback to mock data for development/testing
        fallback_info = _create_fallback_retrieved_info(state["information_gaps"])
        
        return {
            **state,
            "retrieved_info": fallback_info,
            "error_log": state.get("error_log", []) + [error_msg]
        }
        
    except Exception as e:
        logger.error(f"Retrieval engine failed: {e}")
        
        # Fallback to mock data
        fallback_info = _create_fallback_retrieved_info(state["information_gaps"])
        
        return {
            **state,
            "retrieved_info": fallback_info,
            "error_log": state.get("error_log", []) + [f"Retrieval engine error: {str(e)}"]
        }

def _run_async_retrieval(retrieval_engine: DynamicRetrievalEngine, 
                        information_gaps: list, 
                        max_results_per_gap: int) -> list:
    """
    Run async retrieval operation, handling event loop issues
    
    Args:
        retrieval_engine: The retrieval engine instance
        information_gaps: List of information gaps to process
        max_results_per_gap: Maximum results per gap
        
    Returns:
        List of retrieved information items
    """
    try:
        # Try to run in existing event loop
        return asyncio.run(retrieval_engine.retrieve_information(
            information_gaps, max_results_per_gap
        ))
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            # Create new event loop for this operation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(retrieval_engine.retrieve_information(
                    information_gaps, max_results_per_gap
                ))
            finally:
                loop.close()
        else:
            raise

def _filter_retrieved_info(retrieved_info: list, state: TTDRState) -> list:
    """
    Filter retrieved information based on quality thresholds
    
    Args:
        retrieved_info: List of retrieved information items
        state: Current workflow state
        
    Returns:
        Filtered list of high-quality information items
    """
    if not retrieved_info:
        return []
    
    # Get quality thresholds from requirements
    requirements = state.get("requirements")
    min_relevance = 0.5  # Default threshold
    min_credibility = 0.4  # Default threshold
    
    if requirements:
        # Adjust thresholds based on quality requirements
        quality_threshold = getattr(requirements, 'quality_threshold', 0.8)
        min_relevance = max(0.3, quality_threshold - 0.3)
        min_credibility = max(0.3, quality_threshold - 0.4)
    
    # Filter based on quality scores
    filtered_info = []
    for info in retrieved_info:
        if (info.relevance_score >= min_relevance and 
            info.credibility_score >= min_credibility):
            filtered_info.append(info)
    
    # If we filtered out too much, relax thresholds
    if len(filtered_info) < len(retrieved_info) * 0.3:
        logger.warning("Quality filtering removed too many results, relaxing thresholds")
        filtered_info = [
            info for info in retrieved_info
            if (info.relevance_score >= 0.3 and info.credibility_score >= 0.3)
        ]
    
    # Sort by combined quality score
    filtered_info.sort(
        key=lambda x: (x.relevance_score + x.credibility_score) / 2,
        reverse=True
    )
    
    return filtered_info

def _create_fallback_retrieved_info(information_gaps: list) -> list:
    """
    Create fallback retrieved information when API calls fail
    
    Args:
        information_gaps: List of information gaps
        
    Returns:
        List of mock retrieved information items
    """
    import uuid
    from datetime import datetime
    
    fallback_info = []
    
    for gap in information_gaps:
        # Create mock source
        source = Source(
            url=f"https://example.com/research/{gap.section_id}",
            title=f"Research Information for {gap.description[:50]}...",
            domain="example.com",
            credibility_score=0.6
        )
        
        # Create mock content based on gap description
        mock_content = f"""
This is placeholder content for the information gap: {gap.description}

Key points to address:
- {gap.gap_type.value} information needed
- Priority level: {gap.priority.value}
- Section: {gap.section_id}

This content would normally be retrieved from external sources via Google Search API.
For development and testing purposes, this mock content is provided.

Additional context and detailed information would be included here in a real retrieval scenario.
"""
        
        # Create retrieved info item
        info = RetrievedInfo(
            source=source,
            content=mock_content.strip(),
            relevance_score=0.6,
            credibility_score=0.6,
            gap_id=gap.id
        )
        
        fallback_info.append(info)
    
    return fallback_info

async def test_retrieval_engine_health() -> Dict[str, Any]:
    """
    Test the health of the retrieval engine components
    
    Returns:
        Dictionary with health check results
    """
    try:
        retrieval_engine = DynamicRetrievalEngine()
        health_status = await retrieval_engine.health_check()
        
        return {
            "status": "healthy" if all(health_status.values()) else "degraded",
            "components": health_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Export the node function for use in the workflow
__all__ = ["retrieval_engine_node", "test_retrieval_engine_health"]