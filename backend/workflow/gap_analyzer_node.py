"""
Gap Analyzer Node for TTD-DR LangGraph workflow.
Identifies information gaps in the current draft using Kimi K2 intelligence.
"""

import logging
import asyncio
import uuid
from typing import List, Dict, Any

from models.core import TTDRState, InformationGap, GapType, Priority, SearchQuery

logger = logging.getLogger(__name__)

def gap_analyzer_node(state: TTDRState) -> TTDRState:
    """
    Analyze current draft for information gaps using Kimi K2
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with identified information gaps
    """
    logger.info("Executing gap_analyzer_node")
    
    try:
        # Import services here to avoid circular imports
        from services.kimi_k2_gap_analyzer import KimiK2InformationGapAnalyzer
        from services.kimi_k2_search_query_generator import KimiK2SearchQueryGenerator
        
        if not state.get("current_draft"):
            logger.warning("No current draft available for gap analysis")
            return {
                **state,
                "information_gaps": [],
                "error_log": state.get("error_log", []) + ["No draft available for gap analysis"]
            }
        
        # Initialize gap analyzer and query generator
        gap_analyzer = KimiK2InformationGapAnalyzer()
        query_generator = KimiK2SearchQueryGenerator()
        
        # Run async gap identification
        gaps = _run_async_operation(
            gap_analyzer.identify_gaps(state["current_draft"])
        )
        
        # Generate search queries for each gap
        for gap in gaps:
            try:
                search_queries = _run_async_operation(
                    query_generator.generate_search_queries(
                        gap=gap,
                        topic=state["topic"],
                        domain=state["current_draft"].structure.domain,
                        max_queries=3
                    )
                )
                gap.search_queries = search_queries
                
            except Exception as e:
                logger.error(f"Failed to generate queries for gap {gap.id}: {e}")
                # Add fallback query
                gap.search_queries = [
                    SearchQuery(
                        query=f"{state['topic']} {gap.description[:50]}",
                        priority=Priority.MEDIUM
                    )
                ]
        
        logger.info(f"Identified {len(gaps)} information gaps with search queries")
        
        return {
            **state,
            "information_gaps": gaps
        }
        
    except Exception as e:
        logger.error(f"Gap analysis failed: {e}")
        
        # Fallback to simple gap identification
        gaps = _create_fallback_gaps(state)
        
        return {
            **state,
            "information_gaps": gaps,
            "error_log": state.get("error_log", []) + [f"Gap analysis error: {str(e)}"]
        }

def _run_async_operation(coro):
    """Run async operation handling event loop issues"""
    try:
        # Check if coro is already a coroutine object
        if asyncio.iscoroutine(coro):
            return asyncio.run(coro)
        else:
            # If it's already a result, return it
            return coro
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                if asyncio.iscoroutine(coro):
                    return loop.run_until_complete(coro)
                else:
                    return coro
            finally:
                loop.close()
        else:
            raise

def _create_fallback_gaps(state: TTDRState) -> List[InformationGap]:
    """Create fallback gaps when Kimi K2 analysis fails"""
    gaps = []
    
    if state.get("current_draft"):
        for section in state["current_draft"].structure.sections:
            gap = InformationGap(
                id=str(uuid.uuid4()),
                section_id=section.id,
                gap_type=GapType.CONTENT,
                description=f"Need more detailed information for {section.title}",
                priority=Priority.MEDIUM,
                search_queries=[
                    SearchQuery(
                        query=f"{state['topic']} {section.title.lower()}",
                        priority=Priority.MEDIUM
                    )
                ]
            )
            gaps.append(gap)
    
    return gaps

# Export the node function
__all__ = ["gap_analyzer_node"]