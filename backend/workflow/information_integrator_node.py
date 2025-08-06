"""
Information integrator node for TTD-DR framework workflow.
Integrates retrieved information into the current draft using Kimi K2 intelligence.
"""

import asyncio
import logging
from typing import List, Dict, Any

from models.core import TTDRState, Draft, RetrievedInfo, InformationGap
from services.kimi_k2_information_integrator import KimiK2InformationIntegrator
from services.kimi_k2_coherence_manager import KimiK2CoherenceManager

logger = logging.getLogger(__name__)

def information_integrator_node(state: TTDRState) -> TTDRState:
    """
    Integrate retrieved information into current draft using Kimi K2
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with integrated information
    """
    logger.info("Executing information_integrator_node with Kimi K2 integration")
    
    try:
        # Validate required state components
        if not state.get("current_draft"):
            logger.warning("No current draft available for information integration")
            return {
                **state,
                "error_log": state.get("error_log", []) + ["No draft available for integration"]
            }
        
        retrieved_info = state.get("retrieved_info", [])
        information_gaps = state.get("information_gaps", [])
        
        if not retrieved_info:
            logger.info("No retrieved information to integrate")
            return state
        
        if not information_gaps:
            logger.warning("No information gaps defined for integration")
            # Still proceed with integration, but may be less targeted
        
        # Initialize Kimi K2 information integrator
        integrator = KimiK2InformationIntegrator()
        
        # Run async integration
        try:
            updated_draft = asyncio.run(integrator.integrate_information(
                draft=state["current_draft"],
                retrieved_info=retrieved_info,
                gaps=information_gaps
            ))
        except RuntimeError:
            # Handle case where event loop is already running
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                updated_draft = loop.run_until_complete(integrator.integrate_information(
                    draft=state["current_draft"],
                    retrieved_info=retrieved_info,
                    gaps=information_gaps
                ))
            finally:
                loop.close()
        
        # Apply coherence maintenance and citation management
        coherence_manager = KimiK2CoherenceManager()
        
        try:
            # Maintain coherence
            coherent_draft, coherence_report = asyncio.run(
                coherence_manager.maintain_coherence(updated_draft)
            )
            
            # Manage citations
            final_draft, citations = asyncio.run(
                coherence_manager.manage_citations(coherent_draft, retrieved_info)
            )
            
            updated_draft = final_draft
            
        except RuntimeError:
            # Handle case where event loop is already running
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                coherent_draft, coherence_report = loop.run_until_complete(
                    coherence_manager.maintain_coherence(updated_draft)
                )
                final_draft, citations = loop.run_until_complete(
                    coherence_manager.manage_citations(coherent_draft, retrieved_info)
                )
                updated_draft = final_draft
            finally:
                loop.close()
        except Exception as coherence_error:
            logger.error(f"Coherence maintenance failed: {coherence_error}")
            coherence_report = None
            citations = []
        
        # Update iteration count
        iteration_count = state.get("iteration_count", 0) + 1
        
        logger.info(f"Information integration completed. Iteration: {iteration_count}")
        logger.info(f"Integration history entries: {len(integrator.integration_history)}")
        if coherence_report:
            logger.info(f"Coherence score: {coherence_report.overall_score:.2f}")
        logger.info(f"Citations managed: {len(citations)}")
        
        # Return updated state
        return {
            **state,
            "current_draft": updated_draft,
            "iteration_count": iteration_count,
            "integration_history": integrator.integration_history,
            "coherence_report": coherence_report,
            "citations": citations,
            "coherence_statistics": coherence_manager.get_coherence_statistics()
        }
        
    except Exception as e:
        logger.error(f"Information integration failed: {e}")
        
        # Fallback to simple integration
        try:
            fallback_draft = _fallback_integration(
                state["current_draft"],
                state.get("retrieved_info", []),
                state.get("information_gaps", [])
            )
            
            return {
                **state,
                "current_draft": fallback_draft,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "error_log": state.get("error_log", []) + [f"Integration error (fallback used): {str(e)}"]
            }
            
        except Exception as fallback_error:
            logger.error(f"Fallback integration also failed: {fallback_error}")
            return {
                **state,
                "error_log": state.get("error_log", []) + [
                    f"Integration failed: {str(e)}",
                    f"Fallback failed: {str(fallback_error)}"
                ]
            }

def _fallback_integration(draft: Draft, retrieved_info: List[RetrievedInfo], 
                         gaps: List[InformationGap]) -> Draft:
    """
    Fallback integration method when Kimi K2 integration fails
    
    Args:
        draft: Current draft
        retrieved_info: Retrieved information to integrate
        gaps: Information gaps being addressed
        
    Returns:
        Updated draft with basic integration
    """
    logger.info("Using fallback integration method")
    
    # Create a copy of the draft content
    updated_content = draft.content.copy()
    
    # Group retrieved info by gap
    info_by_gap = {}
    for info in retrieved_info:
        if info.gap_id:
            if info.gap_id not in info_by_gap:
                info_by_gap[info.gap_id] = []
            info_by_gap[info.gap_id].append(info)
    
    # Integrate information for each gap
    for gap in gaps:
        if gap.id in info_by_gap:
            gap_info = info_by_gap[gap.id]
            section_id = gap.section_id
            
            if section_id in updated_content:
                current_content = updated_content[section_id]
                
                # Append new information with basic formatting
                for info in gap_info:
                    if current_content and not current_content.endswith('\n\n'):
                        current_content += '\n\n'
                    
                    current_content += f"{info.content}\n\nSource: {info.source.title} ({info.source.url})\n"
                
                updated_content[section_id] = current_content
            else:
                # Create new content for empty section
                section_content = ""
                for info in gap_info:
                    section_content += f"{info.content}\n\nSource: {info.source.title} ({info.source.url})\n\n"
                
                updated_content[section_id] = section_content.strip()
    
    # Create updated draft
    from datetime import datetime
    from models.core import DraftMetadata
    
    updated_draft = Draft(
        id=draft.id,
        topic=draft.topic,
        structure=draft.structure,
        content=updated_content,
        metadata=DraftMetadata(
            created_at=draft.metadata.created_at,
            updated_at=datetime.now(),
            author=draft.metadata.author,
            version=draft.metadata.version,
            word_count=sum(len(content.split()) for content in updated_content.values())
        ),
        quality_score=draft.quality_score,
        iteration=draft.iteration + 1
    )
    
    return updated_draft

# Additional utility functions for integration support

def validate_integration_state(state: TTDRState) -> List[str]:
    """
    Validate that the state has all required components for integration
    
    Args:
        state: Current workflow state
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if not state.get("current_draft"):
        errors.append("No current draft available")
    
    if not state.get("retrieved_info"):
        errors.append("No retrieved information available")
    
    if not state.get("information_gaps"):
        errors.append("No information gaps defined")
    
    # Check that retrieved info has gap associations
    retrieved_info = state.get("retrieved_info", [])
    unassociated_info = [info for info in retrieved_info if not info.gap_id]
    if unassociated_info:
        errors.append(f"{len(unassociated_info)} retrieved items not associated with gaps")
    
    return errors

def get_integration_statistics(state: TTDRState) -> Dict[str, Any]:
    """
    Get statistics about the integration process
    
    Args:
        state: Current workflow state
        
    Returns:
        Dictionary with integration statistics
    """
    stats = {
        "total_retrieved_items": len(state.get("retrieved_info", [])),
        "total_gaps": len(state.get("information_gaps", [])),
        "iteration_count": state.get("iteration_count", 0),
        "draft_sections": 0,
        "populated_sections": 0,
        "total_word_count": 0
    }
    
    if state.get("current_draft"):
        draft = state["current_draft"]
        stats["draft_sections"] = len(draft.structure.sections)
        stats["populated_sections"] = len([s for s in draft.content.values() if s.strip()])
        stats["total_word_count"] = sum(len(content.split()) for content in draft.content.values())
    
    # Integration history stats if available
    integration_history = state.get("integration_history", [])
    if integration_history:
        stats["integration_operations"] = len(integration_history)
        stats["unique_sources_integrated"] = len(set(
            entry.get("source_url", "") for entry in integration_history
        ))
    
    return stats