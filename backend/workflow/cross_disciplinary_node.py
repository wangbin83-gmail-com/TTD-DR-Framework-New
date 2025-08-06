"""
Cross-disciplinary research workflow node for LangGraph integration.
This module provides workflow nodes for cross-disciplinary research capabilities
within the TTD-DR framework.
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

from models.core import TTDRState, ResearchDomain, Draft, QualityMetrics
from services.cross_disciplinary_integrator import (
    CrossDisciplinaryIntegrator, 
    CrossDisciplinaryIntegration,
    CrossDisciplinaryConflict
)
from services.kimi_k2_client import KimiK2Client

logger = logging.getLogger(__name__)


def cross_disciplinary_detector_node(state: TTDRState) -> TTDRState:
    """
    LangGraph node for detecting cross-disciplinary research needs.
    
    This node:
    1. Analyzes the research topic and retrieved information
    2. Detects if cross-disciplinary approach is needed
    3. Identifies involved research domains
    4. Updates state with cross-disciplinary metadata
    
    Args:
        state: Current TTD-DR workflow state
        
    Returns:
        Updated state with cross-disciplinary detection results
    """
    try:
        logger.info("Starting cross-disciplinary detection")
        
        # Initialize cross-disciplinary integrator
        kimi_client = KimiK2Client()
        integrator = CrossDisciplinaryIntegrator(kimi_client)
        
        # Get topic and retrieved information
        topic = state.get("topic", "")
        retrieved_info = state.get("retrieved_info", [])
        
        # Detect cross-disciplinary nature
        is_cross_disciplinary, involved_domains = integrator.detect_cross_disciplinary_nature(
            topic=topic,
            retrieved_info=retrieved_info
        )
        
        # Create cross-disciplinary metadata
        cross_disciplinary_metadata = {
            "is_cross_disciplinary": is_cross_disciplinary,
            "involved_domains": [domain.value for domain in involved_domains],
            "detection_timestamp": datetime.now().isoformat(),
            "domains_count": len(involved_domains),
            "detection_confidence": 0.8 if is_cross_disciplinary else 0.2
        }
        
        # Update state
        updated_state = {
            **state,
            "cross_disciplinary_metadata": cross_disciplinary_metadata,
            "requires_cross_disciplinary": is_cross_disciplinary
        }
        
        if is_cross_disciplinary:
            logger.info(f"Cross-disciplinary research detected. Domains: {[d.value for d in involved_domains]}")
        else:
            logger.info("Single-domain research detected")
        
        return updated_state
        
    except Exception as e:
        error_msg = f"Cross-disciplinary detection failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Add error to state but don't fail the workflow
        error_log = state.get("error_log", [])
        error_log.append(f"Cross-Disciplinary Detector: {error_msg}")
        
        return {
            **state,
            "error_log": error_log,
            "requires_cross_disciplinary": False  # Default to single-domain
        }


def cross_disciplinary_integrator_node(state: TTDRState) -> TTDRState:
    """
    LangGraph node for multi-domain knowledge integration.
    
    This node:
    1. Integrates knowledge from multiple research domains
    2. Identifies and resolves cross-disciplinary conflicts
    3. Creates unified cross-disciplinary perspective
    4. Updates draft with integrated knowledge
    
    Args:
        state: Current TTD-DR workflow state
        
    Returns:
        Updated state with cross-disciplinary integration
    """
    try:
        logger.info("Starting cross-disciplinary knowledge integration")
        
        # Check if cross-disciplinary integration is needed
        if not state.get("requires_cross_disciplinary", False):
            logger.info("Cross-disciplinary integration not required, skipping")
            return state
        
        # Initialize integrator
        kimi_client = KimiK2Client()
        integrator = CrossDisciplinaryIntegrator(kimi_client)
        
        # Get required data from state
        topic = state.get("topic", "")
        retrieved_info = state.get("retrieved_info", [])
        current_draft = state.get("current_draft")
        
        # Get involved domains from metadata
        cross_disciplinary_metadata = state.get("cross_disciplinary_metadata", {})
        domain_names = cross_disciplinary_metadata.get("involved_domains", ["GENERAL"])
        domains = [ResearchDomain(name) for name in domain_names]
        
        # Perform multi-domain knowledge integration
        integration_result = integrator.integrate_multi_domain_knowledge(
            topic=topic,
            domains=domains,
            retrieved_info=retrieved_info,
            current_draft=current_draft
        )
        
        # Update draft with integrated knowledge
        updated_draft = current_draft
        if current_draft:
            # Apply cross-disciplinary formatting to draft content
            formatted_content = {}
            for section_id, content in current_draft.content.items():
                if content:
                    # Apply cross-disciplinary formatting
                    formatted_content[section_id] = integrator.format_cross_disciplinary_output(
                        draft=current_draft,
                        integration=integration_result,
                        output_format="comprehensive"
                    )
                else:
                    formatted_content[section_id] = content
            
            updated_draft = current_draft.copy(deep=True)
            updated_draft.content = formatted_content
            updated_draft.metadata.updated_at = datetime.now()
        
        # Create integration metadata for state
        integration_metadata = {
            "integration_completed": True,
            "coherence_score": integration_result.coherence_score,
            "integration_strategy": integration_result.integration_strategy,
            "synthesis_approach": integration_result.synthesis_approach,
            "conflicts_identified": len(integration_result.conflicts_identified),
            "conflicts_resolved": len(integration_result.conflicts_resolved),
            "disciplinary_perspectives": len(integration_result.disciplinary_perspectives),
            "integration_timestamp": datetime.now().isoformat()
        }
        
        # Update state with integration results
        updated_state = {
            **state,
            "current_draft": updated_draft,
            "cross_disciplinary_integration": integration_result,
            "integration_metadata": integration_metadata
        }
        
        logger.info(f"Cross-disciplinary integration completed. Coherence score: {integration_result.coherence_score:.2f}")
        
        return updated_state
        
    except Exception as e:
        error_msg = f"Cross-disciplinary integration failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Add error to state but don't fail the workflow
        error_log = state.get("error_log", [])
        error_log.append(f"Cross-Disciplinary Integrator: {error_msg}")
        
        return {
            **state,
            "error_log": error_log
        }


def cross_disciplinary_conflict_resolver_node(state: TTDRState) -> TTDRState:
    """
    LangGraph node for resolving cross-disciplinary conflicts.
    
    This node:
    1. Identifies conflicts between disciplinary perspectives
    2. Applies appropriate resolution strategies
    3. Updates integration with resolved conflicts
    4. Improves overall coherence
    
    Args:
        state: Current TTD-DR workflow state
        
    Returns:
        Updated state with resolved conflicts
    """
    try:
        logger.info("Starting cross-disciplinary conflict resolution")
        
        # Check if we have cross-disciplinary integration to work with
        integration_result = state.get("cross_disciplinary_integration")
        if not integration_result:
            logger.info("No cross-disciplinary integration found, skipping conflict resolution")
            return state
        
        # Initialize integrator
        kimi_client = KimiK2Client()
        integrator = CrossDisciplinaryIntegrator(kimi_client)
        
        # Get conflicts that need resolution
        unresolved_conflicts = [
            conflict for conflict in integration_result.conflicts_identified
            if not conflict.resolved
        ]
        
        if not unresolved_conflicts:
            logger.info("No unresolved conflicts found")
            return state
        
        logger.info(f"Resolving {len(unresolved_conflicts)} cross-disciplinary conflicts")
        
        # Resolve conflicts
        resolved_conflicts = integrator.resolve_cross_disciplinary_conflicts(
            conflicts=unresolved_conflicts,
            disciplinary_perspectives=integration_result.disciplinary_perspectives
        )
        
        # Update integration result with resolved conflicts
        updated_integration = integration_result.copy(deep=True)
        updated_integration.conflicts_resolved.extend(resolved_conflicts)
        
        # Recalculate coherence score
        updated_integration.coherence_score = integrator._calculate_integration_coherence(
            integration_result.disciplinary_perspectives,
            updated_integration.conflicts_resolved
        )
        
        # Update integration metadata
        integration_metadata = state.get("integration_metadata", {})
        integration_metadata.update({
            "conflicts_resolved": len(updated_integration.conflicts_resolved),
            "coherence_score": updated_integration.coherence_score,
            "conflict_resolution_timestamp": datetime.now().isoformat(),
            "resolution_success_rate": len(resolved_conflicts) / len(unresolved_conflicts) if unresolved_conflicts else 1.0
        })
        
        # Update state
        updated_state = {
            **state,
            "cross_disciplinary_integration": updated_integration,
            "integration_metadata": integration_metadata
        }
        
        logger.info(f"Resolved {len(resolved_conflicts)} conflicts. New coherence score: {updated_integration.coherence_score:.2f}")
        
        return updated_state
        
    except Exception as e:
        error_msg = f"Cross-disciplinary conflict resolution failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Add error to state but don't fail the workflow
        error_log = state.get("error_log", [])
        error_log.append(f"Cross-Disciplinary Conflict Resolver: {error_msg}")
        
        return {
            **state,
            "error_log": error_log
        }


def cross_disciplinary_formatter_node(state: TTDRState) -> TTDRState:
    """
    LangGraph node for specialized cross-disciplinary output formatting.
    
    This node:
    1. Applies specialized formatting for cross-disciplinary research
    2. Ensures proper presentation of multiple perspectives
    3. Formats conflicts and resolutions appropriately
    4. Creates final cross-disciplinary report structure
    
    Args:
        state: Current TTD-DR workflow state
        
    Returns:
        Updated state with formatted cross-disciplinary output
    """
    try:
        logger.info("Starting cross-disciplinary output formatting")
        
        # Check if we have cross-disciplinary content to format
        integration_result = state.get("cross_disciplinary_integration")
        current_draft = state.get("current_draft")
        
        if not integration_result or not current_draft:
            logger.info("No cross-disciplinary content to format")
            return state
        
        # Initialize integrator
        kimi_client = KimiK2Client()
        integrator = CrossDisciplinaryIntegrator(kimi_client)
        
        # Determine output format based on research characteristics
        output_format = "comprehensive"  # Default format
        
        # Adjust format based on number of domains and conflicts
        domains_count = len(integration_result.primary_domains)
        conflicts_count = len(integration_result.conflicts_identified)
        
        if domains_count > 3:
            output_format = "hierarchical"
        elif conflicts_count > 2:
            output_format = "comparative"
        elif integration_result.coherence_score > 0.8:
            output_format = "synthesis"
        
        # Format the cross-disciplinary output
        formatted_output = integrator.format_cross_disciplinary_output(
            draft=current_draft,
            integration=integration_result,
            output_format=output_format
        )
        
        # Create formatted draft
        updated_draft = current_draft.copy(deep=True)
        
        # Update main content with formatted output
        updated_draft.content["cross_disciplinary_report"] = formatted_output
        
        # Add cross-disciplinary sections
        updated_draft.content["disciplinary_perspectives"] = integrator._format_disciplinary_perspectives(
            integration_result.disciplinary_perspectives
        )
        
        if integration_result.conflicts_resolved:
            updated_draft.content["conflict_resolutions"] = integrator._format_conflict_resolutions(
                integration_result.conflicts_resolved
            )
        
        updated_draft.metadata.updated_at = datetime.now()
        
        # Create formatting metadata
        formatting_metadata = {
            "output_format": output_format,
            "sections_created": len([k for k, v in updated_draft.content.items() if v]),
            "formatting_timestamp": datetime.now().isoformat(),
            "cross_disciplinary_formatting": True,
            "coherence_maintained": integration_result.coherence_score > 0.6
        }
        
        # Update state
        updated_state = {
            **state,
            "current_draft": updated_draft,
            "formatting_metadata": formatting_metadata
        }
        
        logger.info(f"Cross-disciplinary formatting completed using {output_format} format")
        
        return updated_state
        
    except Exception as e:
        error_msg = f"Cross-disciplinary formatting failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Add error to state but don't fail the workflow
        error_log = state.get("error_log", [])
        error_log.append(f"Cross-Disciplinary Formatter: {error_msg}")
        
        return {
            **state,
            "error_log": error_log
        }


def cross_disciplinary_quality_assessor_node(state: TTDRState) -> TTDRState:
    """
    LangGraph node for assessing cross-disciplinary research quality.
    
    This node:
    1. Evaluates quality of cross-disciplinary integration
    2. Assesses coherence across disciplines
    3. Validates conflict resolution effectiveness
    4. Provides quality metrics for cross-disciplinary research
    
    Args:
        state: Current TTD-DR workflow state
        
    Returns:
        Updated state with cross-disciplinary quality assessment
    """
    try:
        logger.info("Starting cross-disciplinary quality assessment")
        
        # Check if we have cross-disciplinary content to assess
        integration_result = state.get("cross_disciplinary_integration")
        current_draft = state.get("current_draft")
        
        if not integration_result:
            logger.info("No cross-disciplinary integration to assess")
            return state
        
        # Initialize integrator for quality assessment
        kimi_client = KimiK2Client()
        integrator = CrossDisciplinaryIntegrator(kimi_client)
        
        # Assess cross-disciplinary quality metrics
        quality_metrics = {
            "integration_coherence": integration_result.coherence_score,
            "disciplinary_balance": integrator._assess_disciplinary_balance(integration_result),
            "conflict_resolution_effectiveness": integrator._assess_conflict_resolution(integration_result),
            "cross_domain_synthesis": integrator._assess_synthesis_quality(integration_result),
            "methodological_integration": integrator._assess_methodological_integration(integration_result)
        }
        
        # Calculate overall cross-disciplinary quality score
        overall_score = sum(quality_metrics.values()) / len(quality_metrics)
        
        # Create comprehensive quality assessment
        from ..models.core import QualityMetrics
        cross_disciplinary_quality = QualityMetrics(
            completeness=quality_metrics.get("integration_coherence", 0.5),
            coherence=quality_metrics.get("disciplinary_balance", 0.5),
            accuracy=quality_metrics.get("conflict_resolution_effectiveness", 0.5),
            citation_quality=quality_metrics.get("cross_domain_synthesis", 0.5)
        )
        
        # Add cross-disciplinary specific metrics
        cross_disciplinary_quality_metadata = {
            "cross_disciplinary_metrics": quality_metrics,
            "overall_cross_disciplinary_score": overall_score,
            "domains_integrated": len(integration_result.primary_domains),
            "perspectives_analyzed": len(integration_result.disciplinary_perspectives),
            "conflicts_resolved_ratio": (
                len(integration_result.conflicts_resolved) / 
                len(integration_result.conflicts_identified)
                if integration_result.conflicts_identified else 1.0
            ),
            "assessment_timestamp": datetime.now().isoformat(),
            "quality_threshold_met": overall_score > 0.7
        }
        
        # Update state with quality assessment
        updated_state = {
            **state,
            "quality_metrics": cross_disciplinary_quality,
            "cross_disciplinary_quality_metadata": cross_disciplinary_quality_metadata
        }
        
        logger.info(f"Cross-disciplinary quality assessment completed. Overall score: {overall_score:.2f}")
        
        return updated_state
        
    except Exception as e:
        error_msg = f"Cross-disciplinary quality assessment failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Add error to state but don't fail the workflow
        error_log = state.get("error_log", [])
        error_log.append(f"Cross-Disciplinary Quality Assessor: {error_msg}")
        
        return {
            **state,
            "error_log": error_log
        }


# Helper function for workflow integration
def create_cross_disciplinary_subgraph():
    """
    Create a subgraph for cross-disciplinary research workflow.
    This can be integrated into the main TTD-DR workflow.
    """
    from langgraph.graph import StateGraph, END
    from ..models.core import TTDRState
    
    # Create cross-disciplinary subgraph
    cross_disciplinary_graph = StateGraph(TTDRState)
    
    # Add cross-disciplinary nodes
    cross_disciplinary_graph.add_node("cross_disciplinary_detector", cross_disciplinary_detector_node)
    cross_disciplinary_graph.add_node("cross_disciplinary_integrator", cross_disciplinary_integrator_node)
    cross_disciplinary_graph.add_node("cross_disciplinary_conflict_resolver", cross_disciplinary_conflict_resolver_node)
    cross_disciplinary_graph.add_node("cross_disciplinary_formatter", cross_disciplinary_formatter_node)
    cross_disciplinary_graph.add_node("cross_disciplinary_quality_assessor", cross_disciplinary_quality_assessor_node)
    
    # Define edges
    cross_disciplinary_graph.set_entry_point("cross_disciplinary_detector")
    cross_disciplinary_graph.add_edge("cross_disciplinary_detector", "cross_disciplinary_integrator")
    cross_disciplinary_graph.add_edge("cross_disciplinary_integrator", "cross_disciplinary_conflict_resolver")
    cross_disciplinary_graph.add_edge("cross_disciplinary_conflict_resolver", "cross_disciplinary_formatter")
    cross_disciplinary_graph.add_edge("cross_disciplinary_formatter", "cross_disciplinary_quality_assessor")
    cross_disciplinary_graph.add_edge("cross_disciplinary_quality_assessor", END)
    
    return cross_disciplinary_graph.compile()


# Export functions for workflow integration
__all__ = [
    "cross_disciplinary_detector_node",
    "cross_disciplinary_integrator_node",
    "cross_disciplinary_conflict_resolver_node", 
    "cross_disciplinary_formatter_node",
    "cross_disciplinary_quality_assessor_node",
    "create_cross_disciplinary_subgraph"
]