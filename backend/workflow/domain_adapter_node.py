"""
Domain adapter node for LangGraph workflow integration.
This module provides the workflow node for domain-specific adaptation
within the TTD-DR framework.
"""

from typing import Dict, Any, Optional
import logging
from datetime import datetime

from models.core import TTDRState, ResearchRequirements, ResearchDomain
from models.research_structure import EnhancedResearchStructure
from services.domain_adapter import DomainAdapter, DomainDetectionResult
from services.kimi_k2_client import KimiK2Client

logger = logging.getLogger(__name__)


def domain_adapter_node(state: TTDRState) -> TTDRState:
    """
    LangGraph node for domain-specific adaptation.
    
    This node:
    1. Detects the research domain from the topic
    2. Adapts research requirements for the domain
    3. Generates domain-specific research structure
    4. Updates state with domain-adapted configuration
    
    Args:
        state: Current TTD-DR workflow state
        
    Returns:
        Updated state with domain adaptations applied
    """
    try:
        logger.info(f"Starting domain adaptation for topic: {state['topic']}")
        
        # Initialize domain adapter
        kimi_client = KimiK2Client()
        domain_adapter = DomainAdapter(kimi_client)
        
        # Detect domain from topic
        domain_result = domain_adapter.detect_domain(
            topic=state["topic"],
            content=None  # Could include existing draft content if available
        )
        
        logger.info(f"Detected domain: {domain_result.primary_domain.value} "
                   f"(confidence: {domain_result.confidence:.2f})")
        
        # Adapt research requirements based on detected domain
        adapted_requirements = domain_adapter.adapt_research_requirements(
            requirements=state["requirements"],
            domain_result=domain_result
        )
        
        # Generate domain-specific research structure
        enhanced_structure = domain_adapter.generate_domain_specific_structure(
            topic=state["topic"],
            domain=domain_result.primary_domain,
            complexity_level=adapted_requirements.complexity_level
        )
        
        # Create or update draft with domain-specific structure
        if state.get("current_draft"):
            # Update existing draft with domain adaptations
            updated_draft = state["current_draft"].copy(deep=True)
            updated_draft.structure = enhanced_structure
            updated_draft.metadata.updated_at = datetime.now()
        else:
            # Create new draft with domain-specific structure
            from ..models.core import Draft, DraftMetadata
            updated_draft = Draft(
                id=f"draft_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                topic=state["topic"],
                structure=enhanced_structure,
                content={},
                metadata=DraftMetadata(),
                quality_score=0.0,
                iteration=0
            )
        
        # Update information gaps with domain-specific adaptations
        adapted_gaps = []
        if state.get("information_gaps"):
            adapted_gaps = domain_adapter.adapt_search_queries(
                gaps=state["information_gaps"],
                domain=domain_result.primary_domain
            )
        
        # Store domain adaptation metadata
        domain_metadata = {
            "detection_result": {
                "primary_domain": domain_result.primary_domain.value,
                "confidence": domain_result.confidence,
                "detection_method": domain_result.detection_method,
                "keywords_found": domain_result.keywords_found,
                "reasoning": domain_result.reasoning
            },
            "adaptations_applied": {
                "requirements_adapted": True,
                "structure_generated": True,
                "search_queries_enhanced": len(adapted_gaps) > 0,
                "timestamp": datetime.now().isoformat()
            },
            "domain_strategy": {
                "preferred_sources": adapted_requirements.preferred_source_types,
                "quality_threshold": adapted_requirements.quality_threshold,
                "max_sources": adapted_requirements.max_sources
            }
        }
        
        # Update state with domain adaptations
        updated_state = {
            **state,
            "requirements": adapted_requirements,
            "current_draft": updated_draft,
            "information_gaps": adapted_gaps,
            "domain_metadata": domain_metadata
        }
        
        logger.info(f"Domain adaptation completed successfully for {domain_result.primary_domain.value}")
        
        return updated_state
        
    except Exception as e:
        error_msg = f"Domain adaptation failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Add error to state but don't fail the workflow
        error_log = state.get("error_log", [])
        error_log.append(f"Domain Adapter: {error_msg}")
        
        return {
            **state,
            "error_log": error_log
        }


def domain_quality_assessor_node(state: TTDRState) -> TTDRState:
    """
    LangGraph node for domain-specific quality assessment.
    
    This node applies domain-specific quality criteria and assessment methods
    to evaluate the current draft quality.
    
    Args:
        state: Current TTD-DR workflow state
        
    Returns:
        Updated state with domain-specific quality assessment
    """
    try:
        logger.info("Starting domain-specific quality assessment")
        
        # Get domain from metadata or requirements
        domain = ResearchDomain.GENERAL
        if state.get("domain_metadata"):
            domain_str = state["domain_metadata"]["detection_result"]["primary_domain"]
            domain = ResearchDomain(domain_str)
        elif state.get("requirements"):
            domain = state["requirements"].domain
        
        # Initialize domain adapter for quality assessment
        kimi_client = KimiK2Client()
        domain_adapter = DomainAdapter(kimi_client)
        
        # Get domain-specific quality criteria
        domain_criteria = domain_adapter.get_domain_quality_criteria(domain)
        
        # Apply domain-specific quality assessment
        if state.get("current_draft"):
            draft = state["current_draft"]
            
            # Get domain-specific system prompt for quality assessment
            system_prompt = domain_adapter.get_kimi_system_prompt(domain, "quality_assessment")
            
            # Perform domain-aware quality assessment using Kimi K2
            assessment_prompt = f"""
            {system_prompt}
            
            Assess the quality of this {domain.value} research draft based on domain-specific criteria:
            
            Domain Criteria:
            {json.dumps(domain_criteria, indent=2)}
            
            Draft Topic: {draft.topic}
            Draft Content: {json.dumps(draft.content, indent=2)}
            
            Provide scores (0.0 to 1.0) for each criterion and overall assessment.
            """
            
            try:
                response = kimi_client.generate_content(
                    prompt=assessment_prompt,
                    temperature=0.3,
                    max_tokens=1000
                )
                
                # Parse quality assessment (simplified - would need more robust parsing)
                import json
                assessment_data = json.loads(response)
                
                from ..models.core import QualityMetrics
                quality_metrics = QualityMetrics(
                    completeness=assessment_data.get("completeness", 0.5),
                    coherence=assessment_data.get("coherence", 0.5),
                    accuracy=assessment_data.get("accuracy", 0.5),
                    citation_quality=assessment_data.get("citation_quality", 0.5)
                )
                
                # Add domain-specific metrics
                domain_quality_metadata = {
                    "domain_criteria_scores": domain_criteria,
                    "assessment_method": "domain_specific_kimi",
                    "domain": domain.value,
                    "timestamp": datetime.now().isoformat()
                }
                
                updated_state = {
                    **state,
                    "quality_metrics": quality_metrics,
                    "domain_quality_metadata": domain_quality_metadata
                }
                
                logger.info(f"Domain-specific quality assessment completed. Overall score: {quality_metrics.overall_score:.2f}")
                
                return updated_state
                
            except Exception as e:
                logger.warning(f"Kimi K2 quality assessment failed, using fallback: {e}")
                
                # Fallback to basic quality assessment
                from ..models.core import QualityMetrics
                fallback_metrics = QualityMetrics(
                    completeness=0.6,
                    coherence=0.6,
                    accuracy=0.6,
                    citation_quality=0.6
                )
                
                return {
                    **state,
                    "quality_metrics": fallback_metrics
                }
        
        else:
            logger.warning("No current draft available for quality assessment")
            return state
            
    except Exception as e:
        error_msg = f"Domain quality assessment failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        error_log = state.get("error_log", [])
        error_log.append(f"Domain Quality Assessor: {error_msg}")
        
        return {
            **state,
            "error_log": error_log
        }


def domain_content_formatter_node(state: TTDRState) -> TTDRState:
    """
    LangGraph node for domain-specific content formatting.
    
    This node applies domain-specific formatting, terminology,
    and style guidelines to the research content.
    
    Args:
        state: Current TTD-DR workflow state
        
    Returns:
        Updated state with domain-formatted content
    """
    try:
        logger.info("Starting domain-specific content formatting")
        
        # Get domain information
        domain = ResearchDomain.GENERAL
        if state.get("domain_metadata"):
            domain_str = state["domain_metadata"]["detection_result"]["primary_domain"]
            domain = ResearchDomain(domain_str)
        elif state.get("requirements"):
            domain = state["requirements"].domain
        
        # Initialize domain adapter
        kimi_client = KimiK2Client()
        domain_adapter = DomainAdapter(kimi_client)
        
        # Format content if draft exists
        if state.get("current_draft"):
            draft = state["current_draft"]
            formatted_content = {}
            
            # Apply domain-specific formatting to each section
            for section_id, content in draft.content.items():
                if content:  # Only format non-empty content
                    formatted_content[section_id] = domain_adapter.apply_domain_formatting(
                        content=content,
                        domain=domain,
                        section_type=section_id
                    )
                else:
                    formatted_content[section_id] = content
            
            # Update draft with formatted content
            updated_draft = draft.copy(deep=True)
            updated_draft.content = formatted_content
            updated_draft.metadata.updated_at = datetime.now()
            
            # Add formatting metadata
            formatting_metadata = {
                "domain": domain.value,
                "sections_formatted": len([c for c in formatted_content.values() if c]),
                "formatting_timestamp": datetime.now().isoformat(),
                "formatting_applied": True
            }
            
            updated_state = {
                **state,
                "current_draft": updated_draft,
                "formatting_metadata": formatting_metadata
            }
            
            logger.info(f"Domain-specific formatting applied to {len(formatted_content)} sections")
            
            return updated_state
        
        else:
            logger.warning("No current draft available for formatting")
            return state
            
    except Exception as e:
        error_msg = f"Domain content formatting failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        error_log = state.get("error_log", [])
        error_log.append(f"Domain Content Formatter: {error_msg}")
        
        return {
            **state,
            "error_log": error_log
        }


# Helper function for workflow integration
def create_domain_adaptation_subgraph():
    """
    Create a subgraph for domain adaptation workflow.
    This can be integrated into the main TTD-DR workflow.
    """
    from langgraph.graph import StateGraph, END
    from ..models.core import TTDRState
    
    # Create domain adaptation subgraph
    domain_graph = StateGraph(TTDRState)
    
    # Add domain adaptation nodes
    domain_graph.add_node("domain_adapter", domain_adapter_node)
    domain_graph.add_node("domain_quality_assessor", domain_quality_assessor_node)
    domain_graph.add_node("domain_content_formatter", domain_content_formatter_node)
    
    # Define edges
    domain_graph.set_entry_point("domain_adapter")
    domain_graph.add_edge("domain_adapter", "domain_quality_assessor")
    domain_graph.add_edge("domain_quality_assessor", "domain_content_formatter")
    domain_graph.add_edge("domain_content_formatter", END)
    
    return domain_graph.compile()


# Export functions for workflow integration
__all__ = [
    "domain_adapter_node",
    "domain_quality_assessor_node", 
    "domain_content_formatter_node",
    "create_domain_adaptation_subgraph"
]