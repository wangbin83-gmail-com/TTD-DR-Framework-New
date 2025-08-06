"""
Report synthesizer node implementation for TTD-DR LangGraph workflow.
Generates final polished research reports using Kimi K2 model.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List

from models.core import TTDRState
from services.kimi_k2_report_synthesizer import KimiK2ReportSynthesizer

logger = logging.getLogger(__name__)

async def report_synthesizer_node_async(state: TTDRState) -> TTDRState:
    """
    Async implementation of report synthesizer node for LangGraph integration
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with final synthesized report
    """
    logger.info("Executing report_synthesizer_node with Kimi K2 synthesis")
    
    try:
        # Check if we have required components for synthesis
        if not state.get("current_draft"):
            logger.error("No current draft available for report synthesis")
            return {
                **state,
                "final_report": "Error: No draft available for synthesis",
                "error_log": state.get("error_log", []) + ["No draft available for report synthesis"]
            }
        
        if not state.get("quality_metrics"):
            logger.warning("No quality metrics available for synthesis context")
        
        # Initialize Kimi K2 report synthesizer
        report_synthesizer = KimiK2ReportSynthesizer()
        
        # Perform comprehensive report synthesis
        logger.info("Starting Kimi K2 report synthesis")
        final_report = await report_synthesizer.synthesize_report(
            draft=state["current_draft"],
            quality_metrics=state.get("quality_metrics"),
            evolution_history=state.get("evolution_history", []),
            requirements=state.get("requirements")
        )
        
        # Validate the synthesized report quality
        logger.info("Validating synthesized report quality")
        validation_results = await report_synthesizer.validate_report_quality(
            final_report, state["current_draft"]
        )
        
        # Generate executive summary
        logger.info("Generating executive summary")
        executive_summary = await report_synthesizer.generate_executive_summary(
            final_report, max_length=300
        )
        
        # Generate research methodology documentation (Task 9.2)
        logger.info("Generating research methodology documentation")
        methodology_documentation = await report_synthesizer.generate_research_methodology_documentation(
            state, workflow_log=state.get("workflow_log")
        )
        
        # Generate source bibliography
        logger.info("Generating source bibliography")
        source_bibliography = await report_synthesizer.generate_source_bibliography(
            state.get("retrieved_info", []), citation_style="APA"
        )
        
        # Generate methodology summary for inclusion in main report
        logger.info("Generating methodology summary")
        methodology_summary = await report_synthesizer.generate_methodology_summary(state)
        
        # Log synthesis results
        overall_quality = validation_results.get("overall_quality", 0.0)
        improvement = validation_results.get("improvement_over_draft", 0.0)
        recommendation = validation_results.get("recommendation", "unknown")
        
        logger.info(f"Report synthesis completed - Quality: {overall_quality:.3f}, "
                   f"Improvement: {improvement:+.3f}, Recommendation: {recommendation}")
        logger.info(f"Final report length: {len(final_report)} characters")
        logger.info(f"Executive summary length: {len(executive_summary)} characters")
        logger.info(f"Methodology documentation length: {len(methodology_documentation)} characters")
        logger.info(f"Source bibliography length: {len(source_bibliography)} characters")
        
        # Store synthesis metadata
        synthesis_metadata = {
            "synthesis_timestamp": asyncio.get_event_loop().time(),
            "validation_results": validation_results,
            "executive_summary": executive_summary,
            "methodology_documentation": methodology_documentation,
            "source_bibliography": source_bibliography,
            "methodology_summary": methodology_summary,
            "original_draft_quality": state["current_draft"].quality_score,
            "synthesis_method": "kimi_k2_powered"
        }
        
        return {
            **state,
            "final_report": final_report,
            "synthesis_metadata": synthesis_metadata
        }
        
    except Exception as e:
        logger.error(f"Report synthesis failed: {e}")
        
        # Generate fallback report
        fallback_report = _generate_emergency_fallback_report(state)
        
        return {
            **state,
            "final_report": fallback_report,
            "error_log": state.get("error_log", []) + [f"Report synthesis error: {str(e)}"],
            "synthesis_metadata": {
                "synthesis_method": "emergency_fallback",
                "error": str(e)
            }
        }

def report_synthesizer_node(state: TTDRState) -> TTDRState:
    """
    Synchronous wrapper for report synthesizer node
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with final synthesized report
    """
    try:
        # Run async function in event loop
        return asyncio.run(report_synthesizer_node_async(state))
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            # Handle case where event loop is already running
            logger.info("Event loop already running, using alternative approach")
            
            # Create new event loop in thread
            import concurrent.futures
            import threading
            
            def run_in_thread():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(report_synthesizer_node_async(state))
                finally:
                    loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=180)  # 3 minute timeout for synthesis
        else:
            raise

def _generate_emergency_fallback_report(state: TTDRState) -> str:
    """
    Generate emergency fallback report when all synthesis methods fail
    
    Args:
        state: Current workflow state
        
    Returns:
        Basic fallback report
    """
    logger.info("Generating emergency fallback report")
    
    # Extract basic information
    topic = "Unknown Topic"
    content_sections = []
    quality_score = 0.0
    iteration_count = 0
    
    if state.get("current_draft"):
        draft = state["current_draft"]
        topic = draft.topic
        quality_score = draft.quality_score
        iteration_count = draft.iteration
        
        # Combine draft content
        for section_id, content in draft.content.items():
            if content.strip():
                content_sections.append(f"## {section_id}\n\n{content}")
    
    # Get quality metrics if available
    quality_info = ""
    if state.get("quality_metrics"):
        metrics = state["quality_metrics"]
        quality_info = f"""
**Quality Assessment:**
- Overall Score: {metrics.overall_score:.3f}
- Completeness: {metrics.completeness:.3f}
- Coherence: {metrics.coherence:.3f}
- Accuracy: {metrics.accuracy:.3f}
- Citation Quality: {metrics.citation_quality:.3f}
"""
    
    # Get evolution summary if available
    evolution_info = ""
    if state.get("evolution_history"):
        evolution_count = len(state["evolution_history"])
        recent_improvements = [
            r.performance_after - r.performance_before 
            for r in state["evolution_history"][-3:]
        ]
        avg_improvement = sum(recent_improvements) / len(recent_improvements) if recent_improvements else 0.0
        
        evolution_info = f"""
**Evolution Summary:**
- Total Evolution Cycles: {evolution_count}
- Recent Average Improvement: {avg_improvement:+.3f}
"""
    
    fallback_report = f"""# {topic}

**Research Report - Emergency Generation**  
Generated by TTD-DR Framework  
Date: {asyncio.get_event_loop().time()}

## Executive Summary

This research report on "{topic}" was generated through the TTD-DR (Test-Time Diffusion Deep Researcher) framework. Due to technical limitations in the synthesis process, this report represents a basic compilation of the research content developed through {iteration_count} iterations of refinement.

{quality_info}

{evolution_info}

## Research Content

{chr(10).join(content_sections) if content_sections else "No detailed content available. The research process encountered technical difficulties during content generation."}

## Methodology

This report was generated using the TTD-DR framework, which employs:

1. **Initial Draft Generation**: Creating a structured research skeleton
2. **Iterative Refinement**: Multiple cycles of gap analysis and information retrieval
3. **Quality Assessment**: Continuous evaluation of research completeness and coherence
4. **Self-Evolution**: Adaptive improvement of research strategies
5. **Report Synthesis**: Final compilation and formatting (emergency mode)

## Limitations

This report was generated using emergency fallback procedures due to technical limitations in the primary synthesis system. The content may lack the polish and integration typically provided by the full TTD-DR synthesis process.

## Conclusion

The research on "{topic}" has been compiled through an iterative process designed to ensure comprehensive coverage. While this emergency report format may not reflect the full capabilities of the TTD-DR framework, it represents the best available synthesis of the research content developed during the workflow execution.

---

*Report generated using Test-Time Diffusion Deep Researcher (TTD-DR) Framework*  
*Framework Version: 1.0*  
*Generation Method: Emergency Fallback*  
*Final Quality Score: {quality_score:.3f}*  
*Iterations Completed: {iteration_count}*
"""
    
    return fallback_report

# Utility functions for report synthesis
def get_synthesis_summary(state: TTDRState) -> Dict[str, Any]:
    """
    Get a summary of the synthesis process for logging/debugging
    
    Args:
        state: Current workflow state
        
    Returns:
        Dictionary with synthesis summary
    """
    summary = {
        "has_final_report": bool(state.get("final_report")),
        "report_length": len(state.get("final_report", "")) if state.get("final_report") else 0,
        "synthesis_method": "unknown"
    }
    
    # Extract synthesis metadata if available
    if state.get("synthesis_metadata"):
        metadata = state["synthesis_metadata"]
        summary.update({
            "synthesis_method": metadata.get("synthesis_method", "unknown"),
            "validation_results": metadata.get("validation_results", {}),
            "has_executive_summary": bool(metadata.get("executive_summary"))
        })
        
        # Add validation summary
        if "validation_results" in metadata:
            validation = metadata["validation_results"]
            summary["validation_summary"] = {
                "overall_quality": validation.get("overall_quality", 0.0),
                "improvement_over_draft": validation.get("improvement_over_draft", 0.0),
                "recommendation": validation.get("recommendation", "unknown")
            }
    
    return summary

def assess_synthesis_quality(final_report: str, original_draft) -> Dict[str, Any]:
    """
    Assess the quality of the synthesized report
    
    Args:
        final_report: Final synthesized report
        original_draft: Original draft for comparison
        
    Returns:
        Dictionary with quality assessment
    """
    if not final_report:
        return {
            "status": "no_report",
            "quality_score": 0.0,
            "issues": ["No final report generated"]
        }
    
    # Basic quality metrics
    report_length = len(final_report.split())
    has_structure = "# " in final_report and "## " in final_report
    has_conclusion = "conclusion" in final_report.lower() or "summary" in final_report.lower()
    
    # Compare with original draft
    draft_length = 0
    if original_draft and hasattr(original_draft, 'content'):
        draft_length = sum(len(content.split()) for content in original_draft.content.values())
    
    # Calculate quality indicators
    length_score = min(1.0, report_length / max(500, draft_length))  # Expect at least 500 words
    structure_score = 1.0 if has_structure else 0.5
    completion_score = 1.0 if has_conclusion else 0.7
    
    overall_quality = (length_score + structure_score + completion_score) / 3
    
    # Identify issues
    issues = []
    if report_length < 200:
        issues.append("Report is very short")
    if not has_structure:
        issues.append("Missing proper heading structure")
    if not has_conclusion:
        issues.append("Missing conclusion or summary")
    if "error" in final_report.lower() or "fallback" in final_report.lower():
        issues.append("Report generated using fallback methods")
    
    return {
        "status": "assessed",
        "quality_score": overall_quality,
        "report_length": report_length,
        "has_structure": has_structure,
        "has_conclusion": has_conclusion,
        "length_improvement": report_length - draft_length if draft_length > 0 else report_length,
        "issues": issues,
        "recommendations": _generate_quality_recommendations(overall_quality, issues)
    }

def _generate_quality_recommendations(quality_score: float, issues: List[str]) -> List[str]:
    """Generate recommendations based on quality assessment"""
    recommendations = []
    
    if quality_score < 0.5:
        recommendations.append("Consider regenerating the report with improved synthesis parameters")
    elif quality_score < 0.7:
        recommendations.append("Report quality is acceptable but could benefit from manual review")
    else:
        recommendations.append("Report quality is good and ready for use")
    
    # Issue-specific recommendations
    if "Report is very short" in issues:
        recommendations.append("Increase content depth in future iterations")
    if "Missing proper heading structure" in issues:
        recommendations.append("Improve structural formatting in synthesis process")
    if "Missing conclusion or summary" in issues:
        recommendations.append("Ensure synthesis includes comprehensive conclusion")
    if "Report generated using fallback methods" in issues:
        recommendations.append("Investigate and resolve synthesis system issues")
    
    return recommendations