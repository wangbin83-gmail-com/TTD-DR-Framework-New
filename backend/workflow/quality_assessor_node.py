"""
Quality assessor node implementation for TTD-DR LangGraph workflow.
Provides comprehensive quality evaluation using Kimi K2 model.
"""

import asyncio
import logging
from typing import Dict, Any, Optional

from models.core import TTDRState, QualityMetrics
from services.kimi_k2_quality_assessor import KimiK2QualityAssessor, KimiK2QualityChecker

logger = logging.getLogger(__name__)

async def quality_assessor_node_async(state: TTDRState) -> TTDRState:
    """
    Async implementation of quality assessor node for LangGraph integration
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with quality metrics
    """
    logger.info("Executing quality_assessor_node with Kimi K2 evaluation")
    
    try:
        # Check if we have a draft to assess
        if not state.get("current_draft"):
            logger.warning("No current draft available for quality assessment")
            
            # Return minimal quality metrics
            quality_metrics = QualityMetrics(
                completeness=0.0,
                coherence=0.0,
                accuracy=0.0,
                citation_quality=0.0,
                overall_score=0.0
            )
            
            return {
                **state,
                "quality_metrics": quality_metrics,
                "error_log": state.get("error_log", []) + ["No draft available for quality assessment"]
            }
        
        # Initialize Kimi K2 quality assessor
        quality_assessor = KimiK2QualityAssessor()
        
        # Perform comprehensive quality assessment
        logger.info("Starting Kimi K2 quality assessment")
        quality_metrics = await quality_assessor.evaluate_draft(
            draft=state["current_draft"],
            requirements=state.get("requirements")
        )
        
        logger.info(f"Quality assessment completed - Overall score: {quality_metrics.overall_score:.3f}")
        logger.info(f"Metrics breakdown - Completeness: {quality_metrics.completeness:.3f}, "
                   f"Coherence: {quality_metrics.coherence:.3f}, "
                   f"Accuracy: {quality_metrics.accuracy:.3f}, "
                   f"Citation: {quality_metrics.citation_quality:.3f}")
        
        # Update draft quality score
        updated_draft = state["current_draft"]
        updated_draft.quality_score = quality_metrics.overall_score
        
        return {
            **state,
            "current_draft": updated_draft,
            "quality_metrics": quality_metrics
        }
        
    except Exception as e:
        logger.error(f"Quality assessment failed: {e}")
        
        # Fallback to basic quality metrics
        fallback_metrics = QualityMetrics(
            completeness=0.4,
            coherence=0.4,
            accuracy=0.4,
            citation_quality=0.3,
            overall_score=0.375
        )
        
        return {
            **state,
            "quality_metrics": fallback_metrics,
            "error_log": state.get("error_log", []) + [f"Quality assessment error: {str(e)}"]
        }

def quality_assessor_node(state: TTDRState) -> TTDRState:
    """
    Synchronous wrapper for quality assessor node
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with quality metrics
    """
    try:
        # Run async function in event loop
        return asyncio.run(quality_assessor_node_async(state))
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
                    return loop.run_until_complete(quality_assessor_node_async(state))
                finally:
                    loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=60)  # 60 second timeout
        else:
            raise

async def quality_check_node_async(state: TTDRState) -> str:
    """
    Async implementation of quality check decision node
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node name based on quality assessment
    """
    logger.info("Executing quality_check_node with Kimi K2 intelligence")
    
    try:
        # Check if we have quality metrics
        if not state.get("quality_metrics") or not state.get("requirements"):
            logger.warning("Missing quality metrics or requirements for decision")
            return "gap_analyzer"  # Default to continuing iteration
        
        # Initialize Kimi K2 quality checker
        quality_checker = KimiK2QualityChecker()
        
        # Use Kimi K2 intelligence for continuation decision
        should_continue = await quality_checker.should_continue_iteration(
            quality_metrics=state["quality_metrics"],
            iteration_count=state.get("iteration_count", 0),
            requirements=state["requirements"]
        )
        
        if should_continue:
            logger.info("Kimi K2 decision: Continue iteration (gap_analyzer)")
            return "gap_analyzer"
        else:
            logger.info("Kimi K2 decision: Proceed to self-evolution (self_evolution_enhancer)")
            return "self_evolution_enhancer"
            
    except Exception as e:
        logger.error(f"Quality check decision failed: {e}")
        
        # Fallback to simple decision logic
        return quality_check_fallback(state)

def quality_check_node(state: TTDRState) -> str:
    """
    Synchronous wrapper for quality check decision node
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node name based on quality assessment
    """
    try:
        # Run async function in event loop
        return asyncio.run(quality_check_node_async(state))
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            # Handle case where event loop is already running
            logger.info("Event loop already running, using alternative approach for quality check")
            
            # Create new event loop in thread
            import concurrent.futures
            import threading
            
            def run_in_thread():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(quality_check_node_async(state))
                finally:
                    loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=30)  # 30 second timeout
        else:
            raise
    except Exception as e:
        logger.error(f"Quality check node failed: {e}")
        return quality_check_fallback(state)

def quality_check_fallback(state: TTDRState) -> str:
    """
    Fallback decision logic for quality check
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node name based on simple heuristics
    """
    logger.info("Using fallback quality check logic")
    
    # Get current metrics and requirements
    quality_metrics = state.get("quality_metrics")
    requirements = state.get("requirements")
    iteration_count = state.get("iteration_count", 0)
    
    if not quality_metrics or not requirements:
        logger.warning("Missing quality metrics or requirements, continuing iteration")
        return "gap_analyzer"
    
    # Simple decision logic
    quality_threshold = requirements.quality_threshold
    max_iterations = requirements.max_iterations
    
    # Stop if quality threshold is met or max iterations reached
    if (quality_metrics.overall_score >= quality_threshold or 
        iteration_count >= max_iterations):
        logger.info(f"Stopping iteration - Quality: {quality_metrics.overall_score:.3f}, "
                   f"Threshold: {quality_threshold}, Iteration: {iteration_count}/{max_iterations}")
        return "self_evolution_enhancer"
    else:
        logger.info(f"Continuing iteration - Quality: {quality_metrics.overall_score:.3f}, "
                   f"Threshold: {quality_threshold}, Iteration: {iteration_count}/{max_iterations}")
        return "gap_analyzer"

# Utility functions for quality assessment
def get_quality_summary(quality_metrics: QualityMetrics) -> Dict[str, Any]:
    """
    Get a summary of quality metrics for logging/debugging
    
    Args:
        quality_metrics: Quality metrics to summarize
        
    Returns:
        Dictionary with quality summary
    """
    return {
        "overall_score": round(quality_metrics.overall_score, 3),
        "completeness": round(quality_metrics.completeness, 3),
        "coherence": round(quality_metrics.coherence, 3),
        "accuracy": round(quality_metrics.accuracy, 3),
        "citation_quality": round(quality_metrics.citation_quality, 3),
        "quality_grade": get_quality_grade(quality_metrics.overall_score)
    }

def get_quality_grade(score: float) -> str:
    """
    Convert quality score to letter grade
    
    Args:
        score: Quality score (0.0 to 1.0)
        
    Returns:
        Letter grade representation
    """
    if score >= 0.9:
        return "A"
    elif score >= 0.8:
        return "B"
    elif score >= 0.7:
        return "C"
    elif score >= 0.6:
        return "D"
    else:
        return "F"

def assess_improvement_potential(current_metrics: QualityMetrics, 
                               previous_metrics: Optional[QualityMetrics] = None) -> Dict[str, Any]:
    """
    Assess potential for quality improvement
    
    Args:
        current_metrics: Current quality metrics
        previous_metrics: Previous iteration metrics (optional)
        
    Returns:
        Dictionary with improvement analysis
    """
    analysis = {
        "current_score": current_metrics.overall_score,
        "improvement_areas": [],
        "strengths": []
    }
    
    # Identify improvement areas (scores below 0.7)
    if current_metrics.completeness < 0.7:
        analysis["improvement_areas"].append("completeness")
    if current_metrics.coherence < 0.7:
        analysis["improvement_areas"].append("coherence")
    if current_metrics.accuracy < 0.7:
        analysis["improvement_areas"].append("accuracy")
    if current_metrics.citation_quality < 0.7:
        analysis["improvement_areas"].append("citation_quality")
    
    # Identify strengths (scores above 0.8)
    if current_metrics.completeness >= 0.8:
        analysis["strengths"].append("completeness")
    if current_metrics.coherence >= 0.8:
        analysis["strengths"].append("coherence")
    if current_metrics.accuracy >= 0.8:
        analysis["strengths"].append("accuracy")
    if current_metrics.citation_quality >= 0.8:
        analysis["strengths"].append("citation_quality")
    
    # Calculate improvement trend if previous metrics available
    if previous_metrics:
        improvement = current_metrics.overall_score - previous_metrics.overall_score
        analysis["improvement_trend"] = improvement
        analysis["is_improving"] = improvement > 0.01  # Threshold for meaningful improvement
    
    return analysis