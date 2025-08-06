"""
Self-evolution enhancer node implementation for TTD-DR LangGraph workflow.
Applies intelligent learning algorithms using Kimi K2 model.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List

from models.core import TTDRState, EvolutionRecord
from services.kimi_k2_self_evolution_enhancer import (
    KimiK2SelfEvolutionEnhancer, EvolutionHistoryManager
)

logger = logging.getLogger(__name__)

async def self_evolution_enhancer_node_async(state: TTDRState) -> TTDRState:
    """
    Async implementation of self-evolution enhancer node for LangGraph integration
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with evolution improvements applied
    """
    logger.info("Executing self_evolution_enhancer_node with Kimi K2 intelligence")
    
    try:
        # Check if we have quality metrics for evolution
        if not state.get("quality_metrics"):
            logger.warning("No quality metrics available for self-evolution")
            
            # Create minimal evolution record
            minimal_record = EvolutionRecord(
                component="framework_wide",
                improvement_type="no_evolution",
                description="No quality metrics available for evolution",
                performance_before=0.0,
                performance_after=0.0,
                parameters_changed={}
            )
            
            return {
                **state,
                "evolution_history": state.get("evolution_history", []) + [minimal_record],
                "error_log": state.get("error_log", []) + ["No quality metrics for self-evolution"]
            }
        
        # Initialize Kimi K2 self-evolution enhancer
        evolution_enhancer = KimiK2SelfEvolutionEnhancer()
        evolution_history_manager = EvolutionHistoryManager()
        
        # Analyze current evolution trends
        evolution_trends = evolution_history_manager.analyze_evolution_trends(
            state.get("evolution_history", [])
        )
        
        trend = evolution_trends.get('trend', 'unknown')
        improvement = evolution_trends.get('overall_improvement', 0.0)
        logger.info(f"Evolution trends analysis: {trend} (overall improvement: {improvement:.3f})")
        
        # Apply self-evolution algorithms
        logger.info("Starting Kimi K2 self-evolution enhancement")
        evolution_record = await evolution_enhancer.evolve_components(
            quality_metrics=state["quality_metrics"],
            evolution_history=state.get("evolution_history", []),
            current_state=state
        )
        
        # Get performance metrics for logging
        performance_metrics = evolution_history_manager.get_performance_metrics(
            state.get("evolution_history", []) + [evolution_record]
        )
        
        logger.info(f"Self-evolution completed - Performance change: "
                   f"{evolution_record.performance_after - evolution_record.performance_before:.3f}")
        logger.info(f"Evolution metrics - Success rate: {performance_metrics.get('success_rate', 0.0):.3f}, "
                   f"Stability: {performance_metrics.get('performance_stability', 0.0):.3f}")
        
        # Update evolution history
        updated_evolution_history = state.get("evolution_history", []) + [evolution_record]
        
        # Apply any framework-wide improvements if specified
        updated_state = state.copy()
        if evolution_record.parameters_changed:
            logger.info(f"Applied {len(evolution_record.parameters_changed)} parameter changes")
            # Store parameter changes in state for potential use by other components
            updated_state["framework_parameters"] = updated_state.get("framework_parameters", {})
            updated_state["framework_parameters"].update(evolution_record.parameters_changed)
        
        return {
            **updated_state,
            "evolution_history": updated_evolution_history
        }
        
    except Exception as e:
        logger.error(f"Self-evolution enhancement failed: {e}")
        
        # Create error evolution record
        current_performance = 0.0
        if state.get("quality_metrics"):
            current_performance = state["quality_metrics"].overall_score
        
        error_record = EvolutionRecord(
            component="framework_wide",
            improvement_type="evolution_error",
            description=f"Evolution failed: {str(e)}",
            performance_before=current_performance,
            performance_after=current_performance,
            parameters_changed={}
        )
        
        return {
            **state,
            "evolution_history": state.get("evolution_history", []) + [error_record],
            "error_log": state.get("error_log", []) + [f"Self-evolution error: {str(e)}"]
        }

def self_evolution_enhancer_node(state: TTDRState) -> TTDRState:
    """
    Synchronous wrapper for self-evolution enhancer node
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with evolution improvements applied
    """
    try:
        # Run async function in event loop
        return asyncio.run(self_evolution_enhancer_node_async(state))
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
                    return loop.run_until_complete(self_evolution_enhancer_node_async(state))
                finally:
                    loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=120)  # 2 minute timeout for evolution
        else:
            raise

# Utility functions for evolution analysis
def analyze_evolution_effectiveness(evolution_history: List[EvolutionRecord]) -> Dict[str, Any]:
    """
    Analyze the effectiveness of evolution strategies
    
    Args:
        evolution_history: List of evolution records to analyze
        
    Returns:
        Dictionary with effectiveness analysis
    """
    if not evolution_history:
        return {"status": "no_data", "recommendations": ["Start collecting evolution data"]}
    
    # Group by improvement type
    improvement_types = {}
    for record in evolution_history:
        imp_type = record.improvement_type
        if imp_type not in improvement_types:
            improvement_types[imp_type] = []
        improvement_types[imp_type].append(record.performance_after - record.performance_before)
    
    # Analyze effectiveness by type
    type_effectiveness = {}
    for imp_type, improvements in improvement_types.items():
        avg_improvement = sum(improvements) / len(improvements)
        success_rate = len([imp for imp in improvements if imp > 0]) / len(improvements)
        
        # Calculate effectiveness score (0.0 to 1.0 range)
        # Normalize average improvement to 0-1 range and combine with success rate
        normalized_improvement = max(0.0, min(1.0, avg_improvement + 0.5))  # Shift range to handle negatives
        effectiveness_score = (normalized_improvement + success_rate) / 2
        
        type_effectiveness[imp_type] = {
            "average_improvement": avg_improvement,
            "success_rate": success_rate,
            "total_attempts": len(improvements),
            "effectiveness_score": effectiveness_score
        }
    
    # Overall analysis
    total_improvements = [r.performance_after - r.performance_before for r in evolution_history]
    overall_success_rate = len([imp for imp in total_improvements if imp > 0]) / len(total_improvements)
    overall_avg_improvement = sum(total_improvements) / len(total_improvements)
    
    # Generate recommendations
    recommendations = []
    
    # Find most effective strategies
    if type_effectiveness:
        best_strategy = max(type_effectiveness.items(), key=lambda x: x[1]["effectiveness_score"])
        recommendations.append(f"Focus on '{best_strategy[0]}' strategy (effectiveness: {best_strategy[1]['effectiveness_score']:.3f})")
        
        # Find least effective strategies
        worst_strategy = min(type_effectiveness.items(), key=lambda x: x[1]["effectiveness_score"])
        if worst_strategy[1]["effectiveness_score"] < 0.3:
            recommendations.append(f"Consider revising '{worst_strategy[0]}' strategy (low effectiveness: {worst_strategy[1]['effectiveness_score']:.3f})")
    
    # Success rate recommendations
    if overall_success_rate < 0.5:
        recommendations.append("Overall success rate is low - consider more conservative evolution strategies")
    elif overall_success_rate > 0.8:
        recommendations.append("High success rate - consider more aggressive evolution strategies")
    
    # Improvement magnitude recommendations
    if overall_avg_improvement < 0.01:
        recommendations.append("Average improvements are small - consider larger parameter changes")
    elif overall_avg_improvement > 0.2:
        recommendations.append("Large improvements detected - monitor for stability issues")
    
    return {
        "status": "analyzed",
        "overall_success_rate": overall_success_rate,
        "overall_avg_improvement": overall_avg_improvement,
        "strategy_effectiveness": type_effectiveness,
        "total_evolution_attempts": len(evolution_history),
        "recommendations": recommendations
    }

def get_evolution_summary(evolution_history: List[EvolutionRecord], 
                         recent_count: int = 5) -> Dict[str, Any]:
    """
    Get a summary of recent evolution activities
    
    Args:
        evolution_history: List of evolution records
        recent_count: Number of recent records to summarize
        
    Returns:
        Dictionary with evolution summary
    """
    if not evolution_history:
        return {
            "status": "no_evolution_data",
            "recent_activities": [],
            "performance_trend": "unknown"
        }
    
    # Get recent records
    recent_records = evolution_history[-recent_count:] if len(evolution_history) > recent_count else evolution_history
    
    # Summarize recent activities
    recent_activities = []
    for record in recent_records:
        activity = {
            "timestamp": record.timestamp.isoformat(),
            "component": record.component,
            "improvement_type": record.improvement_type,
            "performance_change": record.performance_after - record.performance_before,
            "description": record.description[:100] + "..." if len(record.description) > 100 else record.description
        }
        recent_activities.append(activity)
    
    # Calculate performance trend
    if len(recent_records) >= 2:
        improvements = [r.performance_after - r.performance_before for r in recent_records]
        avg_improvement = sum(improvements) / len(improvements)
        
        if avg_improvement > 0.02:
            trend = "strongly_improving"
        elif avg_improvement > 0.005:
            trend = "improving"
        elif avg_improvement > -0.005:
            trend = "stable"
        elif avg_improvement > -0.02:
            trend = "declining"
        else:
            trend = "strongly_declining"
    else:
        trend = "insufficient_data"
    
    # Component activity summary
    component_activity = {}
    for record in recent_records:
        component = record.component
        if component not in component_activity:
            component_activity[component] = {"count": 0, "total_improvement": 0.0}
        
        component_activity[component]["count"] += 1
        component_activity[component]["total_improvement"] += record.performance_after - record.performance_before
    
    return {
        "status": "active",
        "recent_activities": recent_activities,
        "performance_trend": trend,
        "component_activity": component_activity,
        "total_evolution_records": len(evolution_history),
        "recent_records_analyzed": len(recent_records)
    }

def predict_evolution_impact(current_quality: float, evolution_history: List[EvolutionRecord]) -> Dict[str, Any]:
    """
    Predict the potential impact of future evolution steps
    
    Args:
        current_quality: Current overall quality score
        evolution_history: Historical evolution data
        
    Returns:
        Dictionary with impact predictions
    """
    if not evolution_history:
        return {
            "predicted_improvement": 0.05,  # Conservative default
            "confidence": 0.3,
            "recommendation": "Start with conservative evolution strategies"
        }
    
    # Analyze historical patterns
    recent_improvements = [
        r.performance_after - r.performance_before 
        for r in evolution_history[-10:]  # Last 10 records
    ]
    
    if not recent_improvements:
        return {
            "predicted_improvement": 0.05,
            "confidence": 0.3,
            "recommendation": "Insufficient evolution history"
        }
    
    # Calculate prediction metrics
    avg_improvement = sum(recent_improvements) / len(recent_improvements)
    improvement_variance = sum((imp - avg_improvement) ** 2 for imp in recent_improvements) / len(recent_improvements)
    improvement_stability = max(0.0, 1.0 - improvement_variance)
    
    # Predict next improvement
    predicted_improvement = avg_improvement
    
    # Adjust for current quality level (diminishing returns)
    if current_quality > 0.8:
        predicted_improvement *= 0.5  # Harder to improve when already good
    elif current_quality < 0.5:
        predicted_improvement *= 1.5  # More room for improvement
    
    # Calculate confidence based on stability and history length
    confidence = min(1.0, improvement_stability * (len(recent_improvements) / 10))
    
    # Generate recommendation
    if predicted_improvement > 0.05 and confidence > 0.7:
        recommendation = "High potential for improvement - proceed with evolution"
    elif predicted_improvement > 0.02 and confidence > 0.5:
        recommendation = "Moderate improvement potential - continue evolution"
    elif predicted_improvement < 0:
        recommendation = "Negative trend detected - review evolution strategies"
    else:
        recommendation = "Low improvement potential - consider alternative approaches"
    
    return {
        "predicted_improvement": predicted_improvement,
        "confidence": confidence,
        "improvement_stability": improvement_stability,
        "historical_average": avg_improvement,
        "recommendation": recommendation,
        "analysis_window": len(recent_improvements)
    }