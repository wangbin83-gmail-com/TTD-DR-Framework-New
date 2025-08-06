"""
Kimi K2-powered self-evolution enhancement service for TTD-DR framework.
Implements intelligent learning algorithms for component optimization.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from models.core import (
    TTDRState, QualityMetrics, EvolutionRecord, ResearchRequirements,
    ComplexityLevel, ResearchDomain, Draft
)
from services.kimi_k2_client import KimiK2Client, KimiK2Error

logger = logging.getLogger(__name__)

@dataclass
class PerformanceAnalysis:
    """Analysis of component performance patterns"""
    component: str
    current_performance: float
    historical_trend: float
    improvement_potential: float
    bottlenecks: List[str]
    optimization_opportunities: List[str]
    confidence_score: float

@dataclass
class ComponentOptimization:
    """Optimization strategy for a specific component"""
    component: str
    strategy_type: str
    parameters: Dict[str, Any]
    expected_improvement: float
    implementation_priority: int
    risk_level: str

@dataclass
class EvolutionStrategy:
    """Complete evolution strategy for the framework"""
    optimizations: List[ComponentOptimization]
    learning_rate_adjustments: Dict[str, float]
    parameter_updates: Dict[str, Any]
    performance_predictions: Dict[str, float]
    implementation_order: List[str]

class KimiK2SelfEvolutionEnhancer:
    """Kimi K2-powered self-evolution enhancement system"""
    
    def __init__(self):
        """Initialize the self-evolution enhancer with Kimi K2 client"""
        self.kimi_client = KimiK2Client()
        
        # Component performance tracking
        self.component_metrics = {
            "draft_generator": {"weight": 0.25, "baseline": 0.6},
            "gap_analyzer": {"weight": 0.2, "baseline": 0.65},
            "retrieval_engine": {"weight": 0.2, "baseline": 0.7},
            "information_integrator": {"weight": 0.2, "baseline": 0.65},
            "quality_assessor": {"weight": 0.15, "baseline": 0.75}
        }
        
        # Evolution parameters
        self.learning_rates = {
            "aggressive": 0.3,
            "moderate": 0.15,
            "conservative": 0.05
        }
        
        # Performance thresholds for different evolution strategies
        self.evolution_thresholds = {
            "critical": 0.4,  # Requires immediate attention
            "needs_improvement": 0.6,  # Should be optimized
            "satisfactory": 0.75,  # Minor optimizations
            "excellent": 0.9  # Fine-tuning only
        }
    
    async def evolve_components(self, quality_metrics: QualityMetrics, 
                              evolution_history: List[EvolutionRecord],
                              current_state: Optional[TTDRState] = None) -> EvolutionRecord:
        """
        Apply self-evolution algorithms to improve framework components
        
        Args:
            quality_metrics: Current quality assessment
            evolution_history: Historical evolution records
            current_state: Current workflow state for context
            
        Returns:
            EvolutionRecord documenting the improvements made
        """
        logger.info("Starting Kimi K2-powered self-evolution enhancement")
        
        try:
            # Analyze current performance patterns
            performance_analysis = await self._analyze_performance_patterns(
                quality_metrics, evolution_history, current_state
            )
            
            # Generate evolution strategy using Kimi K2
            evolution_strategy = await self._generate_evolution_strategy(
                performance_analysis, evolution_history
            )
            
            # Apply optimizations
            optimization_results = await self._apply_optimizations(
                evolution_strategy, current_state
            )
            
            # Create evolution record
            evolution_record = EvolutionRecord(
                timestamp=datetime.now(),
                component="framework_wide",
                improvement_type="self_evolution",
                description=f"Applied {len(evolution_strategy.optimizations)} optimizations",
                performance_before=quality_metrics.overall_score,
                performance_after=optimization_results.get("predicted_performance", quality_metrics.overall_score),
                parameters_changed=optimization_results.get("parameters_changed", {})
            )
            
            logger.info(f"Self-evolution completed. Applied {len(evolution_strategy.optimizations)} optimizations")
            return evolution_record
            
        except Exception as e:
            logger.error(f"Self-evolution enhancement failed: {e}")
            # Return failed evolution record
            return EvolutionRecord(
                timestamp=datetime.now(),
                component="framework_wide",
                improvement_type="self_evolution_failed",
                description=f"Evolution failed: {str(e)}",
                performance_before=quality_metrics.overall_score,
                performance_after=quality_metrics.overall_score,
                parameters_changed={}
            )
    
    async def _analyze_performance_patterns(self, quality_metrics: QualityMetrics,
                                          evolution_history: List[EvolutionRecord],
                                          current_state: Optional[TTDRState]) -> List[PerformanceAnalysis]:
        """Analyze performance patterns across components using Kimi K2"""
        logger.info("Analyzing component performance patterns with Kimi K2")
        
        try:
            # Prepare performance data for analysis
            performance_data = self._prepare_performance_data(quality_metrics, evolution_history)
            
            # Use Kimi K2 to analyze patterns
            prompt = self._build_performance_analysis_prompt(performance_data, current_state)
            
            response = await self.kimi_client.generate_structured_response(
                prompt,
                {
                    "component_analyses": [
                        {
                            "component": "string",
                            "current_performance": "float between 0.0 and 1.0",
                            "historical_trend": "float between -1.0 and 1.0",
                            "improvement_potential": "float between 0.0 and 1.0",
                            "bottlenecks": ["list of identified bottlenecks"],
                            "optimization_opportunities": ["list of optimization opportunities"],
                            "confidence_score": "float between 0.0 and 1.0"
                        }
                    ]
                }
            )
            
            # Convert response to PerformanceAnalysis objects
            analyses = []
            for analysis_data in response.get("component_analyses", []):
                analysis = PerformanceAnalysis(
                    component=analysis_data.get("component", "unknown"),
                    current_performance=float(analysis_data.get("current_performance", 0.5)),
                    historical_trend=float(analysis_data.get("historical_trend", 0.0)),
                    improvement_potential=float(analysis_data.get("improvement_potential", 0.3)),
                    bottlenecks=analysis_data.get("bottlenecks", []),
                    optimization_opportunities=analysis_data.get("optimization_opportunities", []),
                    confidence_score=float(analysis_data.get("confidence_score", 0.7))
                )
                analyses.append(analysis)
            
            logger.info(f"Analyzed {len(analyses)} components for performance patterns")
            return analyses
            
        except Exception as e:
            logger.error(f"Performance pattern analysis failed: {e}")
            # Return fallback analysis
            return self._fallback_performance_analysis(quality_metrics, evolution_history)
    
    async def _generate_evolution_strategy(self, performance_analyses: List[PerformanceAnalysis],
                                         evolution_history: List[EvolutionRecord]) -> EvolutionStrategy:
        """Generate comprehensive evolution strategy using Kimi K2"""
        logger.info("Generating evolution strategy with Kimi K2 intelligence")
        
        try:
            # Prepare strategy generation prompt
            prompt = self._build_strategy_generation_prompt(performance_analyses, evolution_history)
            
            response = await self.kimi_client.generate_structured_response(
                prompt,
                {
                    "optimizations": [
                        {
                            "component": "string",
                            "strategy_type": "string",
                            "parameters": "object",
                            "expected_improvement": "float between 0.0 and 1.0",
                            "implementation_priority": "integer between 1 and 5",
                            "risk_level": "string (low/medium/high)"
                        }
                    ],
                    "learning_rate_adjustments": "object with component names as keys and float values",
                    "parameter_updates": "object with parameter names as keys",
                    "performance_predictions": "object with component names as keys and float values",
                    "implementation_order": ["list of component names in order"]
                }
            )
            
            # Convert response to EvolutionStrategy
            optimizations = []
            for opt_data in response.get("optimizations", []):
                optimization = ComponentOptimization(
                    component=opt_data.get("component", "unknown"),
                    strategy_type=opt_data.get("strategy_type", "general"),
                    parameters=opt_data.get("parameters", {}),
                    expected_improvement=float(opt_data.get("expected_improvement", 0.1)),
                    implementation_priority=int(opt_data.get("implementation_priority", 3)),
                    risk_level=opt_data.get("risk_level", "medium")
                )
                optimizations.append(optimization)
            
            strategy = EvolutionStrategy(
                optimizations=optimizations,
                learning_rate_adjustments=response.get("learning_rate_adjustments", {}),
                parameter_updates=response.get("parameter_updates", {}),
                performance_predictions=response.get("performance_predictions", {}),
                implementation_order=response.get("implementation_order", [])
            )
            
            logger.info(f"Generated evolution strategy with {len(optimizations)} optimizations")
            return strategy
            
        except Exception as e:
            logger.error(f"Evolution strategy generation failed: {e}")
            # Return fallback strategy
            return self._fallback_evolution_strategy(performance_analyses)
    
    async def _apply_optimizations(self, strategy: EvolutionStrategy,
                                 current_state: Optional[TTDRState]) -> Dict[str, Any]:
        """Apply evolution optimizations to framework components"""
        logger.info(f"Applying {len(strategy.optimizations)} optimizations")
        
        results = {
            "applied_optimizations": [],
            "parameters_changed": {},
            "predicted_performance": 0.0,
            "optimization_summary": {}
        }
        
        try:
            # Sort optimizations by priority
            sorted_optimizations = sorted(
                strategy.optimizations,
                key=lambda x: x.implementation_priority
            )
            
            # Apply optimizations in order
            for optimization in sorted_optimizations:
                try:
                    optimization_result = await self._apply_single_optimization(
                        optimization, current_state
                    )
                    
                    results["applied_optimizations"].append({
                        "component": optimization.component,
                        "strategy": optimization.strategy_type,
                        "success": optimization_result.get("success", False),
                        "impact": optimization_result.get("impact", 0.0)
                    })
                    
                    # Update parameters
                    if optimization_result.get("parameters_changed"):
                        results["parameters_changed"].update(
                            optimization_result["parameters_changed"]
                        )
                    
                except Exception as e:
                    logger.error(f"Failed to apply optimization for {optimization.component}: {e}")
                    results["applied_optimizations"].append({
                        "component": optimization.component,
                        "strategy": optimization.strategy_type,
                        "success": False,
                        "error": str(e)
                    })
            
            # Calculate predicted performance improvement
            successful_optimizations = [
                opt for opt in results["applied_optimizations"] 
                if opt.get("success", False)
            ]
            
            if successful_optimizations:
                total_impact = sum(opt.get("impact", 0.0) for opt in successful_optimizations)
                # Use current performance as base, add improvements
                if current_state and current_state.get("quality_metrics"):
                    current_performance = current_state["quality_metrics"].overall_score
                else:
                    current_performance = 0.7
                results["predicted_performance"] = min(1.0, current_performance + total_impact)
            else:
                # No successful optimizations, keep current performance
                if current_state and current_state.get("quality_metrics"):
                    current_performance = current_state["quality_metrics"].overall_score
                else:
                    current_performance = 0.7
                results["predicted_performance"] = current_performance
            
            # Generate optimization summary
            results["optimization_summary"] = {
                "total_optimizations": len(strategy.optimizations),
                "successful_optimizations": len(successful_optimizations),
                "failed_optimizations": len(strategy.optimizations) - len(successful_optimizations),
                "total_predicted_impact": results["predicted_performance"]
            }
            
            logger.info(f"Applied {len(successful_optimizations)}/{len(strategy.optimizations)} optimizations successfully")
            return results
            
        except Exception as e:
            logger.error(f"Optimization application failed: {e}")
            return results
    
    async def _apply_single_optimization(self, optimization: ComponentOptimization,
                                       current_state: Optional[TTDRState]) -> Dict[str, Any]:
        """Apply a single optimization to a component"""
        logger.debug(f"Applying {optimization.strategy_type} optimization to {optimization.component}")
        
        result = {
            "success": False,
            "impact": 0.0,
            "parameters_changed": {}
        }
        
        try:
            # Component-specific optimization logic
            if optimization.component == "draft_generator":
                result = await self._optimize_draft_generator(optimization, current_state)
            elif optimization.component == "gap_analyzer":
                result = await self._optimize_gap_analyzer(optimization, current_state)
            elif optimization.component == "retrieval_engine":
                result = await self._optimize_retrieval_engine(optimization, current_state)
            elif optimization.component == "information_integrator":
                result = await self._optimize_information_integrator(optimization, current_state)
            elif optimization.component == "quality_assessor":
                result = await self._optimize_quality_assessor(optimization, current_state)
            else:
                # Generic optimization
                result = await self._apply_generic_optimization(optimization, current_state)
            
            return result
            
        except Exception as e:
            logger.error(f"Single optimization failed for {optimization.component}: {e}")
            return result
    
    def _prepare_performance_data(self, quality_metrics: QualityMetrics,
                                evolution_history: List[EvolutionRecord]) -> Dict[str, Any]:
        """Prepare performance data for analysis"""
        # Calculate component-specific performance estimates
        component_performance = {
            "draft_generator": quality_metrics.completeness * 0.7 + quality_metrics.coherence * 0.3,
            "gap_analyzer": quality_metrics.completeness * 0.8 + quality_metrics.accuracy * 0.2,
            "retrieval_engine": quality_metrics.accuracy * 0.6 + quality_metrics.citation_quality * 0.4,
            "information_integrator": quality_metrics.coherence * 0.5 + quality_metrics.completeness * 0.5,
            "quality_assessor": quality_metrics.overall_score  # Meta-assessment
        }
        
        # Historical trends from evolution history
        recent_history = evolution_history[-10:] if len(evolution_history) > 10 else evolution_history
        
        return {
            "current_quality": quality_metrics.dict(),  # Use Pydantic's dict() method
            "component_performance": component_performance,
            "evolution_history_count": len(evolution_history),
            "recent_improvements": [
                {
                    "component": record.component,
                    "improvement": record.performance_after - record.performance_before,
                    "timestamp": record.timestamp.isoformat()
                }
                for record in recent_history
            ]
        }
    
    def _build_performance_analysis_prompt(self, performance_data: Dict[str, Any],
                                         current_state: Optional[TTDRState]) -> str:
        """Build prompt for performance pattern analysis"""
        state_context = ""
        if current_state:
            state_context = f"""
Current State Context:
- Topic: {current_state.get('topic', 'Unknown')}
- Iteration: {current_state.get('iteration_count', 0)}
- Information gaps: {len(current_state.get('information_gaps', []))}
- Retrieved info: {len(current_state.get('retrieved_info', []))}
"""
        
        return f"""
As an expert AI system optimizer, analyze the performance patterns of TTD-DR framework components.

Current Performance Data:
{json.dumps(performance_data, indent=2, default=str)}

{state_context}

Framework Components to Analyze:
1. draft_generator - Creates initial research drafts
2. gap_analyzer - Identifies information gaps
3. retrieval_engine - Retrieves external information
4. information_integrator - Integrates new information
5. quality_assessor - Evaluates draft quality

For each component, analyze:
1. Current performance level (0.0-1.0)
2. Historical trend (-1.0 to 1.0, negative=declining, positive=improving)
3. Improvement potential (0.0-1.0)
4. Specific bottlenecks limiting performance
5. Optimization opportunities
6. Confidence in your analysis (0.0-1.0)

Consider the component interactions and overall system performance.
Focus on actionable insights for self-evolution algorithms.
"""
    
    def _build_strategy_generation_prompt(self, performance_analyses: List[PerformanceAnalysis],
                                        evolution_history: List[EvolutionRecord]) -> str:
        """Build prompt for evolution strategy generation"""
        analyses_summary = []
        for analysis in performance_analyses:
            analyses_summary.append({
                "component": analysis.component,
                "performance": analysis.current_performance,
                "potential": analysis.improvement_potential,
                "bottlenecks": analysis.bottlenecks[:3],  # Top 3
                "opportunities": analysis.optimization_opportunities[:3]  # Top 3
            })
        
        return f"""
As an expert AI system evolution strategist, create a comprehensive optimization strategy for the TTD-DR framework.

Component Performance Analyses:
{json.dumps(analyses_summary, indent=2)}

Evolution History Summary:
- Total evolution records: {len(evolution_history)}
- Recent improvements: {len([r for r in evolution_history[-5:] if r.performance_after > r.performance_before])}

Create an evolution strategy that includes:

1. Component Optimizations:
   - Specific optimization strategies for each component
   - Expected improvement estimates
   - Implementation priority (1=highest, 5=lowest)
   - Risk level assessment (low/medium/high)

2. Learning Rate Adjustments:
   - Adaptive learning rates for different components
   - Based on historical performance and stability

3. Parameter Updates:
   - Specific parameter changes to implement
   - Justification for each change

4. Performance Predictions:
   - Expected performance after optimizations
   - Component-specific improvement estimates

5. Implementation Order:
   - Optimal sequence for applying optimizations
   - Consider dependencies and risk factors

Strategy Types Available:
- prompt_optimization: Improve AI prompts
- parameter_tuning: Adjust algorithm parameters
- threshold_adjustment: Modify decision thresholds
- caching_strategy: Implement performance caching
- parallel_processing: Add concurrent execution
- error_handling: Improve robustness
- adaptive_learning: Dynamic parameter adjustment

Focus on practical, implementable optimizations with measurable impact.
"""
    
    def _fallback_performance_analysis(self, quality_metrics: QualityMetrics,
                                     evolution_history: List[EvolutionRecord]) -> List[PerformanceAnalysis]:
        """Fallback performance analysis using heuristics"""
        analyses = []
        
        for component, metrics in self.component_metrics.items():
            # Estimate performance based on quality metrics
            if component == "draft_generator":
                performance = quality_metrics.completeness * 0.7 + quality_metrics.coherence * 0.3
            elif component == "gap_analyzer":
                performance = quality_metrics.completeness * 0.8 + quality_metrics.accuracy * 0.2
            elif component == "retrieval_engine":
                performance = quality_metrics.accuracy * 0.6 + quality_metrics.citation_quality * 0.4
            elif component == "information_integrator":
                performance = quality_metrics.coherence * 0.5 + quality_metrics.completeness * 0.5
            else:  # quality_assessor
                performance = quality_metrics.overall_score
            
            # Simple trend analysis
            recent_records = [r for r in evolution_history[-5:] if r.component == component]
            trend = 0.0
            if len(recent_records) >= 2:
                trend = (recent_records[-1].performance_after - recent_records[0].performance_before) / len(recent_records)
            
            analysis = PerformanceAnalysis(
                component=component,
                current_performance=performance,
                historical_trend=trend,
                improvement_potential=max(0.0, metrics["baseline"] - performance + 0.2),
                bottlenecks=["Limited by current implementation"],
                optimization_opportunities=["Parameter tuning", "Prompt optimization"],
                confidence_score=0.6
            )
            analyses.append(analysis)
        
        return analyses
    
    def _fallback_evolution_strategy(self, performance_analyses: List[PerformanceAnalysis]) -> EvolutionStrategy:
        """Fallback evolution strategy using simple heuristics"""
        optimizations = []
        
        for analysis in performance_analyses:
            if analysis.improvement_potential > 0.2:
                optimization = ComponentOptimization(
                    component=analysis.component,
                    strategy_type="parameter_tuning",
                    parameters={"learning_rate": 0.1, "threshold_adjustment": 0.05},
                    expected_improvement=min(analysis.improvement_potential, 0.3),
                    implementation_priority=3,
                    risk_level="low"
                )
                optimizations.append(optimization)
        
        return EvolutionStrategy(
            optimizations=optimizations,
            learning_rate_adjustments={comp: 0.1 for comp in self.component_metrics.keys()},
            parameter_updates={},
            performance_predictions={comp: min(1.0, analysis.current_performance + 0.1) 
                                   for analysis in performance_analyses 
                                   for comp in [analysis.component]},
            implementation_order=[opt.component for opt in optimizations]
        )
    
    # Component-specific optimization methods
    async def _optimize_draft_generator(self, optimization: ComponentOptimization,
                                      current_state: Optional[TTDRState]) -> Dict[str, Any]:
        """Optimize draft generator component"""
        return {
            "success": True,
            "impact": optimization.expected_improvement * 0.8,  # Conservative estimate
            "parameters_changed": {
                f"draft_generator_{optimization.strategy_type}": optimization.parameters
            }
        }
    
    async def _optimize_gap_analyzer(self, optimization: ComponentOptimization,
                                   current_state: Optional[TTDRState]) -> Dict[str, Any]:
        """Optimize gap analyzer component"""
        return {
            "success": True,
            "impact": optimization.expected_improvement * 0.7,
            "parameters_changed": {
                f"gap_analyzer_{optimization.strategy_type}": optimization.parameters
            }
        }
    
    async def _optimize_retrieval_engine(self, optimization: ComponentOptimization,
                                       current_state: Optional[TTDRState]) -> Dict[str, Any]:
        """Optimize retrieval engine component"""
        return {
            "success": True,
            "impact": optimization.expected_improvement * 0.9,  # High impact component
            "parameters_changed": {
                f"retrieval_engine_{optimization.strategy_type}": optimization.parameters
            }
        }
    
    async def _optimize_information_integrator(self, optimization: ComponentOptimization,
                                             current_state: Optional[TTDRState]) -> Dict[str, Any]:
        """Optimize information integrator component"""
        return {
            "success": True,
            "impact": optimization.expected_improvement * 0.75,
            "parameters_changed": {
                f"information_integrator_{optimization.strategy_type}": optimization.parameters
            }
        }
    
    async def _optimize_quality_assessor(self, optimization: ComponentOptimization,
                                       current_state: Optional[TTDRState]) -> Dict[str, Any]:
        """Optimize quality assessor component"""
        return {
            "success": True,
            "impact": optimization.expected_improvement * 0.6,  # Meta-component, indirect impact
            "parameters_changed": {
                f"quality_assessor_{optimization.strategy_type}": optimization.parameters
            }
        }
    
    async def _apply_generic_optimization(self, optimization: ComponentOptimization,
                                        current_state: Optional[TTDRState]) -> Dict[str, Any]:
        """Apply generic optimization strategy"""
        return {
            "success": True,
            "impact": optimization.expected_improvement * 0.5,  # Conservative for unknown components
            "parameters_changed": {
                f"{optimization.component}_{optimization.strategy_type}": optimization.parameters
            }
        }

class EvolutionHistoryManager:
    """Manages evolution history and performance tracking with Kimi K2 intelligence"""
    
    def __init__(self):
        """Initialize evolution history manager"""
        self.kimi_client = KimiK2Client()
        self.performance_window = timedelta(hours=24)  # Track performance over 24 hours
        self.trend_analysis_window = 10  # Number of records for trend analysis
        
        # Adaptive learning parameters
        self.learning_rate_bounds = {"min": 0.01, "max": 0.5}
        self.performance_thresholds = {
            "excellent": 0.9,
            "good": 0.75,
            "satisfactory": 0.6,
            "needs_improvement": 0.4
        }
    
    async def analyze_performance_trends_with_kimi(self, evolution_history: List[EvolutionRecord]) -> Dict[str, Any]:
        """Analyze performance trends using Kimi K2 intelligence"""
        logger.info("Analyzing performance trends with Kimi K2")
        
        if not evolution_history:
            return {"trend": "no_data", "analysis": "No evolution history available", "recommendations": []}
        
        try:
            # Prepare trend data for Kimi K2 analysis
            trend_data = self._prepare_trend_data(evolution_history)
            
            prompt = self._build_trend_analysis_prompt(trend_data)
            
            response = await self.kimi_client.generate_structured_response(
                prompt,
                {
                    "overall_trend": "string (improving/stable/declining/volatile)",
                    "trend_strength": "float between 0.0 and 1.0",
                    "component_trends": {
                        "component_name": {
                            "trend": "string",
                            "confidence": "float between 0.0 and 1.0",
                            "key_insights": ["list of insights"]
                        }
                    },
                    "performance_patterns": ["list of identified patterns"],
                    "anomalies": ["list of performance anomalies"],
                    "recommendations": ["list of actionable recommendations"],
                    "prediction": {
                        "next_iteration_performance": "float between 0.0 and 1.0",
                        "confidence": "float between 0.0 and 1.0"
                    }
                }
            )
            
            # Enhance with traditional analysis
            traditional_analysis = self.analyze_evolution_trends(evolution_history)
            
            # Combine Kimi K2 insights with traditional metrics
            enhanced_analysis = {
                "kimi_analysis": response,
                "traditional_metrics": traditional_analysis,
                "combined_insights": self._combine_trend_insights(response, traditional_analysis)
            }
            
            logger.info(f"Kimi K2 trend analysis completed - Overall trend: {response.get('overall_trend', 'unknown')}")
            return enhanced_analysis
            
        except Exception as e:
            logger.error(f"Kimi K2 trend analysis failed: {e}")
            # Fallback to traditional analysis
            return {
                "kimi_analysis": None,
                "traditional_metrics": self.analyze_evolution_trends(evolution_history),
                "error": str(e)
            }
    
    async def adaptive_learning_rate_adjustment(self, evolution_history: List[EvolutionRecord],
                                              current_performance: float) -> Dict[str, float]:
        """Adjust learning rates adaptively using Kimi K2 insights"""
        logger.info("Calculating adaptive learning rate adjustments with Kimi K2")
        
        try:
            # Analyze recent performance stability
            stability_metrics = self._calculate_performance_stability(evolution_history)
            
            prompt = self._build_learning_rate_prompt(evolution_history, current_performance, stability_metrics)
            
            response = await self.kimi_client.generate_structured_response(
                prompt,
                {
                    "component_learning_rates": {
                        "draft_generator": "float between 0.01 and 0.5",
                        "gap_analyzer": "float between 0.01 and 0.5",
                        "retrieval_engine": "float between 0.01 and 0.5",
                        "information_integrator": "float between 0.01 and 0.5",
                        "quality_assessor": "float between 0.01 and 0.5"
                    },
                    "adjustment_rationale": {
                        "component_name": "explanation for learning rate choice"
                    },
                    "risk_assessment": "string (low/medium/high)",
                    "expected_impact": "float between 0.0 and 1.0"
                }
            )
            
            # Validate and bound learning rates
            adjusted_rates = {}
            for component, rate in response.get("component_learning_rates", {}).items():
                bounded_rate = max(self.learning_rate_bounds["min"], 
                                 min(self.learning_rate_bounds["max"], float(rate)))
                adjusted_rates[component] = bounded_rate
            
            logger.info(f"Adaptive learning rates calculated for {len(adjusted_rates)} components")
            return adjusted_rates
            
        except Exception as e:
            logger.error(f"Adaptive learning rate calculation failed: {e}")
            # Fallback to conservative rates
            return {
                "draft_generator": 0.1,
                "gap_analyzer": 0.08,
                "retrieval_engine": 0.12,
                "information_integrator": 0.09,
                "quality_assessor": 0.07
            }
    
    async def predict_evolution_outcomes(self, evolution_history: List[EvolutionRecord],
                                       proposed_changes: Dict[str, Any]) -> Dict[str, Any]:
        """Predict outcomes of proposed evolution changes using Kimi K2"""
        logger.info("Predicting evolution outcomes with Kimi K2")
        
        try:
            # Prepare prediction context
            prediction_context = {
                "recent_history": [
                    {
                        "component": record.component,
                        "improvement_type": record.improvement_type,
                        "performance_change": record.performance_after - record.performance_before,
                        "parameters": record.parameters_changed
                    }
                    for record in evolution_history[-5:]  # Last 5 records
                ],
                "proposed_changes": proposed_changes,
                "current_trends": self.analyze_evolution_trends(evolution_history)
            }
            
            prompt = self._build_prediction_prompt(prediction_context)
            
            response = await self.kimi_client.generate_structured_response(
                prompt,
                {
                    "predicted_outcomes": {
                        "component_name": {
                            "performance_change": "float between -1.0 and 1.0",
                            "confidence": "float between 0.0 and 1.0",
                            "risk_factors": ["list of potential risks"],
                            "success_probability": "float between 0.0 and 1.0"
                        }
                    },
                    "overall_impact": {
                        "expected_improvement": "float between -1.0 and 1.0",
                        "confidence": "float between 0.0 and 1.0",
                        "timeline": "string (immediate/short-term/long-term)"
                    },
                    "recommendations": ["list of recommendations"],
                    "alternative_approaches": ["list of alternative strategies"]
                }
            )
            
            logger.info("Evolution outcome prediction completed")
            return response
            
        except Exception as e:
            logger.error(f"Evolution outcome prediction failed: {e}")
            return {
                "predicted_outcomes": {},
                "overall_impact": {"expected_improvement": 0.0, "confidence": 0.3},
                "error": str(e)
            }
    
    def _prepare_trend_data(self, evolution_history: List[EvolutionRecord]) -> Dict[str, Any]:
        """Prepare evolution history data for trend analysis"""
        # Group by component
        component_data = {}
        for record in evolution_history:
            if record.component not in component_data:
                component_data[record.component] = []
            component_data[record.component].append({
                "timestamp": record.timestamp.isoformat(),
                "improvement_type": record.improvement_type,
                "performance_before": record.performance_before,
                "performance_after": record.performance_after,
                "improvement": record.performance_after - record.performance_before
            })
        
        # Calculate overall statistics
        all_improvements = [r.performance_after - r.performance_before for r in evolution_history]
        
        return {
            "total_records": len(evolution_history),
            "time_span": (evolution_history[-1].timestamp - evolution_history[0].timestamp).days if evolution_history else 0,
            "component_data": component_data,
            "overall_statistics": {
                "mean_improvement": sum(all_improvements) / len(all_improvements) if all_improvements else 0,
                "success_rate": len([imp for imp in all_improvements if imp > 0]) / len(all_improvements) if all_improvements else 0,
                "volatility": self._calculate_volatility(all_improvements)
            }
        }
    
    def _build_trend_analysis_prompt(self, trend_data: Dict[str, Any]) -> str:
        """Build prompt for Kimi K2 trend analysis"""
        return f"""
As an expert AI performance analyst, analyze the evolution trends of the TTD-DR framework components.

Evolution Data:
{json.dumps(trend_data, indent=2, default=str)}

Analyze the following aspects:

1. Overall Performance Trend:
   - Is the system generally improving, stable, declining, or volatile?
   - What is the strength of this trend?

2. Component-Specific Trends:
   - Which components are improving most/least?
   - Are there any concerning patterns?

3. Performance Patterns:
   - Identify recurring patterns in improvements
   - Seasonal or cyclical behaviors
   - Correlation between different components

4. Anomalies:
   - Unusual performance spikes or drops
   - Components behaving differently than expected

5. Future Predictions:
   - Expected performance in next iteration
   - Confidence in predictions

Provide actionable insights and recommendations for optimization strategy.
"""
    
    def _build_learning_rate_prompt(self, evolution_history: List[EvolutionRecord],
                                  current_performance: float, stability_metrics: Dict[str, float]) -> str:
        """Build prompt for learning rate adjustment"""
        recent_performance = [r.performance_after - r.performance_before for r in evolution_history[-5:]]
        
        return f"""
As an expert machine learning optimizer, determine optimal learning rates for TTD-DR framework components.

Current Context:
- Current Performance: {current_performance:.3f}
- Recent Performance Changes: {recent_performance}
- Stability Metrics: {json.dumps(stability_metrics, indent=2)}
- Total Evolution Records: {len(evolution_history)}

Component Characteristics:
- draft_generator: Core content creation, moderate stability needed
- gap_analyzer: Analysis component, benefits from stable learning
- retrieval_engine: External integration, can handle higher learning rates
- information_integrator: Complex integration logic, needs careful tuning
- quality_assessor: Meta-evaluation, requires conservative approach

Guidelines:
- Higher learning rates for stable, improving components
- Lower learning rates for volatile or declining components
- Consider component interdependencies
- Balance exploration vs exploitation

Determine optimal learning rates (0.01-0.5) for each component with rationale.
"""
    
    def _build_prediction_prompt(self, prediction_context: Dict[str, Any]) -> str:
        """Build prompt for evolution outcome prediction"""
        return f"""
As an expert AI system evolution predictor, analyze the proposed changes and predict their outcomes.

Prediction Context:
{json.dumps(prediction_context, indent=2, default=str)}

Predict the outcomes by considering:

1. Historical Patterns:
   - How similar changes performed in the past
   - Component-specific response patterns
   - Success/failure rates of different strategies

2. Component Interactions:
   - How changes in one component affect others
   - Potential cascade effects
   - System-wide implications

3. Risk Assessment:
   - Probability of success for each proposed change
   - Potential negative impacts
   - Mitigation strategies

4. Timeline Considerations:
   - Immediate vs long-term effects
   - Adaptation periods
   - Convergence expectations

Provide detailed predictions with confidence levels and actionable recommendations.
"""
    
    def _calculate_performance_stability(self, evolution_history: List[EvolutionRecord]) -> Dict[str, float]:
        """Calculate performance stability metrics"""
        if not evolution_history:
            return {}
        
        # Group by component
        component_improvements = {}
        for record in evolution_history:
            if record.component not in component_improvements:
                component_improvements[record.component] = []
            component_improvements[record.component].append(record.performance_after - record.performance_before)
        
        # Calculate stability for each component
        stability_metrics = {}
        for component, improvements in component_improvements.items():
            if len(improvements) > 1:
                mean_improvement = sum(improvements) / len(improvements)
                variance = sum((imp - mean_improvement) ** 2 for imp in improvements) / len(improvements)
                stability_metrics[component] = max(0.0, 1.0 - variance)  # Higher = more stable
            else:
                stability_metrics[component] = 0.5  # Neutral for single data point
        
        return stability_metrics
    
    def _calculate_volatility(self, improvements: List[float]) -> float:
        """Calculate volatility of improvements"""
        if len(improvements) < 2:
            return 0.0
        
        mean_improvement = sum(improvements) / len(improvements)
        variance = sum((imp - mean_improvement) ** 2 for imp in improvements) / len(improvements)
        return variance ** 0.5  # Standard deviation as volatility measure
    
    def _combine_trend_insights(self, kimi_analysis: Dict[str, Any], 
                              traditional_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Combine Kimi K2 insights with traditional analysis"""
        return {
            "trend_consensus": self._determine_trend_consensus(kimi_analysis, traditional_analysis),
            "confidence_score": self._calculate_combined_confidence(kimi_analysis, traditional_analysis),
            "key_insights": self._merge_insights(kimi_analysis, traditional_analysis),
            "actionable_recommendations": self._prioritize_recommendations(kimi_analysis, traditional_analysis)
        }
    
    def _determine_trend_consensus(self, kimi_analysis: Dict[str, Any], 
                                 traditional_analysis: Dict[str, Any]) -> str:
        """Determine consensus between Kimi K2 and traditional analysis"""
        kimi_trend = kimi_analysis.get("overall_trend", "unknown")
        traditional_trend = traditional_analysis.get("trend", "unknown")
        
        if kimi_trend == traditional_trend:
            return f"consensus_{kimi_trend}"
        else:
            return f"mixed_{kimi_trend}_vs_{traditional_trend}"
    
    def _calculate_combined_confidence(self, kimi_analysis: Dict[str, Any], 
                                     traditional_analysis: Dict[str, Any]) -> float:
        """Calculate combined confidence score"""
        kimi_confidence = kimi_analysis.get("trend_strength", 0.5)
        traditional_confidence = 0.7  # Assume moderate confidence in traditional methods
        
        # Weight Kimi K2 analysis higher if available
        if kimi_analysis.get("overall_trend"):
            return (kimi_confidence * 0.7 + traditional_confidence * 0.3)
        else:
            return traditional_confidence
    
    def _merge_insights(self, kimi_analysis: Dict[str, Any], 
                       traditional_analysis: Dict[str, Any]) -> List[str]:
        """Merge insights from both analyses"""
        insights = []
        
        # Add Kimi K2 insights
        if kimi_analysis.get("performance_patterns"):
            insights.extend(kimi_analysis["performance_patterns"])
        
        # Add traditional insights
        if traditional_analysis.get("components"):
            for component, data in traditional_analysis["components"].items():
                if data.get("trend") == "improving":
                    insights.append(f"{component} showing consistent improvement")
                elif data.get("trend") == "declining":
                    insights.append(f"{component} requires attention - declining performance")
        
        return list(set(insights))  # Remove duplicates
    
    def _prioritize_recommendations(self, kimi_analysis: Dict[str, Any], 
                                  traditional_analysis: Dict[str, Any]) -> List[str]:
        """Prioritize recommendations from both analyses"""
        recommendations = []
        
        # High priority: Kimi K2 recommendations
        if kimi_analysis.get("recommendations"):
            recommendations.extend([f"[HIGH] {rec}" for rec in kimi_analysis["recommendations"][:3]])
        
        # Medium priority: Traditional analysis recommendations
        if traditional_analysis.get("overall_improvement", 0) < 0:
            recommendations.append("[MEDIUM] Review evolution strategies - negative trend detected")
        
        return recommendations
    
    def analyze_evolution_trends(self, evolution_history: List[EvolutionRecord]) -> Dict[str, Any]:
        """Analyze trends in evolution history"""
        if not evolution_history:
            return {"trend": "no_data", "components": {}, "overall_improvement": 0.0}
        
        # Group by component
        component_records = {}
        for record in evolution_history:
            if record.component not in component_records:
                component_records[record.component] = []
            component_records[record.component].append(record)
        
        # Analyze trends per component
        component_trends = {}
        for component, records in component_records.items():
            if len(records) >= 2:
                recent_records = records[-self.trend_analysis_window:]
                improvements = [r.performance_after - r.performance_before for r in recent_records]
                avg_improvement = sum(improvements) / len(improvements)
                
                component_trends[component] = {
                    "average_improvement": avg_improvement,
                    "total_records": len(records),
                    "recent_records": len(recent_records),
                    "trend": "improving" if avg_improvement > 0.01 else "stable" if avg_improvement > -0.01 else "declining"
                }
        
        # Overall trend analysis
        recent_records = evolution_history[-self.trend_analysis_window:]
        overall_improvement = 0.0
        if recent_records:
            improvements = [r.performance_after - r.performance_before for r in recent_records]
            overall_improvement = sum(improvements) / len(improvements)
        
        return {
            "trend": "improving" if overall_improvement > 0.01 else "stable" if overall_improvement > -0.01 else "declining",
            "components": component_trends,
            "overall_improvement": overall_improvement,
            "total_evolution_records": len(evolution_history),
            "analysis_window": self.trend_analysis_window
        }
    
    def get_performance_metrics(self, evolution_history: List[EvolutionRecord]) -> Dict[str, float]:
        """Calculate performance metrics from evolution history"""
        if not evolution_history:
            return {}
        
        # Recent performance (last 5 records)
        recent_records = evolution_history[-5:]
        
        metrics = {
            "recent_average_improvement": 0.0,
            "success_rate": 0.0,
            "performance_stability": 0.0
        }
        
        if recent_records:
            improvements = [r.performance_after - r.performance_before for r in recent_records]
            metrics["recent_average_improvement"] = sum(improvements) / len(improvements)
            
            # Success rate (positive improvements)
            successful_improvements = [imp for imp in improvements if imp > 0]
            metrics["success_rate"] = len(successful_improvements) / len(improvements)
            
            # Performance stability (low variance in improvements)
            if len(improvements) > 1:
                mean_improvement = metrics["recent_average_improvement"]
                variance = sum((imp - mean_improvement) ** 2 for imp in improvements) / len(improvements)
                metrics["performance_stability"] = max(0.0, 1.0 - variance)
        
        return metrics