"""
Unit tests for Kimi K2 self-evolution enhancer service.
Tests intelligent learning algorithms and component optimization.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import List, Dict, Any

from backend.services.kimi_k2_self_evolution_enhancer import (
    KimiK2SelfEvolutionEnhancer, EvolutionHistoryManager,
    PerformanceAnalysis, ComponentOptimization, EvolutionStrategy
)
from backend.models.core import (
    QualityMetrics, EvolutionRecord, TTDRState, ResearchRequirements,
    ComplexityLevel, ResearchDomain, Draft, ResearchStructure, Section
)

class TestKimiK2SelfEvolutionEnhancer:
    """Test cases for KimiK2SelfEvolutionEnhancer"""
    
    @pytest.fixture
    def mock_kimi_client(self):
        """Mock Kimi K2 client for testing"""
        mock_client = Mock()
        mock_client.generate_structured_response = AsyncMock()
        return mock_client
    
    @pytest.fixture
    def evolution_enhancer(self, mock_kimi_client):
        """Create evolution enhancer with mocked client"""
        enhancer = KimiK2SelfEvolutionEnhancer()
        enhancer.kimi_client = mock_kimi_client
        return enhancer
    
    @pytest.fixture
    def sample_quality_metrics(self):
        """Sample quality metrics for testing"""
        return QualityMetrics(
            completeness=0.7,
            coherence=0.8,
            accuracy=0.6,
            citation_quality=0.5,
            overall_score=0.675
        )
    
    @pytest.fixture
    def sample_evolution_history(self):
        """Sample evolution history for testing"""
        return [
            EvolutionRecord(
                timestamp=datetime.now() - timedelta(hours=2),
                component="draft_generator",
                improvement_type="prompt_optimization",
                description="Improved draft generation prompts",
                performance_before=0.6,
                performance_after=0.65,
                parameters_changed={"prompt_version": "v2.1"}
            ),
            EvolutionRecord(
                timestamp=datetime.now() - timedelta(hours=1),
                component="gap_analyzer",
                improvement_type="parameter_tuning",
                description="Adjusted gap detection thresholds",
                performance_before=0.55,
                performance_after=0.62,
                parameters_changed={"threshold": 0.7}
            )
        ]
    
    @pytest.fixture
    def sample_ttdr_state(self, sample_quality_metrics, sample_evolution_history):
        """Sample TTD-R state for testing"""
        requirements = ResearchRequirements(
            domain=ResearchDomain.TECHNOLOGY,
            complexity_level=ComplexityLevel.INTERMEDIATE,
            max_iterations=5,
            quality_threshold=0.8
        )
        
        return {
            "topic": "Artificial Intelligence in Healthcare",
            "requirements": requirements,
            "current_draft": None,
            "information_gaps": [],
            "retrieved_info": [],
            "iteration_count": 2,
            "quality_metrics": sample_quality_metrics,
            "evolution_history": sample_evolution_history,
            "final_report": None,
            "error_log": []
        }
    
    @pytest.mark.asyncio
    async def test_evolve_components_success(self, evolution_enhancer, mock_kimi_client,
                                           sample_quality_metrics, sample_evolution_history):
        """Test successful component evolution"""
        # Mock performance analysis response
        mock_kimi_client.generate_structured_response.side_effect = [
            # Performance analysis response
            {
                "component_analyses": [
                    {
                        "component": "draft_generator",
                        "current_performance": 0.7,
                        "historical_trend": 0.1,
                        "improvement_potential": 0.3,
                        "bottlenecks": ["Limited prompt diversity"],
                        "optimization_opportunities": ["Prompt optimization", "Parameter tuning"],
                        "confidence_score": 0.8
                    },
                    {
                        "component": "gap_analyzer",
                        "current_performance": 0.6,
                        "historical_trend": 0.05,
                        "improvement_potential": 0.4,
                        "bottlenecks": ["Threshold sensitivity"],
                        "optimization_opportunities": ["Threshold adjustment", "Algorithm improvement"],
                        "confidence_score": 0.75
                    }
                ]
            },
            # Evolution strategy response
            {
                "optimizations": [
                    {
                        "component": "draft_generator",
                        "strategy_type": "prompt_optimization",
                        "parameters": {"prompt_version": "v3.0", "temperature": 0.7},
                        "expected_improvement": 0.15,
                        "implementation_priority": 1,
                        "risk_level": "low"
                    },
                    {
                        "component": "gap_analyzer",
                        "strategy_type": "parameter_tuning",
                        "parameters": {"threshold": 0.65, "sensitivity": 0.8},
                        "expected_improvement": 0.12,
                        "implementation_priority": 2,
                        "risk_level": "medium"
                    }
                ],
                "learning_rate_adjustments": {
                    "draft_generator": 0.1,
                    "gap_analyzer": 0.08
                },
                "parameter_updates": {
                    "global_learning_rate": 0.05
                },
                "performance_predictions": {
                    "draft_generator": 0.85,
                    "gap_analyzer": 0.72
                },
                "implementation_order": ["draft_generator", "gap_analyzer"]
            }
        ]
        
        # Execute evolution
        result = await evolution_enhancer.evolve_components(
            sample_quality_metrics, sample_evolution_history
        )
        
        # Verify result
        assert isinstance(result, EvolutionRecord)
        assert result.component == "framework_wide"
        assert result.improvement_type == "self_evolution"
        assert result.performance_before == sample_quality_metrics.overall_score
        assert result.performance_after >= result.performance_before
        assert len(result.parameters_changed) > 0
        
        # Verify Kimi K2 client was called correctly
        assert mock_kimi_client.generate_structured_response.call_count == 2
    
    @pytest.mark.asyncio
    async def test_evolve_components_kimi_failure(self, evolution_enhancer, mock_kimi_client,
                                                sample_quality_metrics, sample_evolution_history):
        """Test evolution with Kimi K2 API failure"""
        # Mock API failure
        mock_kimi_client.generate_structured_response.side_effect = Exception("API Error")
        
        # Execute evolution
        result = await evolution_enhancer.evolve_components(
            sample_quality_metrics, sample_evolution_history
        )
        
        # Verify fallback behavior - system should still work with fallback logic
        assert isinstance(result, EvolutionRecord)
        assert result.component == "framework_wide"
        assert result.improvement_type == "self_evolution"  # Uses fallback, so still succeeds
        assert result.performance_before == sample_quality_metrics.overall_score
        # Performance after should be same or slightly better due to fallback optimizations
        assert result.performance_after >= result.performance_before
    
    @pytest.mark.asyncio
    async def test_analyze_performance_patterns(self, evolution_enhancer, mock_kimi_client,
                                              sample_quality_metrics, sample_evolution_history):
        """Test performance pattern analysis"""
        # Mock analysis response
        mock_kimi_client.generate_structured_response.return_value = {
            "component_analyses": [
                {
                    "component": "draft_generator",
                    "current_performance": 0.75,
                    "historical_trend": 0.1,
                    "improvement_potential": 0.25,
                    "bottlenecks": ["Prompt limitations", "Context window"],
                    "optimization_opportunities": ["Better prompts", "Chunking strategy"],
                    "confidence_score": 0.85
                }
            ]
        }
        
        # Execute analysis
        analyses = await evolution_enhancer._analyze_performance_patterns(
            sample_quality_metrics, sample_evolution_history, None
        )
        
        # Verify results
        assert len(analyses) == 1
        analysis = analyses[0]
        assert isinstance(analysis, PerformanceAnalysis)
        assert analysis.component == "draft_generator"
        assert 0.0 <= analysis.current_performance <= 1.0
        assert -1.0 <= analysis.historical_trend <= 1.0
        assert 0.0 <= analysis.improvement_potential <= 1.0
        assert len(analysis.bottlenecks) > 0
        assert len(analysis.optimization_opportunities) > 0
        assert 0.0 <= analysis.confidence_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_generate_evolution_strategy(self, evolution_enhancer, mock_kimi_client,
                                             sample_evolution_history):
        """Test evolution strategy generation"""
        # Sample performance analyses
        performance_analyses = [
            PerformanceAnalysis(
                component="draft_generator",
                current_performance=0.7,
                historical_trend=0.05,
                improvement_potential=0.3,
                bottlenecks=["Limited prompts"],
                optimization_opportunities=["Prompt optimization"],
                confidence_score=0.8
            )
        ]
        
        # Mock strategy response
        mock_kimi_client.generate_structured_response.return_value = {
            "optimizations": [
                {
                    "component": "draft_generator",
                    "strategy_type": "prompt_optimization",
                    "parameters": {"version": "v2.0"},
                    "expected_improvement": 0.2,
                    "implementation_priority": 1,
                    "risk_level": "low"
                }
            ],
            "learning_rate_adjustments": {"draft_generator": 0.1},
            "parameter_updates": {"global_param": "value"},
            "performance_predictions": {"draft_generator": 0.9},
            "implementation_order": ["draft_generator"]
        }
        
        # Execute strategy generation
        strategy = await evolution_enhancer._generate_evolution_strategy(
            performance_analyses, sample_evolution_history
        )
        
        # Verify results
        assert isinstance(strategy, EvolutionStrategy)
        assert len(strategy.optimizations) == 1
        
        optimization = strategy.optimizations[0]
        assert isinstance(optimization, ComponentOptimization)
        assert optimization.component == "draft_generator"
        assert optimization.strategy_type == "prompt_optimization"
        assert 0.0 <= optimization.expected_improvement <= 1.0
        assert 1 <= optimization.implementation_priority <= 5
        assert optimization.risk_level in ["low", "medium", "high"]
        
        assert len(strategy.learning_rate_adjustments) > 0
        assert len(strategy.implementation_order) > 0
    
    @pytest.mark.asyncio
    async def test_apply_optimizations(self, evolution_enhancer, sample_ttdr_state):
        """Test optimization application"""
        # Sample evolution strategy
        strategy = EvolutionStrategy(
            optimizations=[
                ComponentOptimization(
                    component="draft_generator",
                    strategy_type="prompt_optimization",
                    parameters={"version": "v2.0"},
                    expected_improvement=0.15,
                    implementation_priority=1,
                    risk_level="low"
                ),
                ComponentOptimization(
                    component="gap_analyzer",
                    strategy_type="parameter_tuning",
                    parameters={"threshold": 0.7},
                    expected_improvement=0.1,
                    implementation_priority=2,
                    risk_level="medium"
                )
            ],
            learning_rate_adjustments={},
            parameter_updates={},
            performance_predictions={},
            implementation_order=["draft_generator", "gap_analyzer"]
        )
        
        # Execute optimization application
        results = await evolution_enhancer._apply_optimizations(strategy, sample_ttdr_state)
        
        # Verify results
        assert "applied_optimizations" in results
        assert "parameters_changed" in results
        assert "predicted_performance" in results
        assert "optimization_summary" in results
        
        applied_opts = results["applied_optimizations"]
        assert len(applied_opts) == 2
        
        # Check optimization results
        for opt_result in applied_opts:
            assert "component" in opt_result
            assert "strategy" in opt_result
            assert "success" in opt_result
            assert opt_result["success"] is True  # Should succeed in test
        
        # Check summary
        summary = results["optimization_summary"]
        assert summary["total_optimizations"] == 2
        assert summary["successful_optimizations"] >= 0
        assert summary["failed_optimizations"] >= 0
    
    def test_prepare_performance_data(self, evolution_enhancer, sample_quality_metrics,
                                    sample_evolution_history):
        """Test performance data preparation"""
        data = evolution_enhancer._prepare_performance_data(
            sample_quality_metrics, sample_evolution_history
        )
        
        # Verify data structure
        assert "current_quality" in data
        assert "component_performance" in data
        assert "evolution_history_count" in data
        assert "recent_improvements" in data
        
        # Verify quality data
        quality_data = data["current_quality"]
        assert quality_data["overall_score"] == sample_quality_metrics.overall_score
        assert quality_data["completeness"] == sample_quality_metrics.completeness
        
        # Verify component performance
        component_perf = data["component_performance"]
        assert len(component_perf) == 5  # All framework components
        for component, performance in component_perf.items():
            assert 0.0 <= performance <= 1.0
        
        # Verify history data
        assert data["evolution_history_count"] == len(sample_evolution_history)
        assert len(data["recent_improvements"]) <= len(sample_evolution_history)
    
    def test_fallback_performance_analysis(self, evolution_enhancer, sample_quality_metrics,
                                         sample_evolution_history):
        """Test fallback performance analysis"""
        analyses = evolution_enhancer._fallback_performance_analysis(
            sample_quality_metrics, sample_evolution_history
        )
        
        # Verify fallback results
        assert len(analyses) == 5  # All framework components
        
        for analysis in analyses:
            assert isinstance(analysis, PerformanceAnalysis)
            assert analysis.component in evolution_enhancer.component_metrics
            assert 0.0 <= analysis.current_performance <= 1.0
            assert -1.0 <= analysis.historical_trend <= 1.0
            assert 0.0 <= analysis.improvement_potential <= 1.0
            assert len(analysis.bottlenecks) > 0
            assert len(analysis.optimization_opportunities) > 0
            assert 0.0 <= analysis.confidence_score <= 1.0
    
    def test_fallback_evolution_strategy(self, evolution_enhancer):
        """Test fallback evolution strategy"""
        # Sample performance analyses
        performance_analyses = [
            PerformanceAnalysis(
                component="draft_generator",
                current_performance=0.6,
                historical_trend=0.0,
                improvement_potential=0.3,
                bottlenecks=["Test bottleneck"],
                optimization_opportunities=["Test opportunity"],
                confidence_score=0.7
            ),
            PerformanceAnalysis(
                component="gap_analyzer",
                current_performance=0.5,
                historical_trend=-0.1,
                improvement_potential=0.4,
                bottlenecks=["Another bottleneck"],
                optimization_opportunities=["Another opportunity"],
                confidence_score=0.6
            )
        ]
        
        strategy = evolution_enhancer._fallback_evolution_strategy(performance_analyses)
        
        # Verify fallback strategy
        assert isinstance(strategy, EvolutionStrategy)
        assert len(strategy.optimizations) == 2  # Both have improvement potential > 0.2
        
        for optimization in strategy.optimizations:
            assert isinstance(optimization, ComponentOptimization)
            assert optimization.strategy_type == "parameter_tuning"
            assert optimization.risk_level == "low"
            assert optimization.implementation_priority == 3
        
        assert len(strategy.learning_rate_adjustments) == 5  # All components
        assert len(strategy.implementation_order) == len(strategy.optimizations)

class TestEvolutionHistoryManager:
    """Test cases for EvolutionHistoryManager"""
    
    @pytest.fixture
    def history_manager(self):
        """Create evolution history manager"""
        return EvolutionHistoryManager()
    
    @pytest.fixture
    def mock_kimi_client_for_history(self):
        """Mock Kimi K2 client for history manager testing"""
        mock_client = Mock()
        mock_client.generate_structured_response = AsyncMock()
        return mock_client
    
    @pytest.fixture
    def sample_evolution_records(self):
        """Sample evolution records for testing"""
        base_time = datetime.now()
        return [
            EvolutionRecord(
                timestamp=base_time - timedelta(hours=5),
                component="draft_generator",
                improvement_type="prompt_optimization",
                description="First optimization",
                performance_before=0.5,
                performance_after=0.6,
                parameters_changed={}
            ),
            EvolutionRecord(
                timestamp=base_time - timedelta(hours=4),
                component="gap_analyzer",
                improvement_type="parameter_tuning",
                description="Second optimization",
                performance_before=0.55,
                performance_after=0.65,
                parameters_changed={}
            ),
            EvolutionRecord(
                timestamp=base_time - timedelta(hours=3),
                component="draft_generator",
                improvement_type="prompt_optimization",
                description="Third optimization",
                performance_before=0.6,
                performance_after=0.7,
                parameters_changed={}
            ),
            EvolutionRecord(
                timestamp=base_time - timedelta(hours=2),
                component="retrieval_engine",
                improvement_type="caching_strategy",
                description="Fourth optimization",
                performance_before=0.65,
                performance_after=0.6,  # Negative improvement
                parameters_changed={}
            )
        ]
    
    def test_analyze_evolution_trends_improving(self, history_manager, sample_evolution_records):
        """Test trend analysis with improving performance"""
        # Modify records to show improvement
        for i, record in enumerate(sample_evolution_records):
            record.performance_after = record.performance_before + 0.05 * (i + 1)
        
        trends = history_manager.analyze_evolution_trends(sample_evolution_records)
        
        # Verify trend analysis
        assert trends["trend"] == "improving"
        assert trends["overall_improvement"] > 0.01
        assert "components" in trends
        assert trends["total_evolution_records"] == len(sample_evolution_records)
        
        # Check component-specific trends
        components = trends["components"]
        assert "draft_generator" in components
        draft_trend = components["draft_generator"]
        assert draft_trend["trend"] in ["improving", "stable", "declining"]
        assert draft_trend["total_records"] == 2  # Two draft_generator records
    
    def test_analyze_evolution_trends_declining(self, history_manager, sample_evolution_records):
        """Test trend analysis with declining performance"""
        # Modify records to show decline
        for record in sample_evolution_records:
            record.performance_after = record.performance_before - 0.05
        
        trends = history_manager.analyze_evolution_trends(sample_evolution_records)
        
        # Verify declining trend
        assert trends["trend"] == "declining"
        assert trends["overall_improvement"] < -0.01
    
    def test_analyze_evolution_trends_empty(self, history_manager):
        """Test trend analysis with empty history"""
        trends = history_manager.analyze_evolution_trends([])
        
        assert trends["trend"] == "no_data"
        assert trends["components"] == {}
        assert trends["overall_improvement"] == 0.0
    
    def test_get_performance_metrics(self, history_manager, sample_evolution_records):
        """Test performance metrics calculation"""
        metrics = history_manager.get_performance_metrics(sample_evolution_records)
        
        # Verify metrics structure
        assert "recent_average_improvement" in metrics
        assert "success_rate" in metrics
        assert "performance_stability" in metrics
        
        # Verify metric ranges
        assert isinstance(metrics["recent_average_improvement"], float)
        assert 0.0 <= metrics["success_rate"] <= 1.0
        assert 0.0 <= metrics["performance_stability"] <= 1.0
    
    def test_get_performance_metrics_empty(self, history_manager):
        """Test performance metrics with empty history"""
        metrics = history_manager.get_performance_metrics([])
        
        assert metrics == {}
    
    @pytest.mark.asyncio
    async def test_analyze_performance_trends_with_kimi(self, history_manager, mock_kimi_client_for_history, sample_evolution_records):
        """Test Kimi K2-powered performance trend analysis"""
        # Mock the Kimi client
        history_manager.kimi_client = mock_kimi_client_for_history
        
        # Mock Kimi K2 response
        mock_kimi_client_for_history.generate_structured_response.return_value = {
            "overall_trend": "improving",
            "trend_strength": 0.8,
            "component_trends": {
                "draft_generator": {
                    "trend": "improving",
                    "confidence": 0.9,
                    "key_insights": ["Consistent improvement in prompt optimization"]
                },
                "gap_analyzer": {
                    "trend": "stable",
                    "confidence": 0.7,
                    "key_insights": ["Parameter tuning showing steady results"]
                }
            },
            "performance_patterns": ["Regular improvement cycles", "Component synergy effects"],
            "anomalies": ["Retrieval engine performance drop"],
            "recommendations": ["Focus on draft_generator optimization", "Investigate retrieval_engine issues"],
            "prediction": {
                "next_iteration_performance": 0.75,
                "confidence": 0.8
            }
        }
        
        # Execute analysis
        result = await history_manager.analyze_performance_trends_with_kimi(sample_evolution_records)
        
        # Verify results
        assert "kimi_analysis" in result
        assert "traditional_metrics" in result
        assert "combined_insights" in result
        
        kimi_analysis = result["kimi_analysis"]
        assert kimi_analysis["overall_trend"] == "improving"
        assert kimi_analysis["trend_strength"] == 0.8
        assert len(kimi_analysis["component_trends"]) == 2
        assert len(kimi_analysis["performance_patterns"]) == 2
        assert len(kimi_analysis["recommendations"]) == 2
        
        # Verify combined insights
        combined = result["combined_insights"]
        assert "trend_consensus" in combined
        assert "confidence_score" in combined
        assert "key_insights" in combined
        assert "actionable_recommendations" in combined
    
    @pytest.mark.asyncio
    async def test_analyze_performance_trends_with_kimi_empty_history(self, history_manager, mock_kimi_client_for_history):
        """Test Kimi K2 trend analysis with empty history"""
        history_manager.kimi_client = mock_kimi_client_for_history
        
        result = await history_manager.analyze_performance_trends_with_kimi([])
        
        # Should return no_data response
        assert result["trend"] == "no_data"
        assert result["analysis"] == "No evolution history available"
        assert result["recommendations"] == []
    
    @pytest.mark.asyncio
    async def test_analyze_performance_trends_with_kimi_failure(self, history_manager, mock_kimi_client_for_history, sample_evolution_records):
        """Test Kimi K2 trend analysis with API failure"""
        history_manager.kimi_client = mock_kimi_client_for_history
        
        # Mock API failure
        mock_kimi_client_for_history.generate_structured_response.side_effect = Exception("API Error")
        
        result = await history_manager.analyze_performance_trends_with_kimi(sample_evolution_records)
        
        # Should fallback to traditional analysis
        assert "kimi_analysis" in result
        assert result["kimi_analysis"] is None
        assert "traditional_metrics" in result
        assert "error" in result
        assert result["error"] == "API Error"
    
    @pytest.mark.asyncio
    async def test_adaptive_learning_rate_adjustment(self, history_manager, mock_kimi_client_for_history, sample_evolution_records):
        """Test adaptive learning rate adjustment with Kimi K2"""
        history_manager.kimi_client = mock_kimi_client_for_history
        
        # Mock Kimi K2 response
        mock_kimi_client_for_history.generate_structured_response.return_value = {
            "component_learning_rates": {
                "draft_generator": 0.15,
                "gap_analyzer": 0.12,
                "retrieval_engine": 0.18,
                "information_integrator": 0.10,
                "quality_assessor": 0.08
            },
            "adjustment_rationale": {
                "draft_generator": "High stability, can handle higher learning rate",
                "gap_analyzer": "Moderate performance, conservative adjustment",
                "retrieval_engine": "Good improvement potential, aggressive rate",
                "information_integrator": "Complex component, careful tuning needed",
                "quality_assessor": "Meta-component, conservative approach"
            },
            "risk_assessment": "medium",
            "expected_impact": 0.15
        }
        
        # Execute adjustment
        result = await history_manager.adaptive_learning_rate_adjustment(sample_evolution_records, 0.7)
        
        # Verify results
        assert len(result) == 5  # All components
        for component, rate in result.items():
            assert 0.01 <= rate <= 0.5  # Within bounds
            assert isinstance(rate, float)
        
        # Verify specific rates are reasonable
        assert result["draft_generator"] == 0.15
        assert result["gap_analyzer"] == 0.12
        assert result["retrieval_engine"] == 0.18
        assert result["information_integrator"] == 0.10
        assert result["quality_assessor"] == 0.08
    
    @pytest.mark.asyncio
    async def test_adaptive_learning_rate_adjustment_failure(self, history_manager, mock_kimi_client_for_history, sample_evolution_records):
        """Test adaptive learning rate adjustment with API failure"""
        history_manager.kimi_client = mock_kimi_client_for_history
        
        # Mock API failure
        mock_kimi_client_for_history.generate_structured_response.side_effect = Exception("API Error")
        
        # Execute adjustment
        result = await history_manager.adaptive_learning_rate_adjustment(sample_evolution_records, 0.7)
        
        # Should return fallback rates
        assert len(result) == 5
        expected_fallback = {
            "draft_generator": 0.1,
            "gap_analyzer": 0.08,
            "retrieval_engine": 0.12,
            "information_integrator": 0.09,
            "quality_assessor": 0.07
        }
        
        for component, expected_rate in expected_fallback.items():
            assert result[component] == expected_rate
    
    @pytest.mark.asyncio
    async def test_predict_evolution_outcomes(self, history_manager, mock_kimi_client_for_history, sample_evolution_records):
        """Test evolution outcome prediction with Kimi K2"""
        history_manager.kimi_client = mock_kimi_client_for_history
        
        # Mock Kimi K2 response
        mock_kimi_client_for_history.generate_structured_response.return_value = {
            "predicted_outcomes": {
                "draft_generator": {
                    "performance_change": 0.12,
                    "confidence": 0.85,
                    "risk_factors": ["Prompt complexity increase"],
                    "success_probability": 0.8
                },
                "gap_analyzer": {
                    "performance_change": 0.08,
                    "confidence": 0.75,
                    "risk_factors": ["Threshold sensitivity"],
                    "success_probability": 0.7
                }
            },
            "overall_impact": {
                "expected_improvement": 0.10,
                "confidence": 0.8,
                "timeline": "short-term"
            },
            "recommendations": ["Implement draft_generator changes first", "Monitor gap_analyzer closely"],
            "alternative_approaches": ["Conservative parameter tuning", "Gradual rollout strategy"]
        }
        
        # Proposed changes
        proposed_changes = {
            "draft_generator": {"prompt_version": "v3.0"},
            "gap_analyzer": {"threshold": 0.65}
        }
        
        # Execute prediction
        result = await history_manager.predict_evolution_outcomes(sample_evolution_records, proposed_changes)
        
        # Verify results
        assert "predicted_outcomes" in result
        assert "overall_impact" in result
        assert "recommendations" in result
        assert "alternative_approaches" in result
        
        # Verify predicted outcomes
        outcomes = result["predicted_outcomes"]
        assert "draft_generator" in outcomes
        assert "gap_analyzer" in outcomes
        
        draft_outcome = outcomes["draft_generator"]
        assert draft_outcome["performance_change"] == 0.12
        assert draft_outcome["confidence"] == 0.85
        assert len(draft_outcome["risk_factors"]) == 1
        assert draft_outcome["success_probability"] == 0.8
        
        # Verify overall impact
        overall = result["overall_impact"]
        assert overall["expected_improvement"] == 0.10
        assert overall["confidence"] == 0.8
        assert overall["timeline"] == "short-term"
        
        # Verify recommendations
        assert len(result["recommendations"]) == 2
        assert len(result["alternative_approaches"]) == 2
    
    @pytest.mark.asyncio
    async def test_predict_evolution_outcomes_failure(self, history_manager, mock_kimi_client_for_history, sample_evolution_records):
        """Test evolution outcome prediction with API failure"""
        history_manager.kimi_client = mock_kimi_client_for_history
        
        # Mock API failure
        mock_kimi_client_for_history.generate_structured_response.side_effect = Exception("API Error")
        
        # Execute prediction
        result = await history_manager.predict_evolution_outcomes(sample_evolution_records, {})
        
        # Should return fallback response
        assert "predicted_outcomes" in result
        assert result["predicted_outcomes"] == {}
        assert "overall_impact" in result
        assert result["overall_impact"]["expected_improvement"] == 0.0
        assert result["overall_impact"]["confidence"] == 0.3
        assert "error" in result
        assert result["error"] == "API Error"
    
    def test_prepare_trend_data(self, history_manager, sample_evolution_records):
        """Test trend data preparation"""
        trend_data = history_manager._prepare_trend_data(sample_evolution_records)
        
        # Verify structure
        assert "total_records" in trend_data
        assert "time_span" in trend_data
        assert "component_data" in trend_data
        assert "overall_statistics" in trend_data
        
        # Verify counts
        assert trend_data["total_records"] == len(sample_evolution_records)
        assert trend_data["time_span"] >= 0
        
        # Verify component data
        component_data = trend_data["component_data"]
        assert "draft_generator" in component_data
        assert "gap_analyzer" in component_data
        assert "retrieval_engine" in component_data
        
        # Verify draft_generator has 2 records
        assert len(component_data["draft_generator"]) == 2
        
        # Verify overall statistics
        stats = trend_data["overall_statistics"]
        assert "mean_improvement" in stats
        assert "success_rate" in stats
        assert "volatility" in stats
        assert isinstance(stats["mean_improvement"], float)
        assert 0.0 <= stats["success_rate"] <= 1.0
        assert stats["volatility"] >= 0.0
    
    def test_calculate_performance_stability(self, history_manager, sample_evolution_records):
        """Test performance stability calculation"""
        stability = history_manager._calculate_performance_stability(sample_evolution_records)
        
        # Should have stability metrics for each component
        assert "draft_generator" in stability
        assert "gap_analyzer" in stability
        assert "retrieval_engine" in stability
        
        # All stability scores should be between 0 and 1
        for component, score in stability.items():
            assert 0.0 <= score <= 1.0
    
    def test_calculate_volatility(self, history_manager):
        """Test volatility calculation"""
        # Test with stable improvements
        stable_improvements = [0.1, 0.11, 0.09, 0.1, 0.12]
        stable_volatility = history_manager._calculate_volatility(stable_improvements)
        
        # Test with volatile improvements
        volatile_improvements = [0.1, -0.05, 0.2, -0.1, 0.15]
        volatile_volatility = history_manager._calculate_volatility(volatile_improvements)
        
        # Volatile should have higher volatility
        assert volatile_volatility > stable_volatility
        assert stable_volatility >= 0.0
        assert volatile_volatility >= 0.0
        
        # Test edge cases
        assert history_manager._calculate_volatility([]) == 0.0
        assert history_manager._calculate_volatility([0.1]) == 0.0
    
    def test_combine_trend_insights(self, history_manager):
        """Test combining Kimi K2 and traditional trend insights"""
        kimi_analysis = {
            "overall_trend": "improving",
            "trend_strength": 0.8,
            "performance_patterns": ["Pattern 1", "Pattern 2"],
            "recommendations": ["Rec 1", "Rec 2"]
        }
        
        traditional_analysis = {
            "trend": "improving",
            "overall_improvement": 0.05,
            "components": {
                "draft_generator": {"trend": "improving"},
                "gap_analyzer": {"trend": "declining"}
            }
        }
        
        combined = history_manager._combine_trend_insights(kimi_analysis, traditional_analysis)
        
        # Verify structure
        assert "trend_consensus" in combined
        assert "confidence_score" in combined
        assert "key_insights" in combined
        assert "actionable_recommendations" in combined
        
        # Verify consensus (both agree on improving)
        assert combined["trend_consensus"] == "consensus_improving"
        
        # Verify confidence score
        assert 0.0 <= combined["confidence_score"] <= 1.0
        
        # Verify insights include both sources
        insights = combined["key_insights"]
        assert len(insights) >= 2  # Should have patterns from Kimi + traditional insights
        
        # Verify recommendations are prioritized
        recommendations = combined["actionable_recommendations"]
        assert len(recommendations) >= 2
        assert any("[HIGH]" in rec for rec in recommendations)  # Kimi recommendations should be high priority

class TestEvolutionUtilities:
    """Test utility functions for evolution analysis"""
    
    def test_component_optimization_creation(self):
        """Test ComponentOptimization creation and validation"""
        optimization = ComponentOptimization(
            component="test_component",
            strategy_type="test_strategy",
            parameters={"param1": "value1"},
            expected_improvement=0.15,
            implementation_priority=2,
            risk_level="medium"
        )
        
        assert optimization.component == "test_component"
        assert optimization.strategy_type == "test_strategy"
        assert optimization.parameters == {"param1": "value1"}
        assert optimization.expected_improvement == 0.15
        assert optimization.implementation_priority == 2
        assert optimization.risk_level == "medium"
    
    def test_performance_analysis_creation(self):
        """Test PerformanceAnalysis creation and validation"""
        analysis = PerformanceAnalysis(
            component="test_component",
            current_performance=0.75,
            historical_trend=0.1,
            improvement_potential=0.25,
            bottlenecks=["bottleneck1", "bottleneck2"],
            optimization_opportunities=["opportunity1"],
            confidence_score=0.8
        )
        
        assert analysis.component == "test_component"
        assert analysis.current_performance == 0.75
        assert analysis.historical_trend == 0.1
        assert analysis.improvement_potential == 0.25
        assert len(analysis.bottlenecks) == 2
        assert len(analysis.optimization_opportunities) == 1
        assert analysis.confidence_score == 0.8
    
    def test_evolution_strategy_creation(self):
        """Test EvolutionStrategy creation and validation"""
        optimizations = [
            ComponentOptimization(
                component="comp1",
                strategy_type="strategy1",
                parameters={},
                expected_improvement=0.1,
                implementation_priority=1,
                risk_level="low"
            )
        ]
        
        strategy = EvolutionStrategy(
            optimizations=optimizations,
            learning_rate_adjustments={"comp1": 0.1},
            parameter_updates={"param": "value"},
            performance_predictions={"comp1": 0.8},
            implementation_order=["comp1"]
        )
        
        assert len(strategy.optimizations) == 1
        assert strategy.learning_rate_adjustments == {"comp1": 0.1}
        assert strategy.parameter_updates == {"param": "value"}
        assert strategy.performance_predictions == {"comp1": 0.8}
        assert strategy.implementation_order == ["comp1"]

if __name__ == "__main__":
    pytest.main([__file__])