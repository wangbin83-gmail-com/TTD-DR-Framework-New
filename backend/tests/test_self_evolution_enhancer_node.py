"""
Unit tests for self-evolution enhancer node implementation.
Tests LangGraph workflow integration and node functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import List, Dict, Any

from backend.workflow.self_evolution_enhancer_node import (
    self_evolution_enhancer_node, self_evolution_enhancer_node_async,
    analyze_evolution_effectiveness, get_evolution_summary, predict_evolution_impact
)
from backend.models.core import (
    TTDRState, QualityMetrics, EvolutionRecord, ResearchRequirements,
    ComplexityLevel, ResearchDomain
)

class TestSelfEvolutionEnhancerNode:
    """Test cases for self-evolution enhancer node"""
    
    @pytest.fixture
    def sample_quality_metrics(self):
        """Sample quality metrics for testing"""
        return QualityMetrics(
            completeness=0.75,
            coherence=0.8,
            accuracy=0.65,
            citation_quality=0.6,
            overall_score=0.7
        )
    
    @pytest.fixture
    def sample_evolution_history(self):
        """Sample evolution history for testing"""
        return [
            EvolutionRecord(
                timestamp=datetime.now() - timedelta(hours=3),
                component="draft_generator",
                improvement_type="prompt_optimization",
                description="Optimized draft generation prompts",
                performance_before=0.6,
                performance_after=0.7,
                parameters_changed={"prompt_version": "v2.0"}
            ),
            EvolutionRecord(
                timestamp=datetime.now() - timedelta(hours=2),
                component="gap_analyzer",
                improvement_type="parameter_tuning",
                description="Adjusted gap detection parameters",
                performance_before=0.55,
                performance_after=0.65,
                parameters_changed={"threshold": 0.7}
            ),
            EvolutionRecord(
                timestamp=datetime.now() - timedelta(hours=1),
                component="retrieval_engine",
                improvement_type="caching_strategy",
                description="Implemented result caching",
                performance_before=0.7,
                performance_after=0.75,
                parameters_changed={"cache_size": 1000}
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
            "topic": "Machine Learning in Healthcare",
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
    
    @pytest.fixture
    def mock_evolution_enhancer(self):
        """Mock evolution enhancer for testing"""
        mock_enhancer = Mock()
        mock_enhancer.evolve_components = AsyncMock()
        return mock_enhancer
    
    @pytest.fixture
    def mock_history_manager(self):
        """Mock evolution history manager for testing"""
        mock_manager = Mock()
        mock_manager.analyze_evolution_trends = Mock()
        mock_manager.get_performance_metrics = Mock()
        return mock_manager
    
    @pytest.mark.asyncio
    async def test_self_evolution_enhancer_node_async_success(self, sample_ttdr_state,
                                                            mock_evolution_enhancer,
                                                            mock_history_manager):
        """Test successful async self-evolution node execution"""
        # Mock evolution record result
        evolution_record = EvolutionRecord(
            timestamp=datetime.now(),
            component="framework_wide",
            improvement_type="self_evolution",
            description="Applied 3 optimizations",
            performance_before=0.7,
            performance_after=0.78,
            parameters_changed={"global_learning_rate": 0.05}
        )
        
        mock_evolution_enhancer.evolve_components.return_value = evolution_record
        
        # Mock trend analysis
        mock_history_manager.analyze_evolution_trends.return_value = {
            "trend": "improving",
            "overall_improvement": 0.05,
            "components": {}
        }
        
        # Mock performance metrics
        mock_history_manager.get_performance_metrics.return_value = {
            "success_rate": 0.8,
            "performance_stability": 0.7
        }
        
        # Patch the imports
        with patch('backend.workflow.self_evolution_enhancer_node.KimiK2SelfEvolutionEnhancer',
                  return_value=mock_evolution_enhancer), \
             patch('backend.workflow.self_evolution_enhancer_node.EvolutionHistoryManager',
                  return_value=mock_history_manager):
            
            # Execute node
            result_state = await self_evolution_enhancer_node_async(sample_ttdr_state)
        
        # Verify results
        assert "evolution_history" in result_state
        assert len(result_state["evolution_history"]) == len(sample_ttdr_state["evolution_history"]) + 1
        
        # Check the new evolution record
        new_record = result_state["evolution_history"][-1]
        assert new_record.component == "framework_wide"
        assert new_record.improvement_type == "self_evolution"
        assert new_record.performance_after >= new_record.performance_before
        
        # Check framework parameters were updated
        assert "framework_parameters" in result_state
        assert "global_learning_rate" in result_state["framework_parameters"]
        
        # Verify mocks were called
        mock_evolution_enhancer.evolve_components.assert_called_once()
        mock_history_manager.analyze_evolution_trends.assert_called_once()
        mock_history_manager.get_performance_metrics.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_self_evolution_enhancer_node_async_no_quality_metrics(self):
        """Test node execution with no quality metrics"""
        # State without quality metrics
        state_without_metrics = {
            "topic": "Test Topic",
            "requirements": ResearchRequirements(),
            "current_draft": None,
            "information_gaps": [],
            "retrieved_info": [],
            "iteration_count": 1,
            "quality_metrics": None,
            "evolution_history": [],
            "final_report": None,
            "error_log": []
        }
        
        # Execute node
        result_state = await self_evolution_enhancer_node_async(state_without_metrics)
        
        # Verify fallback behavior
        assert "evolution_history" in result_state
        assert len(result_state["evolution_history"]) == 1
        
        # Check the evolution record
        evolution_record = result_state["evolution_history"][0]
        assert evolution_record.component == "framework_wide"
        assert evolution_record.improvement_type == "no_evolution"
        assert "No quality metrics" in evolution_record.description
        
        # Check error was logged
        assert "error_log" in result_state
        assert len(result_state["error_log"]) == 1
        assert "No quality metrics" in result_state["error_log"][0]
    
    @pytest.mark.asyncio
    async def test_self_evolution_enhancer_node_async_evolution_failure(self, sample_ttdr_state,
                                                                      mock_evolution_enhancer):
        """Test node execution with evolution failure"""
        # Mock evolution failure
        mock_evolution_enhancer.evolve_components.side_effect = Exception("Evolution failed")
        
        with patch('backend.workflow.self_evolution_enhancer_node.KimiK2SelfEvolutionEnhancer',
                  return_value=mock_evolution_enhancer), \
             patch('backend.workflow.self_evolution_enhancer_node.EvolutionHistoryManager'):
            
            # Execute node
            result_state = await self_evolution_enhancer_node_async(sample_ttdr_state)
        
        # Verify error handling
        assert "evolution_history" in result_state
        assert len(result_state["evolution_history"]) == len(sample_ttdr_state["evolution_history"]) + 1
        
        # Check error evolution record
        error_record = result_state["evolution_history"][-1]
        assert error_record.component == "framework_wide"
        assert error_record.improvement_type == "evolution_error"
        assert "Evolution failed" in error_record.description
        
        # Check error was logged
        assert "error_log" in result_state
        assert any("Self-evolution error" in error for error in result_state["error_log"])
    
    def test_self_evolution_enhancer_node_sync_wrapper(self, sample_ttdr_state):
        """Test synchronous wrapper function"""
        with patch('backend.workflow.self_evolution_enhancer_node.self_evolution_enhancer_node_async') as mock_async:
            # Mock async function
            mock_async.return_value = sample_ttdr_state
            
            # Execute sync wrapper
            result = self_evolution_enhancer_node(sample_ttdr_state)
            
            # Verify result
            assert result == sample_ttdr_state
            mock_async.assert_called_once_with(sample_ttdr_state)
    
    def test_self_evolution_enhancer_node_event_loop_running(self, sample_ttdr_state):
        """Test sync wrapper with running event loop"""
        with patch('asyncio.run') as mock_run, \
             patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
            
            # Mock event loop already running error
            mock_run.side_effect = RuntimeError("asyncio.run() cannot be called from a running event loop")
            
            # Mock thread executor
            mock_future = Mock()
            mock_future.result.return_value = sample_ttdr_state
            mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
            
            # Execute sync wrapper
            result = self_evolution_enhancer_node(sample_ttdr_state)
            
            # Verify thread execution was used
            assert result == sample_ttdr_state
            mock_executor.assert_called_once()

class TestEvolutionAnalysisUtilities:
    """Test evolution analysis utility functions"""
    
    @pytest.fixture
    def sample_evolution_records(self):
        """Sample evolution records for testing"""
        return [
            EvolutionRecord(
                timestamp=datetime.now() - timedelta(hours=5),
                component="draft_generator",
                improvement_type="prompt_optimization",
                description="First optimization",
                performance_before=0.5,
                performance_after=0.6,
                parameters_changed={}
            ),
            EvolutionRecord(
                timestamp=datetime.now() - timedelta(hours=4),
                component="gap_analyzer",
                improvement_type="parameter_tuning",
                description="Second optimization",
                performance_before=0.55,
                performance_after=0.65,
                parameters_changed={}
            ),
            EvolutionRecord(
                timestamp=datetime.now() - timedelta(hours=3),
                component="draft_generator",
                improvement_type="prompt_optimization",
                description="Third optimization",
                performance_before=0.6,
                performance_after=0.7,
                parameters_changed={}
            ),
            EvolutionRecord(
                timestamp=datetime.now() - timedelta(hours=2),
                component="retrieval_engine",
                improvement_type="caching_strategy",
                description="Fourth optimization",
                performance_before=0.65,
                performance_after=0.6,  # Negative improvement
                parameters_changed={}
            ),
            EvolutionRecord(
                timestamp=datetime.now() - timedelta(hours=1),
                component="information_integrator",
                improvement_type="algorithm_improvement",
                description="Fifth optimization",
                performance_before=0.7,
                performance_after=0.75,
                parameters_changed={}
            )
        ]
    
    def test_analyze_evolution_effectiveness_success(self, sample_evolution_records):
        """Test evolution effectiveness analysis with successful records"""
        analysis = analyze_evolution_effectiveness(sample_evolution_records)
        
        # Verify analysis structure
        assert analysis["status"] == "analyzed"
        assert "overall_success_rate" in analysis
        assert "overall_avg_improvement" in analysis
        assert "strategy_effectiveness" in analysis
        assert "total_evolution_attempts" in analysis
        assert "recommendations" in analysis
        
        # Verify metrics
        assert 0.0 <= analysis["overall_success_rate"] <= 1.0
        assert isinstance(analysis["overall_avg_improvement"], float)
        assert analysis["total_evolution_attempts"] == len(sample_evolution_records)
        
        # Verify strategy effectiveness
        strategy_effectiveness = analysis["strategy_effectiveness"]
        assert "prompt_optimization" in strategy_effectiveness
        assert "parameter_tuning" in strategy_effectiveness
        
        # Check strategy metrics
        for strategy, metrics in strategy_effectiveness.items():
            assert "average_improvement" in metrics
            assert "success_rate" in metrics
            assert "total_attempts" in metrics
            assert "effectiveness_score" in metrics
            assert 0.0 <= metrics["success_rate"] <= 1.0
            assert 0.0 <= metrics["effectiveness_score"] <= 1.0
        
        # Verify recommendations
        assert isinstance(analysis["recommendations"], list)
        assert len(analysis["recommendations"]) > 0
    
    def test_analyze_evolution_effectiveness_empty(self):
        """Test evolution effectiveness analysis with empty history"""
        analysis = analyze_evolution_effectiveness([])
        
        assert analysis["status"] == "no_data"
        assert "recommendations" in analysis
        assert len(analysis["recommendations"]) > 0
        assert "Start collecting evolution data" in analysis["recommendations"][0]
    
    def test_get_evolution_summary_with_data(self, sample_evolution_records):
        """Test evolution summary with data"""
        summary = get_evolution_summary(sample_evolution_records, recent_count=3)
        
        # Verify summary structure
        assert summary["status"] == "active"
        assert "recent_activities" in summary
        assert "performance_trend" in summary
        assert "component_activity" in summary
        assert "total_evolution_records" in summary
        assert "recent_records_analyzed" in summary
        
        # Verify recent activities
        recent_activities = summary["recent_activities"]
        assert len(recent_activities) <= 3  # Limited by recent_count
        
        for activity in recent_activities:
            assert "timestamp" in activity
            assert "component" in activity
            assert "improvement_type" in activity
            assert "performance_change" in activity
            assert "description" in activity
        
        # Verify performance trend
        trend = summary["performance_trend"]
        assert trend in ["strongly_improving", "improving", "stable", "declining", 
                        "strongly_declining", "insufficient_data"]
        
        # Verify component activity
        component_activity = summary["component_activity"]
        assert isinstance(component_activity, dict)
        
        for component, activity in component_activity.items():
            assert "count" in activity
            assert "total_improvement" in activity
            assert activity["count"] > 0
        
        # Verify counts
        assert summary["total_evolution_records"] == len(sample_evolution_records)
        assert summary["recent_records_analyzed"] <= len(sample_evolution_records)
    
    def test_get_evolution_summary_empty(self):
        """Test evolution summary with empty history"""
        summary = get_evolution_summary([])
        
        assert summary["status"] == "no_evolution_data"
        assert summary["recent_activities"] == []
        assert summary["performance_trend"] == "unknown"
    
    def test_predict_evolution_impact_with_history(self, sample_evolution_records):
        """Test evolution impact prediction with history"""
        current_quality = 0.75
        prediction = predict_evolution_impact(current_quality, sample_evolution_records)
        
        # Verify prediction structure
        assert "predicted_improvement" in prediction
        assert "confidence" in prediction
        assert "improvement_stability" in prediction
        assert "historical_average" in prediction
        assert "recommendation" in prediction
        assert "analysis_window" in prediction
        
        # Verify prediction ranges
        assert isinstance(prediction["predicted_improvement"], float)
        assert 0.0 <= prediction["confidence"] <= 1.0
        assert 0.0 <= prediction["improvement_stability"] <= 1.0
        assert isinstance(prediction["historical_average"], float)
        assert isinstance(prediction["recommendation"], str)
        assert prediction["analysis_window"] > 0
        
        # Verify recommendation is meaningful
        assert len(prediction["recommendation"]) > 10  # Should be a meaningful sentence
    
    def test_predict_evolution_impact_empty_history(self):
        """Test evolution impact prediction with empty history"""
        current_quality = 0.6
        prediction = predict_evolution_impact(current_quality, [])
        
        # Verify default prediction
        assert prediction["predicted_improvement"] == 0.05
        assert prediction["confidence"] == 0.3
        assert "conservative" in prediction["recommendation"].lower()
    
    def test_predict_evolution_impact_high_quality(self, sample_evolution_records):
        """Test evolution impact prediction with high current quality"""
        current_quality = 0.9  # High quality
        prediction = predict_evolution_impact(current_quality, sample_evolution_records)
        
        # Should predict smaller improvements for high quality
        assert prediction["predicted_improvement"] <= 0.1  # Diminishing returns
    
    def test_predict_evolution_impact_low_quality(self, sample_evolution_records):
        """Test evolution impact prediction with low current quality"""
        current_quality = 0.4  # Low quality
        prediction = predict_evolution_impact(current_quality, sample_evolution_records)
        
        # Should predict larger improvements for low quality
        # (Note: actual value depends on historical data, just check it's reasonable)
        assert isinstance(prediction["predicted_improvement"], float)
        assert prediction["confidence"] > 0.0

class TestEvolutionNodeIntegration:
    """Integration tests for evolution node"""
    
    @pytest.fixture
    def complete_ttdr_state(self):
        """Complete TTD-R state for integration testing"""
        requirements = ResearchRequirements(
            domain=ResearchDomain.SCIENCE,
            complexity_level=ComplexityLevel.ADVANCED,
            max_iterations=8,
            quality_threshold=0.85
        )
        
        quality_metrics = QualityMetrics(
            completeness=0.8,
            coherence=0.75,
            accuracy=0.7,
            citation_quality=0.65,
            overall_score=0.725
        )
        
        evolution_history = [
            EvolutionRecord(
                timestamp=datetime.now() - timedelta(hours=6),
                component="draft_generator",
                improvement_type="prompt_optimization",
                description="Initial prompt optimization",
                performance_before=0.5,
                performance_after=0.6,
                parameters_changed={"prompt_version": "v1.1"}
            ),
            EvolutionRecord(
                timestamp=datetime.now() - timedelta(hours=4),
                component="gap_analyzer",
                improvement_type="parameter_tuning",
                description="Gap detection tuning",
                performance_before=0.55,
                performance_after=0.7,
                parameters_changed={"threshold": 0.65}
            ),
            EvolutionRecord(
                timestamp=datetime.now() - timedelta(hours=2),
                component="information_integrator",
                improvement_type="algorithm_improvement",
                description="Integration algorithm enhancement",
                performance_before=0.6,
                performance_after=0.72,
                parameters_changed={"integration_strategy": "v2.0"}
            )
        ]
        
        return {
            "topic": "Quantum Computing Applications in Drug Discovery",
            "requirements": requirements,
            "current_draft": None,
            "information_gaps": [],
            "retrieved_info": [],
            "iteration_count": 3,
            "quality_metrics": quality_metrics,
            "evolution_history": evolution_history,
            "final_report": None,
            "error_log": [],
            "framework_parameters": {
                "global_learning_rate": 0.1,
                "optimization_threshold": 0.05
            }
        }
    
    def test_evolution_node_state_preservation(self, complete_ttdr_state):
        """Test that evolution node preserves important state information"""
        with patch('backend.workflow.self_evolution_enhancer_node.KimiK2SelfEvolutionEnhancer') as mock_enhancer_class:
            # Mock the enhancer
            mock_enhancer = Mock()
            mock_enhancer.evolve_components = AsyncMock()
            mock_enhancer.evolve_components.return_value = EvolutionRecord(
                timestamp=datetime.now(),
                component="framework_wide",
                improvement_type="self_evolution",
                description="Test evolution",
                performance_before=0.725,
                performance_after=0.75,
                parameters_changed={"new_param": "value"}
            )
            mock_enhancer_class.return_value = mock_enhancer
            
            with patch('backend.workflow.self_evolution_enhancer_node.EvolutionHistoryManager') as mock_history_class:
                # Mock the history manager
                mock_history_manager = Mock()
                mock_history_manager.analyze_evolution_trends.return_value = {
                    "trend": "improving",
                    "overall_improvement": 0.05
                }
                mock_history_manager.get_performance_metrics.return_value = {
                    "success_rate": 0.8,
                    "performance_stability": 0.7
                }
                mock_history_class.return_value = mock_history_manager
                
                # Execute node
                result_state = self_evolution_enhancer_node(complete_ttdr_state)
        
        # Verify state preservation
        assert result_state["topic"] == complete_ttdr_state["topic"]
        assert result_state["requirements"] == complete_ttdr_state["requirements"]
        assert result_state["iteration_count"] == complete_ttdr_state["iteration_count"]
        assert result_state["quality_metrics"] == complete_ttdr_state["quality_metrics"]
        
        # Verify evolution history was extended
        assert len(result_state["evolution_history"]) == len(complete_ttdr_state["evolution_history"]) + 1
        
        # Verify framework parameters were updated
        assert "framework_parameters" in result_state
        assert "new_param" in result_state["framework_parameters"]
        assert result_state["framework_parameters"]["global_learning_rate"] == 0.1  # Preserved
    
    def test_evolution_effectiveness_analysis_integration(self, complete_ttdr_state):
        """Test integration of evolution effectiveness analysis"""
        evolution_history = complete_ttdr_state["evolution_history"]
        
        # Analyze effectiveness
        effectiveness = analyze_evolution_effectiveness(evolution_history)
        
        # Verify analysis provides actionable insights
        assert effectiveness["status"] == "analyzed"
        assert effectiveness["total_evolution_attempts"] == len(evolution_history)
        
        # Check that all improvement types are analyzed
        strategy_effectiveness = effectiveness["strategy_effectiveness"]
        expected_strategies = {"prompt_optimization", "parameter_tuning", "algorithm_improvement"}
        actual_strategies = set(strategy_effectiveness.keys())
        assert expected_strategies.issubset(actual_strategies)
        
        # Verify recommendations are provided
        recommendations = effectiveness["recommendations"]
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)

if __name__ == "__main__":
    pytest.main([__file__])