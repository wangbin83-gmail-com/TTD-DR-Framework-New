"""
Unit tests for quality assessor node implementation.
Tests LangGraph integration and workflow functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from backend.workflow.quality_assessor_node import (
    quality_assessor_node, quality_check_node, quality_assessor_node_async,
    quality_check_node_async, quality_check_fallback, get_quality_summary,
    get_quality_grade, assess_improvement_potential
)
from backend.models.core import (
    TTDRState, Draft, QualityMetrics, ResearchRequirements, ResearchStructure,
    Section, ComplexityLevel, ResearchDomain, DraftMetadata
)

class TestQualityAssessorNode:
    """Test cases for quality assessor node"""
    
    @pytest.fixture
    def sample_state(self):
        """Create sample TTD-DR state for testing"""
        structure = ResearchStructure(
            sections=[
                Section(id="intro", title="Introduction"),
                Section(id="methods", title="Methods"),
                Section(id="results", title="Results")
            ],
            estimated_length=2000,
            complexity_level=ComplexityLevel.INTERMEDIATE,
            domain=ResearchDomain.TECHNOLOGY
        )
        
        draft = Draft(
            id="test-draft",
            topic="AI in Healthcare",
            structure=structure,
            content={
                "intro": "Introduction to AI in healthcare applications...",
                "methods": "Systematic review methodology was employed...",
                "results": "Results show significant improvements..."
            },
            metadata=DraftMetadata(),
            quality_score=0.0,
            iteration=1
        )
        
        requirements = ResearchRequirements(
            domain=ResearchDomain.TECHNOLOGY,
            complexity_level=ComplexityLevel.INTERMEDIATE,
            max_iterations=5,
            quality_threshold=0.8,
            max_sources=20
        )
        
        return {
            "topic": "AI in Healthcare",
            "requirements": requirements,
            "current_draft": draft,
            "information_gaps": [],
            "retrieved_info": [],
            "iteration_count": 1,
            "quality_metrics": None,
            "evolution_history": [],
            "final_report": None,
            "error_log": []
        }
    
    @pytest.fixture
    def sample_quality_metrics(self):
        """Create sample quality metrics"""
        return QualityMetrics(
            completeness=0.75,
            coherence=0.80,
            accuracy=0.70,
            citation_quality=0.65,
            overall_score=0.725
        )
    
    @pytest.mark.asyncio
    async def test_quality_assessor_node_async_success(self, sample_state):
        """Test successful async quality assessment"""
        with patch('backend.workflow.quality_assessor_node.KimiK2QualityAssessor') as mock_assessor_class:
            # Mock the assessor instance and its evaluate_draft method
            mock_assessor = AsyncMock()
            mock_assessor_class.return_value = mock_assessor
            
            expected_metrics = QualityMetrics(
                completeness=0.8,
                coherence=0.75,
                accuracy=0.7,
                citation_quality=0.6,
                overall_score=0.7125
            )
            mock_assessor.evaluate_draft.return_value = expected_metrics
            
            # Execute the node
            result_state = await quality_assessor_node_async(sample_state)
            
            # Verify results
            assert result_state["quality_metrics"] == expected_metrics
            assert result_state["current_draft"].quality_score == expected_metrics.overall_score
            assert mock_assessor.evaluate_draft.called
    
    @pytest.mark.asyncio
    async def test_quality_assessor_node_async_no_draft(self):
        """Test quality assessment with no current draft"""
        state_no_draft = {
            "topic": "Test Topic",
            "requirements": ResearchRequirements(),
            "current_draft": None,
            "information_gaps": [],
            "retrieved_info": [],
            "iteration_count": 0,
            "quality_metrics": None,
            "evolution_history": [],
            "final_report": None,
            "error_log": []
        }
        
        result_state = await quality_assessor_node_async(state_no_draft)
        
        # Should return minimal quality metrics
        assert result_state["quality_metrics"].overall_score == 0.0
        assert "No draft available" in result_state["error_log"][-1]
    
    @pytest.mark.asyncio
    async def test_quality_assessor_node_async_with_error(self, sample_state):
        """Test quality assessment with Kimi K2 error"""
        with patch('backend.workflow.quality_assessor_node.KimiK2QualityAssessor') as mock_assessor_class:
            # Mock the assessor to raise an exception
            mock_assessor = AsyncMock()
            mock_assessor_class.return_value = mock_assessor
            mock_assessor.evaluate_draft.side_effect = Exception("API Error")
            
            result_state = await quality_assessor_node_async(sample_state)
            
            # Should return fallback metrics
            assert result_state["quality_metrics"].overall_score == 0.375
            assert "Quality assessment error" in result_state["error_log"][-1]
    
    def test_quality_assessor_node_sync(self, sample_state):
        """Test synchronous wrapper for quality assessor node"""
        with patch('backend.workflow.quality_assessor_node.quality_assessor_node_async') as mock_async:
            expected_metrics = QualityMetrics(overall_score=0.8)
            expected_state = {**sample_state, "quality_metrics": expected_metrics}
            
            # Mock asyncio.run to return expected state
            with patch('asyncio.run', return_value=expected_state):
                result_state = quality_assessor_node(sample_state)
                
                assert result_state["quality_metrics"].overall_score == 0.8
    
    def test_quality_assessor_node_with_running_event_loop(self, sample_state):
        """Test quality assessor node when event loop is already running"""
        with patch('asyncio.run', side_effect=RuntimeError("asyncio.run() cannot be called from a running event loop")):
            with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
                # Mock the executor to return expected result
                mock_future = Mock()
                expected_metrics = QualityMetrics(overall_score=0.75)
                expected_state = {**sample_state, "quality_metrics": expected_metrics}
                mock_future.result.return_value = expected_state
                
                mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
                
                result_state = quality_assessor_node(sample_state)
                
                assert result_state["quality_metrics"].overall_score == 0.75

class TestQualityCheckNode:
    """Test cases for quality check decision node"""
    
    @pytest.fixture
    def sample_state_with_metrics(self, sample_state, sample_quality_metrics):
        """Create state with quality metrics"""
        return {
            **sample_state,
            "quality_metrics": sample_quality_metrics
        }
    
    @pytest.mark.asyncio
    async def test_quality_check_node_async_continue(self, sample_state_with_metrics):
        """Test quality check decision to continue iteration"""
        with patch('backend.workflow.quality_assessor_node.KimiK2QualityChecker') as mock_checker_class:
            mock_checker = AsyncMock()
            mock_checker_class.return_value = mock_checker
            mock_checker.should_continue_iteration.return_value = True
            
            decision = await quality_check_node_async(sample_state_with_metrics)
            
            assert decision == "gap_analyzer"
            assert mock_checker.should_continue_iteration.called
    
    @pytest.mark.asyncio
    async def test_quality_check_node_async_stop(self, sample_state_with_metrics):
        """Test quality check decision to stop iteration"""
        with patch('backend.workflow.quality_assessor_node.KimiK2QualityChecker') as mock_checker_class:
            mock_checker = AsyncMock()
            mock_checker_class.return_value = mock_checker
            mock_checker.should_continue_iteration.return_value = False
            
            decision = await quality_check_node_async(sample_state_with_metrics)
            
            assert decision == "self_evolution_enhancer"
    
    @pytest.mark.asyncio
    async def test_quality_check_node_async_missing_data(self):
        """Test quality check with missing quality metrics"""
        incomplete_state = {
            "topic": "Test",
            "requirements": None,
            "quality_metrics": None,
            "iteration_count": 1
        }
        
        decision = await quality_check_node_async(incomplete_state)
        
        # Should default to continuing iteration
        assert decision == "gap_analyzer"
    
    @pytest.mark.asyncio
    async def test_quality_check_node_async_with_error(self, sample_state_with_metrics):
        """Test quality check with Kimi K2 error"""
        with patch('backend.workflow.quality_assessor_node.KimiK2QualityChecker') as mock_checker_class:
            mock_checker = AsyncMock()
            mock_checker_class.return_value = mock_checker
            mock_checker.should_continue_iteration.side_effect = Exception("API Error")
            
            decision = await quality_check_node_async(sample_state_with_metrics)
            
            # Should fall back to simple logic
            assert decision in ["gap_analyzer", "self_evolution_enhancer"]
    
    def test_quality_check_node_sync(self, sample_state_with_metrics):
        """Test synchronous wrapper for quality check node"""
        with patch('asyncio.run', return_value="gap_analyzer"):
            decision = quality_check_node(sample_state_with_metrics)
            assert decision == "gap_analyzer"
    
    def test_quality_check_fallback_continue(self, sample_state_with_metrics):
        """Test fallback logic for continuing iteration"""
        # Quality below threshold, iterations remaining
        decision = quality_check_fallback(sample_state_with_metrics)
        assert decision == "gap_analyzer"
    
    def test_quality_check_fallback_stop_quality_met(self, sample_state_with_metrics):
        """Test fallback logic for stopping when quality threshold met"""
        # Set quality above threshold
        high_quality_metrics = QualityMetrics(overall_score=0.85)
        state_high_quality = {
            **sample_state_with_metrics,
            "quality_metrics": high_quality_metrics
        }
        
        decision = quality_check_fallback(state_high_quality)
        assert decision == "self_evolution_enhancer"
    
    def test_quality_check_fallback_stop_max_iterations(self, sample_state_with_metrics):
        """Test fallback logic for stopping at max iterations"""
        # Set iteration count to max
        state_max_iterations = {
            **sample_state_with_metrics,
            "iteration_count": 5  # Max iterations from requirements
        }
        
        decision = quality_check_fallback(state_max_iterations)
        assert decision == "self_evolution_enhancer"
    
    def test_quality_check_fallback_missing_data(self):
        """Test fallback logic with missing data"""
        incomplete_state = {
            "quality_metrics": None,
            "requirements": None,
            "iteration_count": 1
        }
        
        decision = quality_check_fallback(incomplete_state)
        assert decision == "gap_analyzer"

class TestQualityUtilityFunctions:
    """Test cases for quality utility functions"""
    
    @pytest.fixture
    def sample_quality_metrics(self):
        """Create sample quality metrics"""
        return QualityMetrics(
            completeness=0.85,
            coherence=0.78,
            accuracy=0.82,
            citation_quality=0.65,
            overall_score=0.775
        )
    
    def test_get_quality_summary(self, sample_quality_metrics):
        """Test quality summary generation"""
        summary = get_quality_summary(sample_quality_metrics)
        
        assert summary["overall_score"] == 0.775
        assert summary["completeness"] == 0.85
        assert summary["coherence"] == 0.78
        assert summary["accuracy"] == 0.82
        assert summary["citation_quality"] == 0.65
        assert summary["quality_grade"] == "C"  # 0.775 should be grade C
    
    def test_get_quality_grade(self):
        """Test quality grade conversion"""
        assert get_quality_grade(0.95) == "A"
        assert get_quality_grade(0.85) == "B"
        assert get_quality_grade(0.75) == "C"
        assert get_quality_grade(0.65) == "D"
        assert get_quality_grade(0.45) == "F"
    
    def test_assess_improvement_potential(self, sample_quality_metrics):
        """Test improvement potential assessment"""
        analysis = assess_improvement_potential(sample_quality_metrics)
        
        assert analysis["current_score"] == 0.775
        assert "citation_quality" in analysis["improvement_areas"]  # 0.65 < 0.7
        assert "completeness" in analysis["strengths"]  # 0.85 >= 0.8
        assert isinstance(analysis["improvement_areas"], list)
        assert isinstance(analysis["strengths"], list)
    
    def test_assess_improvement_potential_with_previous(self, sample_quality_metrics):
        """Test improvement potential with previous metrics"""
        previous_metrics = QualityMetrics(overall_score=0.7)
        
        analysis = assess_improvement_potential(sample_quality_metrics, previous_metrics)
        
        assert analysis["improvement_trend"] == 0.075  # 0.775 - 0.7
        assert analysis["is_improving"] is True  # Improvement > 0.01
    
    def test_assess_improvement_potential_declining(self, sample_quality_metrics):
        """Test improvement potential with declining quality"""
        previous_metrics = QualityMetrics(overall_score=0.8)
        
        analysis = assess_improvement_potential(sample_quality_metrics, previous_metrics)
        
        assert analysis["improvement_trend"] == -0.025  # 0.775 - 0.8
        assert analysis["is_improving"] is False  # Improvement <= 0.01

class TestQualityAssessmentIntegration:
    """Integration tests for quality assessment workflow"""
    
    def test_complete_quality_assessment_workflow(self):
        """Test complete quality assessment workflow integration"""
        # Create initial state
        structure = ResearchStructure(
            sections=[Section(id="test", title="Test Section")],
            estimated_length=1000,
            complexity_level=ComplexityLevel.BASIC,
            domain=ResearchDomain.GENERAL
        )
        
        draft = Draft(
            id="integration-test",
            topic="Integration Test Topic",
            structure=structure,
            content={"test": "Test content for integration testing..."}
        )
        
        requirements = ResearchRequirements(
            quality_threshold=0.8,
            max_iterations=3
        )
        
        initial_state = {
            "topic": "Integration Test Topic",
            "requirements": requirements,
            "current_draft": draft,
            "information_gaps": [],
            "retrieved_info": [],
            "iteration_count": 1,
            "quality_metrics": None,
            "evolution_history": [],
            "final_report": None,
            "error_log": []
        }
        
        # Mock Kimi K2 responses for quality assessment
        with patch('backend.workflow.quality_assessor_node.KimiK2QualityAssessor') as mock_assessor_class:
            mock_assessor = AsyncMock()
            mock_assessor_class.return_value = mock_assessor
            
            test_metrics = QualityMetrics(
                completeness=0.7,
                coherence=0.75,
                accuracy=0.65,
                citation_quality=0.6,
                overall_score=0.675
            )
            mock_assessor.evaluate_draft.return_value = test_metrics
            
            # Execute quality assessment
            assessed_state = quality_assessor_node(initial_state)
            
            # Verify quality assessment results
            assert assessed_state["quality_metrics"] == test_metrics
            assert assessed_state["current_draft"].quality_score == test_metrics.overall_score
        
        # Mock Kimi K2 responses for quality check
        with patch('backend.workflow.quality_assessor_node.KimiK2QualityChecker') as mock_checker_class:
            mock_checker = AsyncMock()
            mock_checker_class.return_value = mock_checker
            mock_checker.should_continue_iteration.return_value = True
            
            # Execute quality check
            decision = quality_check_node(assessed_state)
            
            # Should continue iteration since quality is below threshold
            assert decision == "gap_analyzer"
    
    def test_quality_assessment_error_handling(self):
        """Test error handling in quality assessment workflow"""
        # Create state with minimal data
        minimal_state = {
            "topic": "Error Test",
            "requirements": ResearchRequirements(),
            "current_draft": None,
            "information_gaps": [],
            "retrieved_info": [],
            "iteration_count": 0,
            "quality_metrics": None,
            "evolution_history": [],
            "final_report": None,
            "error_log": []
        }
        
        # Execute quality assessment with no draft
        result_state = quality_assessor_node(minimal_state)
        
        # Should handle gracefully
        assert result_state["quality_metrics"].overall_score == 0.0
        assert len(result_state["error_log"]) > 0
        
        # Execute quality check with no metrics
        decision = quality_check_node(result_state)
        
        # Should default to continuing
        assert decision == "gap_analyzer"

if __name__ == "__main__":
    pytest.main([__file__])