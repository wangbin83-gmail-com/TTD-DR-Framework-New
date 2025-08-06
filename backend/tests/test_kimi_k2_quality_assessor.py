"""
Unit tests for Kimi K2 quality assessor service.
Tests comprehensive quality evaluation functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from backend.services.kimi_k2_quality_assessor import KimiK2QualityAssessor, KimiK2QualityChecker
from backend.models.core import (
    Draft, QualityMetrics, ResearchRequirements, ResearchStructure, Section,
    ComplexityLevel, ResearchDomain, DraftMetadata
)
from backend.services.kimi_k2_client import KimiK2Error

class TestKimiK2QualityAssessor:
    """Test cases for KimiK2QualityAssessor"""
    
    @pytest.fixture
    def sample_draft(self):
        """Create a sample draft for testing"""
        structure = ResearchStructure(
            sections=[
                Section(id="intro", title="Introduction", content="Introduction content"),
                Section(id="methods", title="Methods", content="Methods content"),
                Section(id="results", title="Results", content="Results content"),
                Section(id="conclusion", title="Conclusion", content="Conclusion content")
            ],
            estimated_length=2000,
            complexity_level=ComplexityLevel.INTERMEDIATE,
            domain=ResearchDomain.TECHNOLOGY
        )
        
        return Draft(
            id="test-draft-1",
            topic="Artificial Intelligence in Healthcare",
            structure=structure,
            content={
                "intro": "This research explores the applications of AI in healthcare...",
                "methods": "We conducted a systematic review of recent literature...",
                "results": "Our analysis revealed significant improvements in diagnostic accuracy...",
                "conclusion": "AI shows great promise for transforming healthcare delivery..."
            },
            metadata=DraftMetadata(),
            quality_score=0.0,
            iteration=1
        )
    
    @pytest.fixture
    def sample_requirements(self):
        """Create sample research requirements"""
        return ResearchRequirements(
            domain=ResearchDomain.TECHNOLOGY,
            complexity_level=ComplexityLevel.INTERMEDIATE,
            max_iterations=5,
            quality_threshold=0.8,
            max_sources=20,
            preferred_source_types=["academic", "news"]
        )
    
    @pytest.fixture
    def mock_kimi_response(self):
        """Mock Kimi K2 API response for quality assessment"""
        return {
            "completeness_score": 0.85,
            "missing_elements": ["detailed methodology", "statistical analysis"],
            "coverage_analysis": "Good coverage of main topics with some gaps in technical details",
            "recommendations": ["Add more technical depth", "Include statistical validation"]
        }
    
    @pytest.fixture
    def quality_assessor(self):
        """Create KimiK2QualityAssessor instance"""
        return KimiK2QualityAssessor()
    
    @pytest.mark.asyncio
    async def test_evaluate_draft_success(self, quality_assessor, sample_draft, sample_requirements, mock_kimi_response):
        """Test successful draft quality evaluation"""
        # Mock Kimi K2 client responses
        with patch.object(quality_assessor.kimi_client, 'generate_structured_response', new_callable=AsyncMock) as mock_generate:
            # Set up different responses for each assessment dimension
            mock_generate.side_effect = [
                {"completeness_score": 0.85},  # Completeness
                {"coherence_score": 0.78},     # Coherence
                {"accuracy_score": 0.82},      # Accuracy
                {"citation_score": 0.65}       # Citation quality
            ]
            
            # Execute quality evaluation
            quality_metrics = await quality_assessor.evaluate_draft(sample_draft, sample_requirements)
            
            # Verify results
            assert isinstance(quality_metrics, QualityMetrics)
            assert quality_metrics.completeness == 0.85
            assert quality_metrics.coherence == 0.78
            assert quality_metrics.accuracy == 0.82
            assert quality_metrics.citation_quality == 0.65
            
            # Verify overall score calculation
            expected_overall = (0.85 * 0.3 + 0.78 * 0.25 + 0.82 * 0.25 + 0.65 * 0.2)
            assert abs(quality_metrics.overall_score - expected_overall) < 0.001
            
            # Verify Kimi K2 client was called for each dimension
            assert mock_generate.call_count == 4
    
    @pytest.mark.asyncio
    async def test_evaluate_draft_with_api_error(self, quality_assessor, sample_draft):
        """Test quality evaluation with Kimi K2 API error"""
        # Mock Kimi K2 client to raise error
        with patch.object(quality_assessor.kimi_client, 'generate_structured_response', new_callable=AsyncMock) as mock_generate:
            mock_generate.side_effect = KimiK2Error("API rate limit exceeded")
            
            # Execute quality evaluation
            quality_metrics = await quality_assessor.evaluate_draft(sample_draft)
            
            # Verify fallback metrics are returned
            assert isinstance(quality_metrics, QualityMetrics)
            assert quality_metrics.overall_score == 0.3  # Fallback value
    
    @pytest.mark.asyncio
    async def test_assess_completeness(self, quality_assessor, sample_draft, sample_requirements):
        """Test completeness assessment"""
        with patch.object(quality_assessor.kimi_client, 'generate_structured_response', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = {"completeness_score": 0.75}
            
            content = quality_assessor._prepare_draft_content(sample_draft)
            completeness = await quality_assessor._assess_completeness(sample_draft, content, sample_requirements)
            
            assert completeness == 0.75
            assert mock_generate.called
            
            # Verify prompt contains relevant information
            call_args = mock_generate.call_args[0]
            prompt = call_args[0]
            assert "completeness" in prompt.lower()
            assert sample_draft.topic in prompt
    
    @pytest.mark.asyncio
    async def test_assess_coherence(self, quality_assessor, sample_draft):
        """Test coherence assessment"""
        with patch.object(quality_assessor.kimi_client, 'generate_structured_response', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = {"coherence_score": 0.88}
            
            content = quality_assessor._prepare_draft_content(sample_draft)
            coherence = await quality_assessor._assess_coherence(sample_draft, content)
            
            assert coherence == 0.88
            assert mock_generate.called
    
    @pytest.mark.asyncio
    async def test_assess_accuracy(self, quality_assessor, sample_draft):
        """Test accuracy assessment"""
        with patch.object(quality_assessor.kimi_client, 'generate_structured_response', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = {"accuracy_score": 0.72}
            
            content = quality_assessor._prepare_draft_content(sample_draft)
            accuracy = await quality_assessor._assess_accuracy(sample_draft, content)
            
            assert accuracy == 0.72
            assert mock_generate.called
    
    @pytest.mark.asyncio
    async def test_assess_citation_quality(self, quality_assessor, sample_draft):
        """Test citation quality assessment"""
        with patch.object(quality_assessor.kimi_client, 'generate_structured_response', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = {"citation_score": 0.68}
            
            content = quality_assessor._prepare_draft_content(sample_draft)
            citation_quality = await quality_assessor._assess_citation_quality(sample_draft, content)
            
            assert citation_quality == 0.68
            assert mock_generate.called
    
    def test_prepare_draft_content(self, quality_assessor, sample_draft):
        """Test draft content preparation"""
        content = quality_assessor._prepare_draft_content(sample_draft)
        
        assert sample_draft.topic in content
        assert "Introduction" in content
        assert "Methods" in content
        assert "Results" in content
        assert "Conclusion" in content
        assert "This research explores" in content
    
    def test_fallback_completeness_assessment(self, quality_assessor, sample_draft):
        """Test fallback completeness assessment"""
        content = quality_assessor._prepare_draft_content(sample_draft)
        completeness = quality_assessor._fallback_completeness_assessment(sample_draft, content)
        
        assert 0.0 <= completeness <= 1.0
        assert completeness > 0.5  # Should be reasonably high for filled draft
    
    def test_fallback_coherence_assessment(self, quality_assessor, sample_draft):
        """Test fallback coherence assessment"""
        content = quality_assessor._prepare_draft_content(sample_draft)
        coherence = quality_assessor._fallback_coherence_assessment(sample_draft, content)
        
        assert 0.0 <= coherence <= 1.0
    
    def test_fallback_accuracy_assessment(self, quality_assessor, sample_draft):
        """Test fallback accuracy assessment"""
        content = quality_assessor._prepare_draft_content(sample_draft)
        accuracy = quality_assessor._fallback_accuracy_assessment(sample_draft, content)
        
        assert 0.0 <= accuracy <= 1.0
    
    def test_fallback_citation_assessment(self, quality_assessor, sample_draft):
        """Test fallback citation assessment"""
        content = quality_assessor._prepare_draft_content(sample_draft)
        citation_quality = quality_assessor._fallback_citation_assessment(sample_draft, content)
        
        assert 0.0 <= citation_quality <= 1.0
    
    def test_handle_assessment_result(self, quality_assessor):
        """Test assessment result handling"""
        # Test normal float result
        result = quality_assessor._handle_assessment_result(0.75, "test", 0.5)
        assert result == 0.75
        
        # Test exception result
        exception = Exception("Test error")
        result = quality_assessor._handle_assessment_result(exception, "test", 0.5)
        assert result == 0.5
        
        # Test out-of-bounds values
        result = quality_assessor._handle_assessment_result(1.5, "test", 0.5)
        assert result == 1.0
        
        result = quality_assessor._handle_assessment_result(-0.5, "test", 0.5)
        assert result == 0.0

class TestKimiK2QualityChecker:
    """Test cases for KimiK2QualityChecker"""
    
    @pytest.fixture
    def quality_checker(self):
        """Create KimiK2QualityChecker instance"""
        return KimiK2QualityChecker()
    
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
    
    @pytest.fixture
    def sample_requirements(self):
        """Create sample research requirements"""
        return ResearchRequirements(
            quality_threshold=0.8,
            max_iterations=5
        )
    
    @pytest.mark.asyncio
    async def test_should_continue_iteration_below_threshold(self, quality_checker, sample_quality_metrics, sample_requirements):
        """Test continuation decision when below quality threshold"""
        with patch.object(quality_checker.kimi_client, 'generate_structured_response', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = {
                "should_continue": True,
                "reasoning": "Quality below threshold with improvement potential",
                "improvement_potential": 0.8
            }
            
            should_continue = await quality_checker.should_continue_iteration(
                sample_quality_metrics, 2, sample_requirements
            )
            
            assert should_continue is True
            assert mock_generate.called
    
    @pytest.mark.asyncio
    async def test_should_continue_iteration_above_threshold(self, quality_checker, sample_requirements):
        """Test continuation decision when above quality threshold"""
        high_quality_metrics = QualityMetrics(
            completeness=0.85,
            coherence=0.88,
            accuracy=0.82,
            citation_quality=0.80,
            overall_score=0.84
        )
        
        should_continue = await quality_checker.should_continue_iteration(
            high_quality_metrics, 2, sample_requirements
        )
        
        # Should stop because quality threshold is met
        assert should_continue is False
    
    @pytest.mark.asyncio
    async def test_should_continue_iteration_max_iterations(self, quality_checker, sample_quality_metrics, sample_requirements):
        """Test continuation decision at max iterations"""
        should_continue = await quality_checker.should_continue_iteration(
            sample_quality_metrics, 5, sample_requirements  # At max iterations
        )
        
        # Should stop because max iterations reached
        assert should_continue is False
    
    @pytest.mark.asyncio
    async def test_should_continue_iteration_with_api_error(self, quality_checker, sample_quality_metrics, sample_requirements):
        """Test continuation decision with Kimi K2 API error"""
        with patch.object(quality_checker.kimi_client, 'generate_structured_response', new_callable=AsyncMock) as mock_generate:
            mock_generate.side_effect = KimiK2Error("API error")
            
            should_continue = await quality_checker.should_continue_iteration(
                sample_quality_metrics, 2, sample_requirements
            )
            
            # Should use fallback logic
            assert isinstance(should_continue, bool)
    
    def test_build_continuation_prompt(self, quality_checker, sample_quality_metrics, sample_requirements):
        """Test continuation prompt building"""
        prompt = quality_checker._build_continuation_prompt(
            sample_quality_metrics, 2, sample_requirements
        )
        
        assert "Overall Score: 0.725" in prompt
        assert "Current Iteration: 2" in prompt
        assert "Max Iterations: 5" in prompt
        assert "Quality Threshold: 0.8" in prompt
        assert "continue iteration" in prompt.lower()
    
    def test_fallback_continuation_decision(self, quality_checker, sample_quality_metrics, sample_requirements):
        """Test fallback continuation decision logic"""
        # Test continuation when below threshold with iterations left
        should_continue = quality_checker._fallback_continuation_decision(
            sample_quality_metrics, 2, sample_requirements
        )
        assert should_continue is True
        
        # Test stopping when at max iterations
        should_continue = quality_checker._fallback_continuation_decision(
            sample_quality_metrics, 5, sample_requirements
        )
        assert should_continue is False
        
        # Test stopping when above threshold
        high_quality_metrics = QualityMetrics(overall_score=0.85)
        should_continue = quality_checker._fallback_continuation_decision(
            high_quality_metrics, 2, sample_requirements
        )
        assert should_continue is False

class TestQualityAssessmentIntegration:
    """Integration tests for quality assessment system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_quality_assessment(self):
        """Test complete quality assessment workflow"""
        # Create test data
        structure = ResearchStructure(
            sections=[
                Section(id="intro", title="Introduction"),
                Section(id="body", title="Main Content"),
                Section(id="conclusion", title="Conclusion")
            ],
            estimated_length=1500,
            complexity_level=ComplexityLevel.INTERMEDIATE,
            domain=ResearchDomain.GENERAL
        )
        
        draft = Draft(
            id="integration-test",
            topic="Test Research Topic",
            structure=structure,
            content={
                "intro": "This is a comprehensive introduction to the topic...",
                "body": "The main content provides detailed analysis and findings...",
                "conclusion": "In conclusion, this research demonstrates..."
            }
        )
        
        requirements = ResearchRequirements(
            quality_threshold=0.75,
            max_iterations=3
        )
        
        # Test quality assessment
        assessor = KimiK2QualityAssessor()
        
        with patch.object(assessor.kimi_client, 'generate_structured_response', new_callable=AsyncMock) as mock_generate:
            # Mock responses for each assessment dimension
            mock_generate.side_effect = [
                {"completeness_score": 0.8},
                {"coherence_score": 0.75},
                {"accuracy_score": 0.7},
                {"citation_score": 0.6}
            ]
            
            quality_metrics = await assessor.evaluate_draft(draft, requirements)
            
            # Verify comprehensive assessment
            assert quality_metrics.completeness == 0.8
            assert quality_metrics.coherence == 0.75
            assert quality_metrics.accuracy == 0.7
            assert quality_metrics.citation_quality == 0.6
            assert 0.0 <= quality_metrics.overall_score <= 1.0
        
        # Test continuation decision
        checker = KimiK2QualityChecker()
        
        with patch.object(checker.kimi_client, 'generate_structured_response', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = {
                "should_continue": True,
                "reasoning": "Quality can be improved with additional iteration"
            }
            
            should_continue = await checker.should_continue_iteration(
                quality_metrics, 1, requirements
            )
            
            assert isinstance(should_continue, bool)

if __name__ == "__main__":
    pytest.main([__file__])