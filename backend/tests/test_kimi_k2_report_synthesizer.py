"""
Unit tests for Kimi K2 report synthesizer service.
Tests report synthesis, formatting, and quality assurance functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from backend.services.kimi_k2_report_synthesizer import KimiK2ReportSynthesizer
from backend.models.core import (
    Draft, QualityMetrics, EvolutionRecord, ResearchRequirements,
    ResearchStructure, Section, DraftMetadata, ResearchDomain, ComplexityLevel
)
from backend.services.kimi_k2_client import KimiK2Response, KimiK2Error

class TestKimiK2ReportSynthesizer:
    """Test cases for KimiK2ReportSynthesizer"""
    
    @pytest.fixture
    def synthesizer(self):
        """Create synthesizer instance for testing"""
        return KimiK2ReportSynthesizer()
    
    @pytest.fixture
    def sample_draft(self):
        """Create sample draft for testing"""
        structure = ResearchStructure(
            sections=[
                Section(id="intro", title="Introduction", content="Introduction content"),
                Section(id="methods", title="Methodology", content="Methods content"),
                Section(id="results", title="Results", content="Results content"),
                Section(id="conclusion", title="Conclusion", content="Conclusion content")
            ],
            estimated_length=1000,
            complexity_level=ComplexityLevel.INTERMEDIATE,
            domain=ResearchDomain.TECHNOLOGY
        )
        
        return Draft(
            id="test_draft_001",
            topic="Artificial Intelligence in Healthcare",
            structure=structure,
            content={
                "intro": "This research explores AI applications in healthcare...",
                "methods": "We conducted a comprehensive literature review...",
                "results": "Our analysis revealed significant benefits...",
                "conclusion": "AI shows great promise for healthcare transformation..."
            },
            metadata=DraftMetadata(word_count=500),
            quality_score=0.75,
            iteration=3
        )
    
    @pytest.fixture
    def sample_quality_metrics(self):
        """Create sample quality metrics for testing"""
        return QualityMetrics(
            completeness=0.8,
            coherence=0.75,
            accuracy=0.85,
            citation_quality=0.7,
            overall_score=0.775
        )
    
    @pytest.fixture
    def sample_evolution_history(self):
        """Create sample evolution history for testing"""
        return [
            EvolutionRecord(
                component="draft_generator",
                improvement_type="content_enhancement",
                description="Improved introduction clarity",
                performance_before=0.6,
                performance_after=0.7,
                parameters_changed={"temperature": 0.3}
            ),
            EvolutionRecord(
                component="gap_analyzer",
                improvement_type="gap_detection",
                description="Enhanced gap identification accuracy",
                performance_before=0.7,
                performance_after=0.75,
                parameters_changed={"analysis_depth": "detailed"}
            )
        ]
    
    @pytest.fixture
    def sample_requirements(self):
        """Create sample research requirements for testing"""
        return ResearchRequirements(
            domain=ResearchDomain.TECHNOLOGY,
            complexity_level=ComplexityLevel.INTERMEDIATE,
            max_iterations=5,
            quality_threshold=0.8,
            max_sources=15,
            preferred_source_types=["academic", "industry"]
        )

    @pytest.mark.asyncio
    async def test_synthesize_report_success(self, synthesizer, sample_draft, 
                                           sample_quality_metrics, sample_evolution_history,
                                           sample_requirements):
        """Test successful report synthesis"""
        # Mock Kimi K2 client response
        mock_response = KimiK2Response(
            content="""# Artificial Intelligence in Healthcare

## Executive Summary
This comprehensive research report examines the transformative potential of artificial intelligence in healthcare...

## Introduction
Artificial intelligence (AI) is revolutionizing healthcare delivery...

## Methodology
This research employed a systematic approach...

## Results
Our analysis demonstrates significant benefits...

## Conclusion
AI technologies show tremendous promise...

## References
[1] Smith, J. et al. (2023). AI in Healthcare...
""",
            usage={"prompt_tokens": 1000, "completion_tokens": 2000},
            model="kimi-k2",
            finish_reason="stop"
        )
        
        with patch.object(synthesizer.kimi_client, 'generate_text', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = mock_response
            
            result = await synthesizer.synthesize_report(
                sample_draft, sample_quality_metrics, sample_evolution_history, sample_requirements
            )
            
            # Verify result
            assert isinstance(result, str)
            assert len(result) > 0
            assert "Artificial Intelligence in Healthcare" in result
            assert "Executive Summary" in result
            assert "Conclusion" in result
            
            # Verify Kimi K2 was called
            mock_generate.assert_called()
            call_args = mock_generate.call_args
            assert "synthesis" in call_args[0][0].lower()
            assert sample_draft.topic in call_args[0][0]

    @pytest.mark.asyncio
    async def test_synthesize_report_kimi_error(self, synthesizer, sample_draft, 
                                              sample_quality_metrics, sample_evolution_history):
        """Test report synthesis with Kimi K2 error"""
        with patch.object(synthesizer.kimi_client, 'generate_text', new_callable=AsyncMock) as mock_generate:
            mock_generate.side_effect = KimiK2Error("API rate limit exceeded", 429, "rate_limit")
            
            result = await synthesizer.synthesize_report(
                sample_draft, sample_quality_metrics, sample_evolution_history
            )
            
            # Should return fallback report
            assert isinstance(result, str)
            assert len(result) > 0
            assert "Fallback Generation" in result
            assert sample_draft.topic in result

    @pytest.mark.asyncio
    async def test_generate_executive_summary(self, synthesizer):
        """Test executive summary generation"""
        sample_report = """# AI in Healthcare

## Introduction
Artificial intelligence is transforming healthcare through advanced diagnostics, personalized treatment plans, and improved patient outcomes. This technology enables healthcare providers to analyze vast amounts of medical data quickly and accurately.

## Key Findings
Our research shows that AI applications in radiology have improved diagnostic accuracy by 25%. Machine learning algorithms can detect patterns in medical imaging that human radiologists might miss.

## Conclusion
AI represents a significant advancement in healthcare technology with the potential to save lives and reduce costs."""
        
        mock_response = KimiK2Response(
            content="This research examines AI's transformative impact on healthcare delivery, demonstrating significant improvements in diagnostic accuracy and patient outcomes through advanced machine learning algorithms.",
            usage={"prompt_tokens": 500, "completion_tokens": 100},
            model="kimi-k2",
            finish_reason="stop"
        )
        
        with patch.object(synthesizer.kimi_client, 'generate_text', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = mock_response
            
            result = await synthesizer.generate_executive_summary(sample_report, max_length=300)
            
            # Verify result
            assert isinstance(result, str)
            assert len(result) > 0
            assert len(result.split()) <= 350  # Allow some flexibility
            assert "AI" in result or "artificial intelligence" in result.lower()
            
            # Verify Kimi K2 was called with summary prompt
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args
            assert "executive summary" in call_args[0][0].lower()

    @pytest.mark.asyncio
    async def test_validate_report_quality(self, synthesizer, sample_draft):
        """Test report quality validation"""
        sample_report = """# Artificial Intelligence in Healthcare

## Executive Summary
This comprehensive research report examines AI applications in healthcare...

## Introduction
AI is revolutionizing healthcare delivery through advanced diagnostics...

## Methodology
We conducted systematic analysis of AI implementations...

## Results
Our findings demonstrate significant improvements in patient outcomes...

## Conclusion
AI technologies show tremendous promise for healthcare transformation...
"""
        
        mock_validation = {
            "completeness": 0.85,
            "coherence": 0.8,
            "formatting": 0.9,
            "accuracy": 0.75,
            "overall_quality": 0.825,
            "improvement_over_draft": 0.075,
            "strengths": ["Comprehensive coverage", "Clear structure"],
            "areas_for_improvement": ["Citation integration", "Technical depth"],
            "recommendation": "accept"
        }
        
        with patch.object(synthesizer.kimi_client, 'generate_structured_response', new_callable=AsyncMock) as mock_structured:
            mock_structured.return_value = mock_validation
            
            result = await synthesizer.validate_report_quality(sample_report, sample_draft)
            
            # Verify result structure
            assert isinstance(result, dict)
            assert "completeness" in result
            assert "coherence" in result
            assert "overall_quality" in result
            assert "recommendation" in result
            assert result["recommendation"] in ["accept", "revise", "reject"]
            
            # Verify quality scores are in valid range
            for metric in ["completeness", "coherence", "formatting", "accuracy", "overall_quality"]:
                if metric in result:
                    assert 0.0 <= result[metric] <= 1.0

    def test_extract_structure_info(self, synthesizer, sample_draft):
        """Test structure information extraction"""
        result = synthesizer._extract_structure_info(sample_draft.structure)
        
        assert isinstance(result, str)
        assert "Sections" in result
        assert "Introduction" in result
        assert "Methodology" in result
        assert "Results" in result
        assert "Conclusion" in result
        assert str(sample_draft.structure.estimated_length) in result

    def test_extract_content_summary(self, synthesizer, sample_draft):
        """Test content summary extraction"""
        result = synthesizer._extract_content_summary(sample_draft.content)
        
        assert isinstance(result, str)
        assert "intro" in result
        assert "methods" in result
        assert "results" in result
        assert "conclusion" in result
        
        # Test empty content
        empty_result = synthesizer._extract_content_summary({})
        assert "No content available" in empty_result

    def test_extract_quality_summary(self, synthesizer, sample_quality_metrics):
        """Test quality summary extraction"""
        result = synthesizer._extract_quality_summary(sample_quality_metrics)
        
        assert isinstance(result, str)
        assert "Overall Score" in result
        assert "Completeness" in result
        assert "Coherence" in result
        assert "Accuracy" in result
        assert "Citation Quality" in result
        assert str(sample_quality_metrics.overall_score) in result

    def test_get_quality_grade(self, synthesizer):
        """Test quality grade conversion"""
        assert synthesizer._get_quality_grade(0.95) == "A (Excellent)"
        assert synthesizer._get_quality_grade(0.85) == "B (Good)"
        assert synthesizer._get_quality_grade(0.75) == "C (Satisfactory)"
        assert synthesizer._get_quality_grade(0.65) == "D (Needs Improvement)"
        assert synthesizer._get_quality_grade(0.45) == "F (Poor)"

    def test_generate_fallback_report(self, synthesizer, sample_draft, sample_quality_metrics):
        """Test fallback report generation"""
        result = synthesizer._generate_fallback_report(sample_draft, sample_quality_metrics)
        
        assert isinstance(result, str)
        assert sample_draft.topic in result
        assert "Fallback Generation" in result
        assert "TTD-DR Framework" in result
        assert str(sample_quality_metrics.overall_score) in result
        
        # Verify content sections are included
        for section_id, content in sample_draft.content.items():
            if content.strip():
                assert content in result