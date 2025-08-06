"""
Tests for Kimi K2-powered research methodology documentation functionality.
Tests Task 9.2 implementation: research process logging, bibliography generation, and methodology documentation.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import Dict, Any, List

from services.kimi_k2_report_synthesizer import KimiK2ReportSynthesizer
from models.core import (
    Draft, QualityMetrics, EvolutionRecord, ResearchRequirements,
    Source, RetrievedInfo, InformationGap, ResearchStructure, Section,
    Domain, ComplexityLevel, Priority, GapType
)

class TestKimiK2MethodologyDocumentation:
    """Test suite for Kimi K2-powered methodology documentation features"""
    
    @pytest.fixture
    def mock_kimi_client(self):
        """Mock Kimi K2 client for testing"""
        client = Mock()
        client.generate_text = AsyncMock()
        client.generate_structured_response = AsyncMock()
        return client
    
    @pytest.fixture
    def report_synthesizer(self, mock_kimi_client):
        """Create report synthesizer with mocked Kimi K2 client"""
        synthesizer = KimiK2ReportSynthesizer()
        synthesizer.kimi_client = mock_kimi_client
        return synthesizer
    
    @pytest.fixture
    def sample_state(self):
        """Create sample workflow state for testing"""
        # Create sample research structure
        sections = [
            Section(
                id="intro",
                title="Introduction",
                subsections=[],
                estimated_length=200,
                priority=Priority.HIGH
            ),
            Section(
                id="analysis",
                title="Analysis",
                subsections=[],
                estimated_length=500,
                priority=Priority.HIGH
            )
        ]
        
        structure = ResearchStructure(
            sections=sections,
            relationships=[],
            estimated_length=1000,
            complexity_level=ComplexityLevel.MEDIUM,
            domain=Domain.TECHNOLOGY
        )
        
        # Create sample draft
        draft = Draft(
            id="test-draft",
            topic="Artificial Intelligence in Healthcare",
            structure=structure,
            content={
                "intro": "Introduction to AI in healthcare...",
                "analysis": "Analysis of AI applications..."
            },
            metadata={},
            quality_score=0.75,
            iteration=3
        )
        
        # Create sample quality metrics
        quality_metrics = QualityMetrics(
            overall_score=0.78,
            completeness=0.80,
            coherence=0.75,
            accuracy=0.82,
            citation_quality=0.70
        )
        
        # Create sample evolution history
        evolution_history = [
            EvolutionRecord(
                component="gap_analyzer",
                improvement_type="query_optimization",
                performance_before=0.65,
                performance_after=0.72,
                description="Improved search query generation accuracy"
            ),
            EvolutionRecord(
                component="information_integrator",
                improvement_type="coherence_enhancement",
                performance_before=0.70,
                performance_after=0.78,
                description="Enhanced content integration coherence"
            )
        ]
        
        # Create sample retrieved information
        source1 = Source(
            url="https://example.com/ai-healthcare-1",
            title="AI Applications in Medical Diagnosis",
            author="Dr. Jane Smith",
            publication_date="2023-01-15",
            domain="healthcare.example.com"
        )
        
        retrieved_info = [
            RetrievedInfo(
                source=source1,
                content="AI systems are increasingly used in medical diagnosis...",
                relevance_score=0.85,
                credibility_score=0.90,
                extraction_timestamp=datetime.now()
            )
        ]
        
        # Create sample information gaps
        information_gaps = [
            InformationGap(
                id="gap-1",
                section_id="analysis",
                gap_type=GapType.EVIDENCE,
                description="Need more evidence on AI diagnostic accuracy",
                priority=Priority.HIGH,
                search_queries=["AI diagnostic accuracy studies", "medical AI performance metrics"]
            )
        ]
        
        return {
            "topic": "Artificial Intelligence in Healthcare",
            "current_draft": draft,
            "quality_metrics": quality_metrics,
            "evolution_history": evolution_history,
            "retrieved_info": retrieved_info,
            "information_gaps": information_gaps,
            "iteration_count": 3,
            "requirements": ResearchRequirements(
                domain=Domain.TECHNOLOGY,
                complexity_level=ComplexityLevel.MEDIUM,
                quality_threshold=0.75,
                preferred_source_types=["academic", "news"]
            )
        }
    
    @pytest.fixture
    def sample_workflow_log(self):
        """Create sample workflow execution log"""
        return [
            {
                "timestamp": 1000.0,
                "node": "draft_generator",
                "action": "generate_initial_draft",
                "status": "completed"
            },
            {
                "timestamp": 1010.0,
                "node": "gap_analyzer",
                "action": "identify_gaps",
                "status": "completed"
            },
            {
                "timestamp": 1020.0,
                "node": "retrieval_engine",
                "action": "retrieve_information",
                "status": "completed"
            },
            {
                "timestamp": 1030.0,
                "node": "report_synthesizer",
                "action": "synthesize_report",
                "status": "completed"
            }
        ]

    @pytest.mark.asyncio
    async def test_generate_research_methodology_documentation_success(self, report_synthesizer, sample_state, sample_workflow_log):
        """Test successful generation of research methodology documentation"""
        # Mock Kimi K2 responses
        methodology_content = """# Research Methodology Documentation

## Research Framework Overview
The TTD-DR (Test-Time Diffusion Deep Researcher) framework was employed for this research...

## Research Process Documentation
The research process involved multiple iterative cycles...

## Information Retrieval Methodology
Search strategies were developed using AI-powered query generation...

## Quality Assurance Framework
Comprehensive quality metrics were applied throughout the process...
"""
        
        enhanced_content = f"{methodology_content}\n\n## Technical Appendix\n\nDetailed technical specifications..."
        
        report_synthesizer.kimi_client.generate_text.side_effect = [
            Mock(content=methodology_content),  # Initial methodology generation
            Mock(content=enhanced_content)      # Enhanced methodology
        ]
        
        # Execute methodology documentation generation
        result = await report_synthesizer.generate_research_methodology_documentation(
            sample_state, sample_workflow_log
        )
        
        # Verify results
        assert result is not None
        assert len(result) > 0
        assert "Research Methodology Documentation" in result
        assert "TTD-DR" in result
        assert "Technical Appendix" in result
        
        # Verify Kimi K2 client was called correctly
        assert report_synthesizer.kimi_client.generate_text.call_count == 2
        
        # Check first call (methodology generation)
        first_call = report_synthesizer.kimi_client.generate_text.call_args_list[0]
        assert "methodology documentation" in first_call[0][0].lower()
        assert first_call[1]["max_tokens"] == 4000
        assert first_call[1]["temperature"] == 0.2
        
        # Check second call (enhancement)
        second_call = report_synthesizer.kimi_client.generate_text.call_args_list[1]
        assert "enhance this research methodology" in second_call[0][0].lower()

    @pytest.mark.asyncio
    async def test_generate_research_methodology_documentation_fallback(self, report_synthesizer, sample_state):
        """Test fallback methodology documentation generation when Kimi K2 fails"""
        # Mock Kimi K2 failure
        report_synthesizer.kimi_client.generate_text.side_effect = Exception("Kimi K2 API error")
        
        # Execute methodology documentation generation
        result = await report_synthesizer.generate_research_methodology_documentation(sample_state)
        
        # Verify fallback result
        assert result is not None
        assert len(result) > 0
        assert "Research Methodology Documentation" in result
        assert "Fallback Documentation" in result
        assert "TTD-DR" in result
        assert sample_state["topic"] in result

    @pytest.mark.asyncio
    async def test_generate_source_bibliography_apa_style(self, report_synthesizer, sample_state):
        """Test APA style bibliography generation"""
        # Mock Kimi K2 response for bibliography
        bibliography_content = """# Bibliography (APA Style)

Smith, J. (2023, January 15). AI Applications in Medical Diagnosis. Retrieved December 15, 2023, from https://example.com/ai-healthcare-1

Johnson, M. (n.d.). Machine Learning in Healthcare. Retrieved December 15, 2023, from https://example.com/ml-healthcare
"""
        
        enhanced_bibliography = f"{bibliography_content}\n\n---\n\n*Bibliography generated using TTD-DR Framework*"
        
        report_synthesizer.kimi_client.generate_text.side_effect = [
            Mock(content=bibliography_content),  # Initial bibliography
            Mock(content=enhanced_bibliography)  # Enhanced bibliography
        ]
        
        # Execute bibliography generation
        result = await report_synthesizer.generate_source_bibliography(
            sample_state["retrieved_info"], citation_style="APA"
        )
        
        # Verify results
        assert result is not None
        assert "Bibliography (APA Style)" in result
        assert "Smith, J." in result
        assert "TTD-DR Framework" in result
        
        # Verify Kimi K2 calls
        assert report_synthesizer.kimi_client.generate_text.call_count == 2
        
        # Check bibliography generation call
        first_call = report_synthesizer.kimi_client.generate_text.call_args_list[0]
        assert "APA citation style" in first_call[0][0]
        assert first_call[1]["temperature"] == 0.1

    @pytest.mark.asyncio
    async def test_generate_source_bibliography_empty_sources(self, report_synthesizer):
        """Test bibliography generation with no sources"""
        result = await report_synthesizer.generate_source_bibliography([], citation_style="APA")
        
        assert result is not None
        assert "No sources were retrieved" in result

    @pytest.mark.asyncio
    async def test_generate_source_bibliography_fallback(self, report_synthesizer, sample_state):
        """Test fallback bibliography generation when Kimi K2 fails"""
        # Mock Kimi K2 failure
        report_synthesizer.kimi_client.generate_text.side_effect = Exception("Kimi K2 API error")
        
        # Execute bibliography generation
        result = await report_synthesizer.generate_source_bibliography(
            sample_state["retrieved_info"], citation_style="APA"
        )
        
        # Verify fallback result
        assert result is not None
        assert "Bibliography (APA Style)" in result
        assert "Fallback" in result
        assert len(sample_state["retrieved_info"]) > 0  # Ensure we have sources to test with

    @pytest.mark.asyncio
    async def test_generate_methodology_summary(self, report_synthesizer, sample_state):
        """Test methodology summary generation"""
        # Mock Kimi K2 response
        summary_content = """## Methodology Summary

This research on "Artificial Intelligence in Healthcare" was conducted using the TTD-DR framework, employing an iterative diffusion-inspired approach. The methodology involved 3 iterations of refinement, identifying 1 information gaps and retrieving 1 external sources. Quality assurance maintained a final score of 0.780, with 2 self-evolution cycles applied for continuous improvement.

The TTD-DR methodology ensures research rigor through systematic gap analysis, comprehensive quality assessment, and adaptive learning mechanisms, providing transparency and reproducibility in automated research generation.
"""
        
        report_synthesizer.kimi_client.generate_text.return_value = Mock(content=summary_content)
        
        # Execute methodology summary generation
        result = await report_synthesizer.generate_methodology_summary(sample_state)
        
        # Verify results
        assert result is not None
        assert "Methodology Summary" in result
        assert "TTD-DR" in result
        assert sample_state["topic"] in result
        assert "3 iterations" in result
        
        # Verify Kimi K2 call
        report_synthesizer.kimi_client.generate_text.assert_called_once()
        call_args = report_synthesizer.kimi_client.generate_text.call_args
        assert "methodology summary" in call_args[0][0].lower()
        assert call_args[1]["max_tokens"] == 1000
        assert call_args[1]["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_generate_methodology_summary_fallback(self, report_synthesizer, sample_state):
        """Test fallback methodology summary generation"""
        # Mock Kimi K2 failure
        report_synthesizer.kimi_client.generate_text.side_effect = Exception("Kimi K2 API error")
        
        # Execute methodology summary generation
        result = await report_synthesizer.generate_methodology_summary(sample_state)
        
        # Verify fallback result
        assert result is not None
        assert "Methodology Summary" in result
        assert "TTD-DR" in result
        assert sample_state["topic"] in result

    def test_extract_methodology_data(self, report_synthesizer, sample_state, sample_workflow_log):
        """Test extraction of methodology data from state and workflow log"""
        methodology_data = report_synthesizer._extract_methodology_data(sample_state, sample_workflow_log)
        
        # Verify extracted data
        assert methodology_data["research_topic"] == sample_state["topic"]
        assert methodology_data["iteration_count"] == sample_state["iteration_count"]
        assert methodology_data["methodology_approach"] == "TTD-DR (Test-Time Diffusion Deep Researcher)"
        
        # Verify workflow stages
        assert len(methodology_data["workflow_stages"]) > 0
        stage_names = [stage["stage"] for stage in methodology_data["workflow_stages"]]
        assert "Draft Generation" in stage_names
        assert "Gap Analysis" in stage_names
        
        # Verify sources and queries
        assert len(methodology_data["sources_used"]) > 0
        assert len(methodology_data["search_queries"]) > 0
        
        # Verify execution time calculation
        assert methodology_data["workflow_execution_time"] is not None
        assert methodology_data["workflow_execution_time"] > 0

    def test_extract_unique_sources(self, report_synthesizer, sample_state):
        """Test extraction of unique sources from retrieved information"""
        unique_sources = report_synthesizer._extract_unique_sources(sample_state["retrieved_info"])
        
        assert len(unique_sources) > 0
        
        # Verify source data structure
        source = unique_sources[0]
        assert "title" in source
        assert "url" in source
        assert "credibility_score" in source
        assert "access_date" in source
        
        # Verify sorting by credibility score
        if len(unique_sources) > 1:
            for i in range(len(unique_sources) - 1):
                assert unique_sources[i]["credibility_score"] >= unique_sources[i + 1]["credibility_score"]

    def test_calculate_execution_time(self, report_synthesizer, sample_workflow_log):
        """Test calculation of workflow execution time"""
        execution_time = report_synthesizer._calculate_execution_time(sample_workflow_log)
        
        assert execution_time is not None
        assert execution_time > 0
        assert execution_time == 30.0  # 1030.0 - 1000.0

    def test_calculate_execution_time_empty_log(self, report_synthesizer):
        """Test execution time calculation with empty log"""
        execution_time = report_synthesizer._calculate_execution_time([])
        assert execution_time is None

    def test_extract_methodology_summary_points(self, report_synthesizer, sample_state):
        """Test extraction of methodology summary points"""
        summary_points = report_synthesizer._extract_methodology_summary_points(sample_state)
        
        assert summary_points["topic"] == sample_state["topic"]
        assert summary_points["framework"] == "TTD-DR (Test-Time Diffusion Deep Researcher)"
        assert summary_points["iterations"] == sample_state["iteration_count"]
        assert summary_points["sources_count"] == len(sample_state["retrieved_info"])
        assert summary_points["gaps_identified"] == len(sample_state["information_gaps"])
        assert summary_points["quality_score"] == sample_state["quality_metrics"].overall_score
        assert summary_points["evolution_cycles"] == len(sample_state["evolution_history"])

    @pytest.mark.asyncio
    async def test_multiple_citation_styles(self, report_synthesizer, sample_state):
        """Test bibliography generation with different citation styles"""
        citation_styles = ["APA", "MLA", "Chicago"]
        
        for style in citation_styles:
            # Mock Kimi K2 response for each style
            bibliography_content = f"# Bibliography ({style} Style)\n\nSample citation in {style} format..."
            report_synthesizer.kimi_client.generate_text.return_value = Mock(content=bibliography_content)
            
            result = await report_synthesizer.generate_source_bibliography(
                sample_state["retrieved_info"], citation_style=style
            )
            
            assert result is not None
            assert f"Bibliography ({style} Style)" in result

    def test_generate_technical_appendix(self, report_synthesizer, sample_state, sample_workflow_log):
        """Test generation of technical appendix"""
        methodology_data = report_synthesizer._extract_methodology_data(sample_state, sample_workflow_log)
        appendix = report_synthesizer._generate_technical_appendix(methodology_data)
        
        assert appendix is not None
        assert "Technical Appendix" in appendix
        assert "Framework Parameters" in appendix
        assert "Workflow Execution Details" in appendix
        assert "Information Sources Summary" in appendix
        assert "Quality Metrics Details" in appendix
        assert "Self-Evolution Enhancement Log" in appendix
        assert "Reproducibility Information" in appendix
        
        # Verify specific content
        assert sample_state["topic"] in appendix
        assert str(sample_state["iteration_count"]) in appendix

    @pytest.mark.asyncio
    async def test_methodology_documentation_completeness(self, report_synthesizer, sample_state, sample_workflow_log):
        """Test that methodology documentation includes all required components"""
        # Mock comprehensive Kimi K2 response
        comprehensive_methodology = """# Research Methodology Documentation

## Research Framework Overview
Detailed framework explanation...

## Research Process Documentation
Step-by-step process documentation...

## Information Retrieval Methodology
Search and retrieval strategies...

## Quality Assurance Framework
Quality metrics and validation...

## Self-Evolution Enhancement Process
Adaptive learning mechanisms...

## Limitations and Considerations
Methodological constraints...

## Reproducibility Guidelines
Replication instructions...
"""
        
        report_synthesizer.kimi_client.generate_text.return_value = Mock(content=comprehensive_methodology)
        
        result = await report_synthesizer.generate_research_methodology_documentation(
            sample_state, sample_workflow_log
        )
        
        # Verify all required sections are present
        required_sections = [
            "Research Framework Overview",
            "Research Process Documentation", 
            "Information Retrieval Methodology",
            "Quality Assurance Framework",
            "Self-Evolution Enhancement Process",
            "Limitations and Considerations",
            "Reproducibility Guidelines"
        ]
        
        for section in required_sections:
            assert section in result

    @pytest.mark.asyncio
    async def test_methodology_documentation_accuracy(self, report_synthesizer, sample_state):
        """Test accuracy of methodology documentation content"""
        # Mock Kimi K2 response
        report_synthesizer.kimi_client.generate_text.return_value = Mock(
            content="Methodology documentation with accurate data..."
        )
        
        result = await report_synthesizer.generate_research_methodology_documentation(sample_state)
        
        # Verify that the prompt contains accurate state information
        call_args = report_synthesizer.kimi_client.generate_text.call_args_list[0]
        prompt = call_args[0][0]
        
        # Check that key state information is included in the prompt
        assert sample_state["topic"] in prompt
        assert str(sample_state["iteration_count"]) in prompt
        assert str(len(sample_state["retrieved_info"])) in prompt
        assert str(len(sample_state["information_gaps"])) in prompt
        assert str(len(sample_state["evolution_history"])) in prompt

if __name__ == "__main__":
    pytest.main([__file__])