"""
Integration test for report synthesis system with Kimi K2 model.
Tests the complete report synthesis workflow including node and service integration.
"""

import sys
sys.path.insert(0, '..')
import asyncio
from unittest.mock import AsyncMock, patch

from .workflow.report_synthesizer_node import report_synthesizer_node, get_synthesis_summary, assess_synthesis_quality
from .services.kimi_k2_report_synthesizer import KimiK2ReportSynthesizer
from .services.kimi_k2_client import KimiK2Response
from .models.core import (
    TTDRState, Draft, QualityMetrics, EvolutionRecord, ResearchRequirements,
    ResearchStructure, Section, DraftMetadata, ResearchDomain, ComplexityLevel
)

def create_comprehensive_test_state():
    """Create a comprehensive test state for integration testing"""
    # Create research structure
    structure = ResearchStructure(
        sections=[
            Section(id="executive_summary", title="Executive Summary", content="Summary content"),
            Section(id="introduction", title="Introduction", content="Introduction content"),
            Section(id="methodology", title="Methodology", content="Methods content"),
            Section(id="results", title="Results and Analysis", content="Results content"),
            Section(id="discussion", title="Discussion", content="Discussion content"),
            Section(id="conclusion", title="Conclusion", content="Conclusion content"),
            Section(id="references", title="References", content="References content")
        ],
        estimated_length=2500,
        complexity_level=ComplexityLevel.ADVANCED,
        domain=ResearchDomain.TECHNOLOGY
    )
    
    # Create comprehensive draft
    draft = Draft(
        id="comprehensive_test_draft",
        topic="The Impact of Large Language Models on Software Development Practices",
        structure=structure,
        content={
            "executive_summary": "This research examines how Large Language Models (LLMs) are transforming software development practices across the industry...",
            "introduction": "Large Language Models have emerged as powerful tools that are reshaping how developers write, test, and maintain code...",
            "methodology": "We conducted a comprehensive analysis using mixed methods including surveys, interviews, and code analysis...",
            "results": "Our findings indicate that LLMs improve developer productivity by 35% on average while maintaining code quality...",
            "discussion": "The implications of LLM adoption extend beyond productivity gains to fundamental changes in development workflows...",
            "conclusion": "LLMs represent a paradigm shift in software development, requiring new approaches to education and practice...",
            "references": "[1] Brown, T. et al. (2020). Language Models are Few-Shot Learners. [2] Chen, M. et al. (2021). Evaluating Large Language Models..."
        },
        metadata=DraftMetadata(word_count=1850),
        quality_score=0.82,
        iteration=4
    )
    
    # Create quality metrics
    quality_metrics = QualityMetrics(
        completeness=0.85,
        coherence=0.80,
        accuracy=0.88,
        citation_quality=0.75,
        overall_score=0.82
    )
    
    # Create evolution history
    evolution_history = [
        EvolutionRecord(
            component="draft_generator",
            improvement_type="content_enhancement",
            description="Enhanced technical depth and clarity",
            performance_before=0.65,
            performance_after=0.75,
            parameters_changed={"temperature": 0.3, "max_tokens": 2000}
        ),
        EvolutionRecord(
            component="gap_analyzer",
            improvement_type="gap_detection",
            description="Improved identification of methodological gaps",
            performance_before=0.70,
            performance_after=0.80,
            parameters_changed={"analysis_depth": "comprehensive"}
        ),
        EvolutionRecord(
            component="information_integrator",
            improvement_type="coherence_improvement",
            description="Better integration of technical concepts",
            performance_before=0.72,
            performance_after=0.82,
            parameters_changed={"integration_strategy": "contextual"}
        )
    ]
    
    # Create research requirements
    requirements = ResearchRequirements(
        domain=ResearchDomain.TECHNOLOGY,
        complexity_level=ComplexityLevel.ADVANCED,
        max_iterations=5,
        quality_threshold=0.85,
        max_sources=25,
        preferred_source_types=["academic", "industry", "technical"]
    )
    
    return {
        "topic": draft.topic,
        "requirements": requirements,
        "current_draft": draft,
        "information_gaps": [],
        "retrieved_info": [],
        "iteration_count": 4,
        "quality_metrics": quality_metrics,
        "evolution_history": evolution_history,
        "final_report": None,
        "error_log": []
    }

def test_comprehensive_report_synthesis():
    """Test comprehensive report synthesis with mocked Kimi K2 responses"""
    state = create_comprehensive_test_state()
    
    # Mock Kimi K2 responses for synthesis
    synthesis_response = KimiK2Response(
        content="""# The Impact of Large Language Models on Software Development Practices

**Research Report**  
Generated by TTD-DR Framework  
Date: December 2024

## Executive Summary

This comprehensive research report examines the transformative impact of Large Language Models (LLMs) on software development practices. Through systematic analysis of industry adoption patterns, developer productivity metrics, and code quality assessments, we demonstrate that LLMs are fundamentally reshaping how software is conceived, developed, and maintained.

Key findings include a 35% average improvement in developer productivity, significant changes in code review processes, and the emergence of new development paradigms centered around AI-assisted programming. However, challenges remain in areas of code security, dependency management, and the need for new educational frameworks.

## Introduction

Large Language Models have emerged as transformative technologies in software development, offering unprecedented capabilities in code generation, debugging, and documentation. This research investigates their comprehensive impact on development practices, team dynamics, and software quality outcomes.

The rapid adoption of tools like GitHub Copilot, ChatGPT, and specialized coding assistants has created a new landscape where human-AI collaboration is becoming the norm rather than the exception. Understanding these changes is crucial for organizations, educators, and developers navigating this technological shift.

## Methodology

Our research employed a mixed-methods approach combining quantitative analysis of development metrics with qualitative insights from developer interviews. We analyzed code repositories, productivity data, and conducted surveys across 500+ developers from various organizations and experience levels.

The methodology included:
- Longitudinal analysis of development velocity and code quality metrics
- Structured interviews with development teams using LLM tools
- Comparative analysis of pre- and post-LLM adoption workflows
- Assessment of code security and maintainability implications

## Results and Analysis

Our comprehensive analysis reveals significant positive impacts of LLM adoption on software development practices:

**Productivity Improvements:**
- 35% average increase in code completion speed
- 28% reduction in debugging time
- 42% improvement in documentation quality
- 25% faster onboarding for new team members

**Code Quality Metrics:**
- Maintained or improved code quality in 78% of cases
- Reduced common bug patterns by 31%
- Improved code consistency across teams
- Enhanced test coverage through AI-assisted test generation

**Workflow Transformations:**
- Shift from traditional coding to prompt engineering and code review
- Evolution of pair programming to human-AI collaboration
- New roles emerging around AI tool optimization and governance
- Changes in code review focus from syntax to logic and architecture

## Discussion

The implications of LLM adoption extend far beyond simple productivity gains. We observe fundamental shifts in how developers approach problem-solving, with increased emphasis on high-level design and reduced focus on implementation details.

**Positive Impacts:**
- Democratization of programming capabilities
- Acceleration of prototyping and experimentation
- Improved accessibility for developers with disabilities
- Enhanced learning opportunities through AI explanations

**Challenges and Concerns:**
- Potential over-reliance on AI-generated code
- Security vulnerabilities in generated code
- Intellectual property and licensing concerns
- Need for new code review and quality assurance practices

**Future Implications:**
The trajectory suggests continued integration of LLMs into development workflows, with potential for even more sophisticated AI assistance in system design, architecture decisions, and project management.

## Conclusion

Large Language Models represent a paradigm shift in software development comparable to the introduction of high-level programming languages or integrated development environments. While productivity gains are substantial and immediate, the long-term implications require careful consideration and proactive adaptation.

Organizations must invest in new training programs, establish governance frameworks for AI tool usage, and develop strategies for maintaining code quality and security in an AI-assisted development environment. The future of software development will likely be characterized by seamless human-AI collaboration, requiring developers to evolve their skills toward higher-level problem-solving and AI interaction.

The research demonstrates that successful LLM adoption requires more than tool deployment—it demands organizational change, new processes, and a fundamental rethinking of development practices. Organizations that proactively address these challenges while leveraging LLM capabilities will gain significant competitive advantages in software development efficiency and innovation.

## References

[1] Brown, T. et al. (2020). Language Models are Few-Shot Learners. Advances in Neural Information Processing Systems.
[2] Chen, M. et al. (2021). Evaluating Large Language Models Trained on Code. arXiv preprint arXiv:2107.03374.
[3] GitHub. (2022). Research: Quantifying GitHub Copilot's impact on developer productivity and happiness.
[4] Stack Overflow. (2023). Developer Survey: AI and Machine Learning Trends.
[5] IEEE Software. (2023). Special Issue: AI-Assisted Software Development.

---

*Report generated using Test-Time Diffusion Deep Researcher (TTD-DR) Framework*  
*Framework Version: 1.0*  
*Quality Score: 0.89*  
*Iterations Completed: 4*  
*Evolution Cycles: 3*
""",
        usage={"prompt_tokens": 2500, "completion_tokens": 4000},
        model="kimi-k2",
        finish_reason="stop"
    )
    
    # Mock formatting response
    formatting_response = KimiK2Response(
        content=synthesis_response.content,  # Same content, properly formatted
        usage={"prompt_tokens": 1000, "completion_tokens": 500},
        model="kimi-k2",
        finish_reason="stop"
    )
    
    # Mock validation response
    validation_response = {
        "completeness": 0.92,
        "coherence": 0.88,
        "formatting": 0.95,
        "accuracy": 0.85,
        "overall_quality": 0.90,
        "improvement_over_draft": 0.08,
        "strengths": [
            "Comprehensive coverage of topic",
            "Clear structure and flow",
            "Strong evidence-based conclusions",
            "Professional formatting"
        ],
        "areas_for_improvement": [
            "Could include more technical implementation details",
            "Additional quantitative metrics would strengthen findings"
        ],
        "recommendation": "accept"
    }
    
    # Mock executive summary response
    summary_response = KimiK2Response(
        content="This research demonstrates that Large Language Models are fundamentally transforming software development practices, delivering 35% productivity improvements while requiring new approaches to code quality, security, and developer education. The study reveals both significant opportunities and important challenges in AI-assisted development workflows.",
        usage={"prompt_tokens": 500, "completion_tokens": 150},
        model="kimi-k2",
        finish_reason="stop"
    )
    
    # Apply mocks and run synthesis
    with patch('backend.workflow.report_synthesizer_node.KimiK2ReportSynthesizer') as mock_synthesizer_class:
        mock_synthesizer = AsyncMock()
        mock_synthesizer_class.return_value = mock_synthesizer
        
        # Configure mock methods
        mock_synthesizer.synthesize_report.return_value = synthesis_response.content
        mock_synthesizer.validate_report_quality.return_value = validation_response
        mock_synthesizer.generate_executive_summary.return_value = summary_response.content
        
        # Execute report synthesis
        result = report_synthesizer_node(state)
        
        # Verify synthesis was called
        mock_synthesizer.synthesize_report.assert_called_once()
        mock_synthesizer.validate_report_quality.assert_called_once()
        mock_synthesizer.generate_executive_summary.assert_called_once()
        
        # Verify result structure
        assert "final_report" in result
        assert result["final_report"] is not None
        assert len(result["final_report"]) > 0
        
        # Verify content quality
        final_report = result["final_report"]
        assert "Large Language Models" in final_report
        assert "Executive Summary" in final_report
        assert "Conclusion" in final_report
        assert "References" in final_report
        assert "TTD-DR Framework" in final_report
        
        # Verify synthesis metadata
        assert "synthesis_metadata" in result
        metadata = result["synthesis_metadata"]
        assert metadata["synthesis_method"] == "kimi_k2_powered"
        assert "validation_results" in metadata
        assert "executive_summary" in metadata
        
        # Verify validation results
        validation = metadata["validation_results"]
        assert validation["overall_quality"] >= 0.85
        assert validation["recommendation"] == "accept"
        assert len(validation["strengths"]) > 0

def test_synthesis_summary_generation():
    """Test synthesis summary generation"""
    state = create_comprehensive_test_state()
    
    # Add synthesis metadata to state
    state["final_report"] = "# Test Report\n\nThis is a test report with comprehensive content..."
    state["synthesis_metadata"] = {
        "synthesis_method": "kimi_k2_powered",
        "validation_results": {
            "overall_quality": 0.88,
            "improvement_over_draft": 0.06,
            "recommendation": "accept"
        },
        "executive_summary": "This is a test executive summary."
    }
    
    summary = get_synthesis_summary(state)
    
    assert summary["has_final_report"] is True
    assert summary["report_length"] > 0
    assert summary["synthesis_method"] == "kimi_k2_powered"
    assert "validation_summary" in summary
    assert summary["validation_summary"]["overall_quality"] == 0.88
    assert summary["validation_summary"]["recommendation"] == "accept"

def test_synthesis_quality_assessment():
    """Test synthesis quality assessment"""
    final_report = """# Comprehensive Research Report

## Executive Summary
This report provides comprehensive analysis of the research topic with detailed findings and recommendations.

## Introduction
The introduction sets the context and establishes the research framework for comprehensive analysis.

## Methodology
Our methodology employed rigorous analytical approaches to ensure comprehensive coverage.

## Results
The results demonstrate significant findings across multiple dimensions of analysis.

## Discussion
The discussion integrates findings with existing literature and provides comprehensive insights.

## Conclusion
In conclusion, this research provides comprehensive understanding of the topic with actionable recommendations.

## References
[1] Author, A. (2023). Comprehensive Research Methods.
[2] Researcher, B. (2024). Advanced Analysis Techniques.
"""
    
    # Create mock draft for comparison
    state = create_comprehensive_test_state()
    original_draft = state["current_draft"]
    
    assessment = assess_synthesis_quality(final_report, original_draft)
    
    assert assessment["status"] == "assessed"
    assert assessment["quality_score"] > 0.7
    assert assessment["report_length"] > 100
    assert assessment["has_structure"] is True
    assert assessment["has_conclusion"] is True
    assert len(assessment["issues"]) >= 0
    assert len(assessment["recommendations"]) > 0

def test_synthesis_error_handling():
    """Test synthesis error handling and fallback mechanisms"""
    state = create_comprehensive_test_state()
    
    # Mock synthesis failure
    with patch('backend.workflow.report_synthesizer_node.KimiK2ReportSynthesizer') as mock_synthesizer_class:
        mock_synthesizer = AsyncMock()
        mock_synthesizer_class.return_value = mock_synthesizer
        
        # Configure mock to raise exception
        mock_synthesizer.synthesize_report.side_effect = Exception("Synthesis failed")
        
        # Execute report synthesis
        result = report_synthesizer_node(state)
        
        # Verify fallback report was generated
        assert "final_report" in result
        assert result["final_report"] is not None
        assert "Emergency Generation" in result["final_report"] or "Fallback Generation" in result["final_report"]
        assert state["current_draft"].topic in result["final_report"]
        
        # Verify error logging
        assert "error_log" in result
        assert len(result["error_log"]) > 0
        assert "Report synthesis error" in str(result["error_log"])
        
        # Verify synthesis metadata indicates fallback
        assert "synthesis_metadata" in result
        assert result["synthesis_metadata"]["synthesis_method"] == "emergency_fallback"

if __name__ == "__main__":
    print("Running comprehensive report synthesis integration tests...")
    
    test_comprehensive_report_synthesis()
    print("✓ Comprehensive report synthesis test passed")
    
    test_synthesis_summary_generation()
    print("✓ Synthesis summary generation test passed")
    
    test_synthesis_quality_assessment()
    print("✓ Synthesis quality assessment test passed")
    
    test_synthesis_error_handling()
    print("✓ Synthesis error handling test passed")
    
    print("\nAll integration tests passed successfully!")