"""
Integration test for research methodology documentation functionality (Task 9.2).
Tests the complete workflow of methodology documentation generation, bibliography creation,
and research process logging using Kimi K2.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from backend.services.kimi_k2_report_synthesizer import KimiK2ReportSynthesizer
from backend.models.core import (
    Draft, QualityMetrics, EvolutionRecord, ResearchRequirements,
    Source, RetrievedInfo, InformationGap, ResearchStructure, Section,
    Domain, ComplexityLevel, Priority, GapType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_comprehensive_test_state() -> Dict[str, Any]:
    """Create comprehensive test state for methodology documentation testing"""
    
    # Create detailed research structure
    sections = [
        Section(
            id="introduction",
            title="Introduction to AI in Education",
            subsections=[
                Section(id="intro_background", title="Background", subsections=[], estimated_length=150, priority=Priority.HIGH),
                Section(id="intro_objectives", title="Research Objectives", subsections=[], estimated_length=100, priority=Priority.HIGH)
            ],
            estimated_length=300,
            priority=Priority.HIGH
        ),
        Section(
            id="literature_review",
            title="Literature Review",
            subsections=[
                Section(id="lit_current_state", title="Current State of AI in Education", subsections=[], estimated_length=400, priority=Priority.HIGH),
                Section(id="lit_challenges", title="Challenges and Limitations", subsections=[], estimated_length=300, priority=Priority.MEDIUM)
            ],
            estimated_length=800,
            priority=Priority.HIGH
        ),
        Section(
            id="analysis",
            title="Analysis and Discussion",
            subsections=[
                Section(id="analysis_benefits", title="Benefits of AI in Education", subsections=[], estimated_length=350, priority=Priority.HIGH),
                Section(id="analysis_implementation", title="Implementation Strategies", subsections=[], estimated_length=400, priority=Priority.HIGH)
            ],
            estimated_length=900,
            priority=Priority.HIGH
        ),
        Section(
            id="conclusion",
            title="Conclusion and Future Directions",
            subsections=[],
            estimated_length=250,
            priority=Priority.HIGH
        )
    ]
    
    structure = ResearchStructure(
        sections=sections,
        relationships=[],
        estimated_length=2250,
        complexity_level=ComplexityLevel.HIGH,
        domain=Domain.EDUCATION
    )
    
    # Create comprehensive draft with detailed content
    draft = Draft(
        id="ai-education-draft",
        topic="Artificial Intelligence Applications in Modern Education",
        structure=structure,
        content={
            "introduction": """Artificial Intelligence (AI) has emerged as a transformative force in modern education, offering unprecedented opportunities to personalize learning experiences, automate administrative tasks, and enhance educational outcomes. This research examines the current applications, benefits, challenges, and future potential of AI technologies in educational settings.""",
            
            "literature_review": """The integration of AI in education has been extensively studied across multiple dimensions. Current research indicates that AI-powered adaptive learning systems can improve student engagement by up to 40% while reducing learning time by 25%. Machine learning algorithms are being successfully deployed for automated essay scoring, intelligent tutoring systems, and predictive analytics for student performance.""",
            
            "analysis": """The analysis reveals that AI applications in education fall into several key categories: personalized learning platforms, intelligent tutoring systems, automated assessment tools, and administrative automation. Each category presents unique benefits and implementation challenges that must be carefully considered by educational institutions.""",
            
            "conclusion": """AI represents a significant opportunity to revolutionize education through personalized learning, improved efficiency, and enhanced educational outcomes. However, successful implementation requires careful consideration of ethical implications, teacher training, and infrastructure requirements."""
        },
        metadata={
            "creation_timestamp": datetime.now(),
            "last_updated": datetime.now(),
            "word_count": 1850,
            "section_count": 4
        },
        quality_score=0.82,
        iteration=5
    )
    
    # Create detailed quality metrics
    quality_metrics = QualityMetrics(
        overall_score=0.84,
        completeness=0.88,
        coherence=0.82,
        accuracy=0.86,
        citation_quality=0.78
    )
    
    # Create comprehensive evolution history
    evolution_history = [
        EvolutionRecord(
            component="draft_generator",
            improvement_type="structure_optimization",
            performance_before=0.65,
            performance_after=0.72,
            description="Improved research structure generation with better section organization"
        ),
        EvolutionRecord(
            component="gap_analyzer",
            improvement_type="gap_detection_accuracy",
            performance_before=0.70,
            performance_after=0.78,
            description="Enhanced information gap identification using contextual analysis"
        ),
        EvolutionRecord(
            component="retrieval_engine",
            improvement_type="query_optimization",
            performance_before=0.68,
            performance_after=0.75,
            description="Optimized search query generation for better source retrieval"
        ),
        EvolutionRecord(
            component="information_integrator",
            improvement_type="coherence_enhancement",
            performance_before=0.72,
            performance_after=0.80,
            description="Improved content integration maintaining logical flow"
        ),
        EvolutionRecord(
            component="quality_assessor",
            improvement_type="metric_calibration",
            performance_before=0.74,
            performance_after=0.82,
            description="Calibrated quality assessment metrics for better accuracy"
        )
    ]
    
    # Create diverse retrieved information sources
    sources = [
        Source(
            url="https://www.nature.com/articles/ai-education-2023",
            title="Artificial Intelligence in Education: A Comprehensive Review",
            author="Dr. Sarah Johnson, Prof. Michael Chen",
            publication_date="2023-03-15",
            domain="nature.com"
        ),
        Source(
            url="https://educationaltechnology.net/ai-personalized-learning",
            title="Personalized Learning Through AI: Current State and Future Prospects",
            author="Emily Rodriguez",
            publication_date="2023-05-22",
            domain="educationaltechnology.net"
        ),
        Source(
            url="https://www.sciencedirect.com/science/article/ai-tutoring-systems",
            title="Intelligent Tutoring Systems: A Meta-Analysis of Effectiveness",
            author="Dr. James Wilson, Dr. Lisa Park",
            publication_date="2023-01-10",
            domain="sciencedirect.com"
        ),
        Source(
            url="https://www.edweek.org/technology/ai-classroom-implementation",
            title="Implementing AI in the Classroom: Challenges and Solutions",
            author="Maria Gonzalez",
            publication_date="2023-04-08",
            domain="edweek.org"
        ),
        Source(
            url="https://www.tandfonline.com/ai-assessment-automation",
            title="Automated Assessment in Education: Benefits and Limitations",
            author="Dr. Robert Kim, Dr. Anna Thompson",
            publication_date="2023-02-28",
            domain="tandfonline.com"
        )
    ]
    
    retrieved_info = []
    for i, source in enumerate(sources):
        retrieved_info.append(RetrievedInfo(
            source=source,
            content=f"Content from {source.title}: This source provides valuable insights into AI applications in education, discussing implementation strategies, benefits, and challenges. The research methodology employed rigorous analysis of current AI technologies and their educational impact.",
            relevance_score=0.85 + (i * 0.02),  # Varying relevance scores
            credibility_score=0.88 + (i * 0.01),  # Varying credibility scores
            extraction_timestamp=datetime.now()
        ))
    
    # Create detailed information gaps
    information_gaps = [
        InformationGap(
            id="gap-ethical-considerations",
            section_id="analysis",
            gap_type=GapType.ANALYSIS,
            description="Need more detailed analysis of ethical considerations in AI education implementation",
            priority=Priority.HIGH,
            search_queries=[
                "AI ethics in education",
                "ethical implications AI classroom",
                "privacy concerns AI educational technology"
            ]
        ),
        InformationGap(
            id="gap-cost-benefit-analysis",
            section_id="analysis",
            gap_type=GapType.EVIDENCE,
            description="Require comprehensive cost-benefit analysis of AI implementation in schools",
            priority=Priority.MEDIUM,
            search_queries=[
                "cost benefit analysis AI education",
                "ROI artificial intelligence schools",
                "educational AI implementation costs"
            ]
        ),
        InformationGap(
            id="gap-teacher-training",
            section_id="literature_review",
            gap_type=GapType.CONTENT,
            description="Missing information on teacher training requirements for AI integration",
            priority=Priority.HIGH,
            search_queries=[
                "teacher training AI education",
                "professional development AI classroom",
                "educator preparation artificial intelligence"
            ]
        )
    ]
    
    # Create comprehensive workflow log
    workflow_log = [
        {
            "timestamp": 1000.0,
            "node": "draft_generator",
            "action": "generate_initial_draft",
            "status": "completed",
            "duration": 15.2,
            "details": "Generated initial research structure with 4 main sections"
        },
        {
            "timestamp": 1015.2,
            "node": "gap_analyzer",
            "action": "identify_information_gaps",
            "status": "completed",
            "duration": 8.7,
            "details": "Identified 3 critical information gaps requiring additional research"
        },
        {
            "timestamp": 1023.9,
            "node": "retrieval_engine",
            "action": "retrieve_external_information",
            "status": "completed",
            "duration": 22.3,
            "details": "Retrieved information from 5 high-quality sources"
        },
        {
            "timestamp": 1046.2,
            "node": "information_integrator",
            "action": "integrate_retrieved_information",
            "status": "completed",
            "duration": 12.8,
            "details": "Successfully integrated external information maintaining coherence"
        },
        {
            "timestamp": 1059.0,
            "node": "quality_assessor",
            "action": "assess_draft_quality",
            "status": "completed",
            "duration": 6.5,
            "details": "Quality assessment completed with overall score of 0.84"
        },
        {
            "timestamp": 1065.5,
            "node": "self_evolution_enhancer",
            "action": "apply_evolution_enhancements",
            "status": "completed",
            "duration": 18.9,
            "details": "Applied 5 evolution enhancements across all components"
        },
        {
            "timestamp": 1084.4,
            "node": "report_synthesizer",
            "action": "synthesize_final_report",
            "status": "in_progress",
            "duration": 0.0,
            "details": "Beginning final report synthesis and methodology documentation"
        }
    ]
    
    return {
        "topic": "Artificial Intelligence Applications in Modern Education",
        "current_draft": draft,
        "quality_metrics": quality_metrics,
        "evolution_history": evolution_history,
        "retrieved_info": retrieved_info,
        "information_gaps": information_gaps,
        "iteration_count": 5,
        "workflow_log": workflow_log,
        "requirements": ResearchRequirements(
            domain=Domain.EDUCATION,
            complexity_level=ComplexityLevel.HIGH,
            quality_threshold=0.80,
            preferred_source_types=["academic", "educational", "research"]
        ),
        "final_report": None
    }

async def test_methodology_documentation_generation():
    """Test comprehensive methodology documentation generation"""
    logger.info("Starting methodology documentation generation test")
    
    # Create test state
    state = await create_comprehensive_test_state()
    
    # Initialize report synthesizer
    synthesizer = KimiK2ReportSynthesizer()
    
    try:
        # Test 1: Generate comprehensive methodology documentation
        logger.info("Test 1: Generating comprehensive methodology documentation")
        methodology_doc = await synthesizer.generate_research_methodology_documentation(
            state, workflow_log=state["workflow_log"]
        )
        
        print("\n" + "="*80)
        print("COMPREHENSIVE METHODOLOGY DOCUMENTATION")
        print("="*80)
        print(methodology_doc)
        print("="*80)
        
        # Verify methodology documentation quality
        assert len(methodology_doc) > 1000, "Methodology documentation should be comprehensive"
        assert "Research Methodology Documentation" in methodology_doc
        assert "TTD-DR" in methodology_doc
        assert state["topic"] in methodology_doc
        
        logger.info(f"✓ Methodology documentation generated successfully ({len(methodology_doc)} characters)")
        
        # Test 2: Generate source bibliography
        logger.info("Test 2: Generating source bibliography")
        bibliography = await synthesizer.generate_source_bibliography(
            state["retrieved_info"], citation_style="APA"
        )
        
        print("\n" + "="*80)
        print("SOURCE BIBLIOGRAPHY (APA STYLE)")
        print("="*80)
        print(bibliography)
        print("="*80)
        
        # Verify bibliography quality
        assert len(bibliography) > 200, "Bibliography should contain substantial content"
        assert "Bibliography" in bibliography
        assert "APA" in bibliography
        
        logger.info(f"✓ Bibliography generated successfully ({len(bibliography)} characters)")
        
        # Test 3: Generate methodology summary
        logger.info("Test 3: Generating methodology summary")
        methodology_summary = await synthesizer.generate_methodology_summary(state)
        
        print("\n" + "="*80)
        print("METHODOLOGY SUMMARY")
        print("="*80)
        print(methodology_summary)
        print("="*80)
        
        # Verify methodology summary quality
        assert len(methodology_summary) > 100, "Methodology summary should be substantial"
        assert "TTD-DR" in methodology_summary
        assert state["topic"] in methodology_summary
        
        logger.info(f"✓ Methodology summary generated successfully ({len(methodology_summary)} characters)")
        
        # Test 4: Test different citation styles
        logger.info("Test 4: Testing different citation styles")
        citation_styles = ["APA", "MLA", "Chicago"]
        
        for style in citation_styles:
            style_bibliography = await synthesizer.generate_source_bibliography(
                state["retrieved_info"], citation_style=style
            )
            
            assert style in style_bibliography, f"Bibliography should indicate {style} style"
            logger.info(f"✓ {style} bibliography generated successfully")
        
        # Test 5: Verify methodology data extraction accuracy
        logger.info("Test 5: Verifying methodology data extraction")
        methodology_data = synthesizer._extract_methodology_data(state, state["workflow_log"])
        
        # Verify extracted data accuracy
        assert methodology_data["research_topic"] == state["topic"]
        assert methodology_data["iteration_count"] == state["iteration_count"]
        assert len(methodology_data["sources_used"]) == len(state["retrieved_info"])
        assert len(methodology_data["workflow_stages"]) > 0
        assert methodology_data["workflow_execution_time"] is not None
        
        logger.info("✓ Methodology data extraction verified")
        
        # Test 6: Test technical appendix generation
        logger.info("Test 6: Testing technical appendix generation")
        technical_appendix = synthesizer._generate_technical_appendix(methodology_data)
        
        print("\n" + "="*80)
        print("TECHNICAL APPENDIX")
        print("="*80)
        print(technical_appendix)
        print("="*80)
        
        # Verify technical appendix content
        required_sections = [
            "Framework Parameters",
            "Workflow Execution Details",
            "Information Sources Summary",
            "Quality Metrics Details",
            "Self-Evolution Enhancement Log",
            "Reproducibility Information"
        ]
        
        for section in required_sections:
            assert section in technical_appendix, f"Technical appendix should contain {section}"
        
        logger.info("✓ Technical appendix generated successfully")
        
        # Test 7: Performance and completeness assessment
        logger.info("Test 7: Assessing documentation completeness and performance")
        
        # Check documentation completeness
        completeness_score = 0
        required_elements = [
            "research framework",
            "research process",
            "information retrieval",
            "quality assurance",
            "self-evolution",
            "limitations",
            "reproducibility"
        ]
        
        for element in required_elements:
            if element.replace(" ", "").lower() in methodology_doc.replace(" ", "").lower():
                completeness_score += 1
        
        completeness_percentage = (completeness_score / len(required_elements)) * 100
        logger.info(f"Documentation completeness: {completeness_percentage:.1f}%")
        
        # Performance metrics
        total_content_length = len(methodology_doc) + len(bibliography) + len(methodology_summary)
        logger.info(f"Total generated content: {total_content_length} characters")
        
        # Final verification
        assert completeness_percentage >= 70, "Documentation should be at least 70% complete"
        assert total_content_length >= 2000, "Total content should be substantial"
        
        logger.info("="*80)
        logger.info("ALL METHODOLOGY DOCUMENTATION TESTS PASSED SUCCESSFULLY!")
        logger.info("="*80)
        
        return {
            "methodology_documentation": methodology_doc,
            "source_bibliography": bibliography,
            "methodology_summary": methodology_summary,
            "technical_appendix": technical_appendix,
            "completeness_score": completeness_percentage,
            "total_content_length": total_content_length
        }
        
    except Exception as e:
        logger.error(f"Methodology documentation test failed: {e}")
        raise

async def test_methodology_documentation_edge_cases():
    """Test methodology documentation with edge cases and error conditions"""
    logger.info("Testing methodology documentation edge cases")
    
    synthesizer = KimiK2ReportSynthesizer()
    
    # Test with minimal state
    minimal_state = {
        "topic": "Test Topic",
        "iteration_count": 0,
        "retrieved_info": [],
        "information_gaps": [],
        "evolution_history": []
    }
    
    # Test fallback methodology generation
    logger.info("Testing fallback methodology generation")
    fallback_methodology = await synthesizer.generate_research_methodology_documentation(minimal_state)
    
    assert fallback_methodology is not None
    assert len(fallback_methodology) > 0
    assert "Test Topic" in fallback_methodology
    
    # Test empty bibliography generation
    logger.info("Testing empty bibliography generation")
    empty_bibliography = await synthesizer.generate_source_bibliography([], citation_style="APA")
    
    assert "No sources were retrieved" in empty_bibliography
    
    # Test methodology summary with minimal data
    logger.info("Testing methodology summary with minimal data")
    minimal_summary = await synthesizer.generate_methodology_summary(minimal_state)
    
    assert minimal_summary is not None
    assert "Test Topic" in minimal_summary
    
    logger.info("✓ Edge case testing completed successfully")

async def main():
    """Main test execution function"""
    logger.info("Starting comprehensive methodology documentation integration test")
    
    try:
        # Run main functionality tests
        results = await test_methodology_documentation_generation()
        
        # Run edge case tests
        await test_methodology_documentation_edge_cases()
        
        # Print final summary
        print("\n" + "="*80)
        print("INTEGRATION TEST SUMMARY")
        print("="*80)
        print(f"✓ Methodology Documentation: {len(results['methodology_documentation'])} characters")
        print(f"✓ Source Bibliography: {len(results['source_bibliography'])} characters")
        print(f"✓ Methodology Summary: {len(results['methodology_summary'])} characters")
        print(f"✓ Technical Appendix: {len(results['technical_appendix'])} characters")
        print(f"✓ Documentation Completeness: {results['completeness_score']:.1f}%")
        print(f"✓ Total Content Generated: {results['total_content_length']} characters")
        print("="*80)
        print("🎉 ALL TESTS PASSED - TASK 9.2 IMPLEMENTATION VERIFIED!")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        print("\n" + "="*80)
        print("❌ INTEGRATION TEST FAILED")
        print("="*80)
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())