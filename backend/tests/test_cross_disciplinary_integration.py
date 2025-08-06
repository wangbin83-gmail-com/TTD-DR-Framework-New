"""
Integration tests for cross-disciplinary research capabilities.
Tests the complete workflow from detection through formatting.
"""

import pytest
import json
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from backend.services.cross_disciplinary_integrator import (
    CrossDisciplinaryIntegrator,
    CrossDisciplinaryIntegration,
    CrossDisciplinaryConflict,
    DisciplinaryPerspective
)
from backend.workflow.cross_disciplinary_node import (
    cross_disciplinary_detector_node,
    cross_disciplinary_integrator_node,
    cross_disciplinary_conflict_resolver_node,
    cross_disciplinary_formatter_node,
    cross_disciplinary_quality_assessor_node
)
from backend.models.core import (
    TTDRState, ResearchDomain, Draft, DraftMetadata, QualityMetrics,
    RetrievedInfo, Source, ResearchRequirements, ComplexityLevel
)
from backend.services.kimi_k2_client import KimiK2Client, KimiK2Response


class TestCrossDisciplinaryIntegrationWorkflow:
    """Integration tests for complete cross-disciplinary workflow"""
    
    @pytest.fixture
    def comprehensive_test_state(self):
        """Comprehensive test state with multi-domain content"""
        return TTDRState(
            topic="Artificial Intelligence Applications in Healthcare Business Operations and Scientific Research",
            requirements=ResearchRequirements(
                domain=ResearchDomain.GENERAL,
                complexity_level=ComplexityLevel.ADVANCED,
                max_iterations=8,
                quality_threshold=0.85,
                max_sources=30,
                preferred_source_types=["academic", "industry", "clinical"]
            ),
            current_draft=Draft(
                id="comprehensive_draft_001",
                topic="AI Applications in Healthcare Business Operations and Scientific Research",
                structure=None,
                content={
                    "executive_summary": "This research examines the intersection of AI technology, healthcare applications, business operations, and scientific research methodologies.",
                    "technical_overview": "AI systems utilize machine learning algorithms, neural networks, and deep learning architectures to process complex healthcare data.",
                    "healthcare_applications": "Clinical decision support systems, diagnostic imaging analysis, drug discovery, and personalized treatment recommendations.",
                    "business_impact": "Cost reduction, operational efficiency, revenue optimization, and competitive advantage in healthcare markets.",
                    "research_methodology": "Mixed methods approach combining quantitative analysis, qualitative case studies, and experimental validation.",
                    "regulatory_considerations": "FDA approval processes, HIPAA compliance, data privacy, and ethical considerations in AI healthcare applications.",
                    "future_directions": "Emerging trends in AI healthcare technology and potential research opportunities."
                },
                metadata=DraftMetadata(
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    version="1.2",
                    author="TTD-DR System"
                ),
                quality_score=0.75,
                iteration=2
            ),
            information_gaps=[],
            retrieved_info=[
                # Technology domain sources
                RetrievedInfo(
                    source=Source(
                        url="https://arxiv.org/ai-healthcare-2024",
                        title="Deep Learning Architectures for Medical Image Analysis",
                        domain="academic"
                    ),
                    content="Recent advances in convolutional neural networks and transformer architectures have significantly improved medical image analysis accuracy. State-of-the-art models achieve 95%+ accuracy in radiology image classification tasks.",
                    relevance_score=0.92,
                    credibility_score=0.95,
                    extraction_timestamp=datetime.now()
                ),
                RetrievedInfo(
                    source=Source(
                        url="https://nature.com/ai-drug-discovery",
                        title="AI-Driven Drug Discovery: Computational Methods and Clinical Validation",
                        domain="academic"
                    ),
                    content="Machine learning models can predict drug-target interactions with 85% accuracy, reducing drug discovery timelines from 10-15 years to 3-5 years. Clinical validation remains the primary bottleneck.",
                    relevance_score=0.88,
                    credibility_score=0.98,
                    extraction_timestamp=datetime.now()
                ),
                # Healthcare/Science domain sources
                RetrievedInfo(
                    source=Source(
                        url="https://nejm.org/ai-clinical-decision-support",
                        title="Clinical Decision Support Systems: Evidence from Randomized Controlled Trials",
                        domain="clinical"
                    ),
                    content="Systematic review of 47 RCTs shows AI-powered clinical decision support systems improve diagnostic accuracy by 23% and reduce medical errors by 31%. However, physician adoption remains challenging due to workflow integration issues.",
                    relevance_score=0.94,
                    credibility_score=0.97,
                    extraction_timestamp=datetime.now()
                ),
                RetrievedInfo(
                    source=Source(
                        url="https://jama.org/ai-healthcare-outcomes",
                        title="Patient Outcomes and AI Implementation: Multi-Center Study",
                        domain="clinical"
                    ),
                    content="Multi-center study (n=15,847 patients) demonstrates 18% reduction in hospital readmissions and 12% improvement in treatment outcomes with AI-assisted care protocols. Cost-effectiveness analysis shows $2.3M annual savings per hospital.",
                    relevance_score=0.91,
                    credibility_score=0.96,
                    extraction_timestamp=datetime.now()
                ),
                # Business domain sources
                RetrievedInfo(
                    source=Source(
                        url="https://mckinsey.com/ai-healthcare-market-analysis",
                        title="AI in Healthcare: Market Analysis and Investment Trends",
                        domain="industry"
                    ),
                    content="Healthcare AI market projected to reach $148B by 2029, with 35% CAGR. Key investment areas: diagnostic imaging ($45B), drug discovery ($28B), and clinical decision support ($31B). ROI typically achieved within 18-24 months.",
                    relevance_score=0.86,
                    credibility_score=0.82,
                    extraction_timestamp=datetime.now()
                ),
                RetrievedInfo(
                    source=Source(
                        url="https://deloitte.com/healthcare-ai-implementation",
                        title="Healthcare AI Implementation: Challenges and Success Factors",
                        domain="industry"
                    ),
                    content="Survey of 200 healthcare executives reveals key implementation challenges: data integration (78%), regulatory compliance (65%), staff training (61%). Successful implementations require 12-18 month change management programs.",
                    relevance_score=0.83,
                    credibility_score=0.79,
                    extraction_timestamp=datetime.now()
                ),
                # Academic/Research methodology sources
                RetrievedInfo(
                    source=Source(
                        url="https://academic.oup.com/ai-research-methodology",
                        title="Research Methodologies for AI Healthcare Studies: Best Practices",
                        domain="academic"
                    ),
                    content="Comprehensive framework for AI healthcare research includes: prospective cohort studies, randomized controlled trials, real-world evidence collection, and ethical review processes. Sample size calculations require 20-30% larger cohorts for AI validation studies.",
                    relevance_score=0.89,
                    credibility_score=0.93,
                    extraction_timestamp=datetime.now()
                )
            ],
            iteration_count=2,
            quality_metrics=QualityMetrics(
                completeness=0.78,
                coherence=0.82,
                accuracy=0.85,
                citation_quality=0.73
            ),
            evolution_history=[],
            final_report=None
        )
    
    @pytest.fixture
    def mock_kimi_responses(self):
        """Comprehensive mock responses for Kimi K2 throughout the workflow"""
        return {
            # Domain detection response
            "domain_detection": json.dumps({
                "domains": [
                    {"domain": "TECHNOLOGY", "relevance": 0.85, "reasoning": "AI algorithms, machine learning, neural networks"},
                    {"domain": "SCIENCE", "relevance": 0.92, "reasoning": "Clinical research, RCTs, evidence-based medicine"},
                    {"domain": "BUSINESS", "relevance": 0.78, "reasoning": "Market analysis, ROI, implementation strategies"},
                    {"domain": "ACADEMIC", "relevance": 0.88, "reasoning": "Research methodology, systematic reviews"}
                ],
                "is_cross_disciplinary": True
            }),
            
            # Disciplinary perspective analysis responses
            "technology_perspective": json.dumps({
                "key_concepts": ["machine learning", "neural networks", "deep learning", "algorithms", "data processing"],
                "methodological_approach": "Experimental software development with performance benchmarking and validation testing",
                "theoretical_framework": "Computer science theory, information theory, and computational learning theory",
                "evidence_types": ["performance metrics", "accuracy scores", "computational benchmarks", "technical specifications"],
                "terminology": {
                    "ML": "Machine Learning",
                    "CNN": "Convolutional Neural Network",
                    "AI": "Artificial Intelligence",
                    "GPU": "Graphics Processing Unit"
                },
                "confidence": 0.87
            }),
            
            "science_perspective": json.dumps({
                "key_concepts": ["clinical trials", "evidence-based medicine", "patient outcomes", "diagnostic accuracy", "medical validation"],
                "methodological_approach": "Randomized controlled trials, systematic reviews, meta-analyses, and prospective cohort studies",
                "theoretical_framework": "Evidence-based medicine, clinical epidemiology, and biostatistics",
                "evidence_types": ["RCT results", "systematic reviews", "clinical outcomes", "biostatistical analysis"],
                "terminology": {
                    "RCT": "Randomized Controlled Trial",
                    "CI": "Confidence Interval",
                    "p-value": "Statistical significance probability",
                    "NNT": "Number Needed to Treat"
                },
                "confidence": 0.94
            }),
            
            "business_perspective": json.dumps({
                "key_concepts": ["ROI", "market analysis", "implementation costs", "competitive advantage", "operational efficiency"],
                "methodological_approach": "Business case analysis, cost-benefit analysis, market research, and stakeholder interviews",
                "theoretical_framework": "Strategic management theory, operations research, and financial analysis",
                "evidence_types": ["financial data", "market reports", "case studies", "survey results"],
                "terminology": {
                    "ROI": "Return on Investment",
                    "CAGR": "Compound Annual Growth Rate",
                    "TCO": "Total Cost of Ownership",
                    "KPI": "Key Performance Indicator"
                },
                "confidence": 0.81
            }),
            
            "academic_perspective": json.dumps({
                "key_concepts": ["research methodology", "peer review", "academic rigor", "theoretical frameworks", "knowledge contribution"],
                "methodological_approach": "Systematic literature reviews, theoretical analysis, and methodological validation",
                "theoretical_framework": "Philosophy of science, research methodology theory, and academic discourse analysis",
                "evidence_types": ["peer-reviewed publications", "theoretical models", "methodological frameworks", "citation analysis"],
                "terminology": {
                    "SLR": "Systematic Literature Review",
                    "IF": "Impact Factor",
                    "h-index": "Hirsch index for citation impact",
                    "DOI": "Digital Object Identifier"
                },
                "confidence": 0.89
            }),
            
            # Conflict detection responses
            "conflict_tech_science": json.dumps({
                "has_conflict": True,
                "conflict_type": "methodological",
                "description": "Technology domain emphasizes performance metrics and computational efficiency, while science domain prioritizes clinical validation and patient safety. Different standards for 'success' and validation requirements.",
                "severity": 0.72,
                "conflicting_aspects": ["validation standards", "success metrics", "timeline expectations"]
            }),
            
            "conflict_business_science": json.dumps({
                "has_conflict": True,
                "conflict_type": "theoretical",
                "description": "Business domain focuses on financial returns and market adoption, while science domain emphasizes clinical efficacy and patient outcomes. Tension between profit motives and patient care priorities.",
                "severity": 0.68,
                "conflicting_aspects": ["success definitions", "priority frameworks", "ethical considerations"]
            }),
            
            "conflict_tech_academic": json.dumps({
                "has_conflict": False,
                "conflict_type": "none",
                "description": "Technology and academic domains align well in terms of rigorous methodology and evidence-based approaches.",
                "severity": 0.15,
                "conflicting_aspects": []
            }),
            
            # Conflict resolution responses
            "resolution_tech_science": json.dumps({
                "resolution_strategy": "Establish dual validation framework combining computational performance metrics with clinical outcome measures. Implement staged validation process: technical validation → clinical pilot → full clinical trial.",
                "integrated_approach": "Hybrid validation methodology that satisfies both technical performance requirements and clinical safety standards",
                "remaining_tensions": ["Timeline pressures", "Resource allocation between technical and clinical validation"],
                "confidence": 0.84
            }),
            
            "resolution_business_science": json.dumps({
                "resolution_strategy": "Develop value-based healthcare framework that aligns financial incentives with patient outcomes. Implement shared risk models and outcome-based pricing structures.",
                "integrated_approach": "Triple aim framework focusing on patient experience, population health, and cost reduction simultaneously",
                "remaining_tensions": ["Short-term vs long-term ROI expectations"],
                "confidence": 0.79
            })
        }
    
    @pytest.fixture
    def mock_kimi_client(self, mock_kimi_responses):
        """Mock Kimi K2 client with comprehensive responses"""
        client = Mock(spec=KimiK2Client)
        client.generate_text = AsyncMock()
        
        # Set up response sequence for complete workflow
        response_sequence = [
            # Domain detection
            KimiK2Response(content=mock_kimi_responses["domain_detection"]),
            # Disciplinary perspectives
            KimiK2Response(content=mock_kimi_responses["technology_perspective"]),
            KimiK2Response(content=mock_kimi_responses["science_perspective"]),
            KimiK2Response(content=mock_kimi_responses["business_perspective"]),
            KimiK2Response(content=mock_kimi_responses["academic_perspective"]),
            # Conflict detection
            KimiK2Response(content=mock_kimi_responses["conflict_tech_science"]),
            KimiK2Response(content=mock_kimi_responses["conflict_business_science"]),
            KimiK2Response(content=mock_kimi_responses["conflict_tech_academic"]),
            # Conflict resolution
            KimiK2Response(content=mock_kimi_responses["resolution_tech_science"]),
            KimiK2Response(content=mock_kimi_responses["resolution_business_science"])
        ]
        
        client.generate_text.side_effect = response_sequence
        return client
    
    def test_complete_cross_disciplinary_workflow(self, comprehensive_test_state, mock_kimi_client):
        """Test complete cross-disciplinary workflow from detection to formatting"""
        
        with patch('backend.workflow.cross_disciplinary_node.KimiK2Client', return_value=mock_kimi_client):
            # Step 1: Cross-disciplinary detection
            state_after_detection = cross_disciplinary_detector_node(comprehensive_test_state)
            
            # Verify detection results
            assert state_after_detection["requires_cross_disciplinary"] is True
            detection_metadata = state_after_detection["cross_disciplinary_metadata"]
            assert detection_metadata["is_cross_disciplinary"] is True
            assert len(detection_metadata["involved_domains"]) >= 3
            assert "TECHNOLOGY" in detection_metadata["involved_domains"]
            assert "SCIENCE" in detection_metadata["involved_domains"]
            assert "BUSINESS" in detection_metadata["involved_domains"]
            
            # Step 2: Multi-domain integration
            state_after_integration = cross_disciplinary_integrator_node(state_after_detection)
            
            # Verify integration results
            assert "cross_disciplinary_integration" in state_after_integration
            integration = state_after_integration["cross_disciplinary_integration"]
            assert len(integration.primary_domains) >= 3
            assert len(integration.disciplinary_perspectives) >= 3
            assert integration.coherence_score > 0.0
            
            integration_metadata = state_after_integration["integration_metadata"]
            assert integration_metadata["integration_completed"] is True
            assert integration_metadata["disciplinary_perspectives"] >= 3
            
            # Step 3: Conflict resolution
            state_after_resolution = cross_disciplinary_conflict_resolver_node(state_after_integration)
            
            # Verify conflict resolution
            resolved_integration = state_after_resolution["cross_disciplinary_integration"]
            assert len(resolved_integration.conflicts_resolved) > 0
            
            resolution_metadata = state_after_resolution["integration_metadata"]
            assert resolution_metadata["conflicts_resolved"] > 0
            assert "resolution_success_rate" in resolution_metadata
            
            # Step 4: Specialized formatting
            state_after_formatting = cross_disciplinary_formatter_node(state_after_resolution)
            
            # Verify formatting results
            formatted_draft = state_after_formatting["current_draft"]
            assert "cross_disciplinary_report" in formatted_draft.content
            assert "disciplinary_perspectives" in formatted_draft.content
            
            formatting_metadata = state_after_formatting["formatting_metadata"]
            assert formatting_metadata["cross_disciplinary_formatting"] is True
            
            # Step 5: Quality assessment
            final_state = cross_disciplinary_quality_assessor_node(state_after_formatting)
            
            # Verify quality assessment
            assert "cross_disciplinary_quality_metadata" in final_state
            quality_metadata = final_state["cross_disciplinary_quality_metadata"]
            assert "overall_cross_disciplinary_score" in quality_metadata
            assert quality_metadata["domains_integrated"] >= 3
            assert 0.0 <= quality_metadata["overall_cross_disciplinary_score"] <= 1.0
            
            # Verify final state integrity
            assert final_state["topic"] == comprehensive_test_state["topic"]
            assert final_state["requirements"] == comprehensive_test_state["requirements"]
            assert len(final_state["retrieved_info"]) == len(comprehensive_test_state["retrieved_info"])
    
    def test_cross_disciplinary_quality_and_coherence(self, comprehensive_test_state, mock_kimi_client):
        """Test that cross-disciplinary integration maintains quality and coherence"""
        
        with patch('backend.workflow.cross_disciplinary_node.KimiK2Client', return_value=mock_kimi_client):
            # Run through complete workflow
            state = comprehensive_test_state
            state = cross_disciplinary_detector_node(state)
            state = cross_disciplinary_integrator_node(state)
            state = cross_disciplinary_conflict_resolver_node(state)
            state = cross_disciplinary_formatter_node(state)
            final_state = cross_disciplinary_quality_assessor_node(state)
            
            # Analyze quality metrics
            integration = final_state["cross_disciplinary_integration"]
            quality_metadata = final_state["cross_disciplinary_quality_metadata"]
            
            # Test coherence maintenance
            assert integration.coherence_score >= 0.6  # Minimum acceptable coherence
            
            # Test quality improvement or maintenance
            original_quality = comprehensive_test_state["quality_metrics"].overall_score
            final_quality = quality_metadata["overall_cross_disciplinary_score"]
            
            # Cross-disciplinary integration should maintain or improve quality
            assert final_quality >= original_quality * 0.9  # Allow for small quality trade-offs
            
            # Test conflict resolution effectiveness
            if integration.conflicts_identified:
                resolution_rate = len(integration.conflicts_resolved) / len(integration.conflicts_identified)
                assert resolution_rate >= 0.7  # At least 70% of conflicts should be resolved
            
            # Test disciplinary balance
            perspectives = integration.disciplinary_perspectives
            confidence_scores = [p.confidence for p in perspectives]
            min_confidence = min(confidence_scores)
            max_confidence = max(confidence_scores)
            
            # Perspectives should be reasonably balanced (no single domain dominates excessively)
            confidence_range = max_confidence - min_confidence
            assert confidence_range <= 0.4  # Maximum 40% difference between highest and lowest confidence
    
    def test_cross_disciplinary_content_integration(self, comprehensive_test_state, mock_kimi_client):
        """Test that cross-disciplinary content is properly integrated"""
        
        with patch('backend.workflow.cross_disciplinary_node.KimiK2Client', return_value=mock_kimi_client):
            # Run through workflow
            state = comprehensive_test_state
            state = cross_disciplinary_detector_node(state)
            state = cross_disciplinary_integrator_node(state)
            state = cross_disciplinary_conflict_resolver_node(state)
            final_state = cross_disciplinary_formatter_node(state)
            
            # Analyze content integration
            final_draft = final_state["current_draft"]
            integration = final_state["cross_disciplinary_integration"]
            
            # Test that original content is preserved
            original_sections = set(comprehensive_test_state["current_draft"].content.keys())
            final_sections = set(final_draft.content.keys())
            
            # Original sections should be preserved
            assert original_sections.issubset(final_sections)
            
            # Test that cross-disciplinary sections are added
            cross_disciplinary_sections = {
                "cross_disciplinary_report",
                "disciplinary_perspectives"
            }
            
            for section in cross_disciplinary_sections:
                assert section in final_draft.content
                assert final_draft.content[section]  # Should not be empty
            
            # Test that conflict resolutions are documented if conflicts existed
            if integration.conflicts_resolved:
                assert "conflict_resolutions" in final_draft.content
                assert final_draft.content["conflict_resolutions"]
            
            # Test content quality
            cross_disciplinary_report = final_draft.content["cross_disciplinary_report"]
            
            # Should mention multiple domains
            domain_mentions = 0
            for domain in integration.primary_domains:
                if domain.value.lower() in cross_disciplinary_report.lower():
                    domain_mentions += 1
            
            assert domain_mentions >= 2  # At least 2 domains should be mentioned
            
            # Should have structured format
            assert "Cross-Disciplinary Research Report" in cross_disciplinary_report
            assert "Disciplinary Perspectives" in cross_disciplinary_report
    
    def test_error_resilience_in_workflow(self, comprehensive_test_state):
        """Test that workflow handles errors gracefully without breaking"""
        
        # Mock Kimi K2 client that fails intermittently
        mock_client = Mock(spec=KimiK2Client)
        mock_client.generate_text = AsyncMock()
        
        # Set up mixed success/failure responses
        responses = [
            KimiK2Response(content='{"domains": [{"domain": "TECHNOLOGY", "relevance": 0.8}], "is_cross_disciplinary": false}'),  # Success
            Exception("Network timeout"),  # Failure
            KimiK2Response(content='{"key_concepts": ["AI"], "confidence": 0.5}'),  # Success
            Exception("API rate limit"),  # Failure
            KimiK2Response(content='{"has_conflict": false}')  # Success
        ]
        
        mock_client.generate_text.side_effect = responses
        
        with patch('backend.workflow.cross_disciplinary_node.KimiK2Client', return_value=mock_client):
            # Run workflow - should not crash despite errors
            state = comprehensive_test_state
            
            try:
                state = cross_disciplinary_detector_node(state)
                # Should handle detection errors gracefully
                assert "error_log" in state or "cross_disciplinary_metadata" in state
                
                state = cross_disciplinary_integrator_node(state)
                # Should handle integration errors gracefully
                assert "error_log" in state or "cross_disciplinary_integration" in state
                
                # Workflow should continue despite errors
                assert state["topic"] == comprehensive_test_state["topic"]
                assert "requirements" in state
                
            except Exception as e:
                pytest.fail(f"Workflow should handle errors gracefully, but raised: {e}")
    
    def test_performance_and_scalability(self, comprehensive_test_state, mock_kimi_client):
        """Test performance characteristics of cross-disciplinary workflow"""
        
        with patch('backend.workflow.cross_disciplinary_node.KimiK2Client', return_value=mock_kimi_client):
            import time
            
            start_time = time.time()
            
            # Run complete workflow
            state = comprehensive_test_state
            state = cross_disciplinary_detector_node(state)
            state = cross_disciplinary_integrator_node(state)
            state = cross_disciplinary_conflict_resolver_node(state)
            state = cross_disciplinary_formatter_node(state)
            final_state = cross_disciplinary_quality_assessor_node(state)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Workflow should complete in reasonable time (allowing for mock overhead)
            assert execution_time < 5.0  # Should complete within 5 seconds
            
            # Test memory efficiency - state size should not grow excessively
            import sys
            final_state_size = sys.getsizeof(str(final_state))
            original_state_size = sys.getsizeof(str(comprehensive_test_state))
            
            # Final state should not be more than 5x original size
            size_ratio = final_state_size / original_state_size
            assert size_ratio < 5.0
            
            # Test that workflow scales with number of domains
            integration = final_state["cross_disciplinary_integration"]
            num_domains = len(integration.primary_domains)
            num_perspectives = len(integration.disciplinary_perspectives)
            
            # Should handle multiple domains efficiently
            assert num_domains >= 3
            assert num_perspectives == num_domains  # One perspective per domain
            
            # Coherence score should remain reasonable even with multiple domains
            assert integration.coherence_score >= 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])