"""
Tests for cross-disciplinary research integration system.
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from backend.services.cross_disciplinary_integrator import (
    CrossDisciplinaryIntegrator,
    CrossDisciplinaryIntegration,
    CrossDisciplinaryConflict,
    DisciplinaryPerspective,
    DisciplinaryRelationship
)
from backend.models.core import (
    ResearchDomain, Draft, DraftMetadata, QualityMetrics,
    RetrievedInfo, Source, ResearchRequirements, ComplexityLevel
)
from backend.services.kimi_k2_client import KimiK2Client, KimiK2Response


class TestCrossDisciplinaryIntegrator:
    """Test cases for CrossDisciplinaryIntegrator"""
    
    @pytest.fixture
    def mock_kimi_client(self):
        """Mock Kimi K2 client for testing"""
        client = Mock(spec=KimiK2Client)
        client.generate_text = AsyncMock()
        return client
    
    @pytest.fixture
    def integrator(self, mock_kimi_client):
        """Create integrator instance for testing"""
        return CrossDisciplinaryIntegrator(mock_kimi_client)
    
    @pytest.fixture
    def sample_retrieved_info(self):
        """Sample retrieved information for testing"""
        return [
            RetrievedInfo(
                source=Source(url="https://tech.example.com", title="Tech Article"),
                content="This article discusses artificial intelligence and machine learning algorithms in software development.",
                relevance_score=0.8,
                credibility_score=0.7,
                extraction_timestamp=datetime.now()
            ),
            RetrievedInfo(
                source=Source(url="https://science.example.com", title="Science Paper"),
                content="Research methodology and experimental design for AI systems evaluation and validation.",
                relevance_score=0.9,
                credibility_score=0.9,
                extraction_timestamp=datetime.now()
            ),
            RetrievedInfo(
                source=Source(url="https://business.example.com", title="Business Report"),
                content="Market analysis of AI adoption in enterprise software and business strategy implications.",
                relevance_score=0.7,
                credibility_score=0.6,
                extraction_timestamp=datetime.now()
            )
        ]
    
    @pytest.fixture
    def sample_draft(self):
        """Sample draft for testing"""
        return Draft(
            id="test_draft_001",
            topic="Artificial Intelligence in Business Applications",
            structure=None,
            content={
                "introduction": "AI is transforming business operations...",
                "technical_analysis": "Machine learning algorithms provide...",
                "business_impact": "Companies are seeing significant ROI...",
                "research_methodology": "This study employs mixed methods..."
            },
            metadata=DraftMetadata(),
            quality_score=0.7,
            iteration=2
        )
    
    def test_detect_cross_disciplinary_nature_single_domain(self, integrator, sample_retrieved_info):
        """Test detection when topic is single-domain"""
        topic = "Python programming best practices"
        
        # Mock Kimi K2 response for single domain
        integrator.kimi_client.generate_text.return_value = KimiK2Response(
            content=json.dumps({
                "domains": [
                    {"domain": "TECHNOLOGY", "relevance": 0.9, "reasoning": "Programming topic"}
                ],
                "is_cross_disciplinary": False
            })
        )
        
        is_cross_disciplinary, domains = integrator.detect_cross_disciplinary_nature(
            topic, sample_retrieved_info[:1]  # Only tech info
        )
        
        assert not is_cross_disciplinary
        assert len(domains) == 1
        assert ResearchDomain.TECHNOLOGY in domains
    
    def test_detect_cross_disciplinary_nature_multi_domain(self, integrator, sample_retrieved_info):
        """Test detection when topic spans multiple domains"""
        topic = "AI impact on business and scientific research"
        
        # Mock Kimi K2 response for multiple domains
        integrator.kimi_client.generate_text.return_value = KimiK2Response(
            content=json.dumps({
                "domains": [
                    {"domain": "TECHNOLOGY", "relevance": 0.8, "reasoning": "AI technology"},
                    {"domain": "BUSINESS", "relevance": 0.7, "reasoning": "Business impact"},
                    {"domain": "SCIENCE", "relevance": 0.6, "reasoning": "Research applications"}
                ],
                "is_cross_disciplinary": True
            })
        )
        
        is_cross_disciplinary, domains = integrator.detect_cross_disciplinary_nature(
            topic, sample_retrieved_info
        )
        
        assert is_cross_disciplinary
        assert len(domains) >= 2
        assert ResearchDomain.TECHNOLOGY in domains
        assert ResearchDomain.BUSINESS in domains
    
    def test_integrate_multi_domain_knowledge(self, integrator, sample_retrieved_info, sample_draft):
        """Test multi-domain knowledge integration"""
        topic = "AI in Business and Science"
        domains = [ResearchDomain.TECHNOLOGY, ResearchDomain.BUSINESS, ResearchDomain.SCIENCE]
        
        # Mock Kimi K2 responses for different analysis steps
        responses = [
            # Domain analysis responses
            json.dumps({
                "key_concepts": ["artificial intelligence", "algorithms"],
                "methodological_approach": "Software engineering practices",
                "theoretical_framework": "Computer science theory",
                "evidence_types": ["code examples", "performance metrics"],
                "terminology": {"AI": "Artificial Intelligence"},
                "confidence": 0.8
            }),
            json.dumps({
                "key_concepts": ["market analysis", "ROI"],
                "methodological_approach": "Business case studies",
                "theoretical_framework": "Strategic management theory",
                "evidence_types": ["financial data", "case studies"],
                "terminology": {"ROI": "Return on Investment"},
                "confidence": 0.7
            }),
            json.dumps({
                "key_concepts": ["research methodology", "validation"],
                "methodological_approach": "Experimental design",
                "theoretical_framework": "Scientific method",
                "evidence_types": ["peer-reviewed studies", "experiments"],
                "terminology": {"p-value": "statistical significance"},
                "confidence": 0.9
            }),
            # Conflict detection responses
            json.dumps({
                "has_conflict": True,
                "conflict_type": "methodological",
                "description": "Different approaches to validation",
                "severity": 0.6
            }),
            json.dumps({
                "has_conflict": False,
                "conflict_type": "none",
                "description": "No significant conflicts",
                "severity": 0.1
            }),
            # Conflict resolution response
            json.dumps({
                "resolution_strategy": "Combine quantitative and qualitative methods",
                "integrated_approach": "Mixed methods validation",
                "remaining_tensions": [],
                "confidence": 0.8
            })
        ]
        
        integrator.kimi_client.generate_text.side_effect = [
            KimiK2Response(content=response) for response in responses
        ]
        
        integration = integrator.integrate_multi_domain_knowledge(
            topic, domains, sample_retrieved_info, sample_draft
        )
        
        assert isinstance(integration, CrossDisciplinaryIntegration)
        assert len(integration.primary_domains) == 3
        assert len(integration.disciplinary_perspectives) == 3
        assert integration.coherence_score > 0.0
        assert integration.integration_strategy in ["synthesis", "dialectical", "hierarchical"]
    
    def test_resolve_cross_disciplinary_conflicts(self, integrator):
        """Test cross-disciplinary conflict resolution"""
        # Create sample conflicts
        conflicts = [
            CrossDisciplinaryConflict(
                conflict_id="conflict_001",
                domains_involved=[ResearchDomain.TECHNOLOGY, ResearchDomain.SCIENCE],
                conflict_type="methodological",
                description="Different validation approaches",
                conflicting_information=[],
                severity=0.7
            ),
            CrossDisciplinaryConflict(
                conflict_id="conflict_002",
                domains_involved=[ResearchDomain.BUSINESS, ResearchDomain.SCIENCE],
                conflict_type="theoretical",
                description="Different success metrics",
                conflicting_information=[],
                severity=0.5
            )
        ]
        
        # Create sample perspectives
        perspectives = [
            DisciplinaryPerspective(
                domain=ResearchDomain.TECHNOLOGY,
                confidence=0.8,
                methodological_approach="Agile development"
            ),
            DisciplinaryPerspective(
                domain=ResearchDomain.SCIENCE,
                confidence=0.9,
                methodological_approach="Experimental validation"
            ),
            DisciplinaryPerspective(
                domain=ResearchDomain.BUSINESS,
                confidence=0.7,
                methodological_approach="Business case analysis"
            )
        ]
        
        # Mock resolution responses
        integrator.kimi_client.generate_text.side_effect = [
            KimiK2Response(content=json.dumps({
                "resolution_strategy": "Integrate agile and experimental approaches",
                "integrated_approach": "Iterative validation cycles",
                "remaining_tensions": [],
                "confidence": 0.8
            })),
            KimiK2Response(content=json.dumps({
                "resolution_strategy": "Balance scientific rigor with business metrics",
                "integrated_approach": "Multi-criteria evaluation",
                "remaining_tensions": ["Time constraints"],
                "confidence": 0.7
            }))
        ]
        
        resolved_conflicts = integrator.resolve_cross_disciplinary_conflicts(
            conflicts, perspectives
        )
        
        assert len(resolved_conflicts) == 2
        assert all(conflict.resolved for conflict in resolved_conflicts)
        assert all(conflict.resolution_strategy for conflict in resolved_conflicts)
    
    def test_format_cross_disciplinary_output_comprehensive(self, integrator, sample_draft):
        """Test comprehensive cross-disciplinary output formatting"""
        # Create sample integration
        integration = CrossDisciplinaryIntegration(
            primary_domains=[ResearchDomain.TECHNOLOGY, ResearchDomain.BUSINESS],
            disciplinary_perspectives=[
                DisciplinaryPerspective(
                    domain=ResearchDomain.TECHNOLOGY,
                    confidence=0.8,
                    key_concepts=["AI", "algorithms"],
                    methodological_approach="Software engineering",
                    theoretical_framework="Computer science"
                ),
                DisciplinaryPerspective(
                    domain=ResearchDomain.BUSINESS,
                    confidence=0.7,
                    key_concepts=["ROI", "strategy"],
                    methodological_approach="Case studies",
                    theoretical_framework="Strategic management"
                )
            ],
            integration_strategy="synthesis",
            conflicts_identified=[],
            conflicts_resolved=[],
            synthesis_approach="evidence_based_synthesis",
            coherence_score=0.8
        )
        
        formatted_output = integrator.format_cross_disciplinary_output(
            sample_draft, integration, "comprehensive"
        )
        
        assert "Cross-Disciplinary Research Report" in formatted_output
        assert "TECHNOLOGY" in formatted_output
        assert "BUSINESS" in formatted_output
        assert "Disciplinary Perspectives" in formatted_output
        assert "Integrated Analysis" in formatted_output
    
    def test_fallback_behavior_when_kimi_fails(self, integrator, sample_retrieved_info):
        """Test fallback behavior when Kimi K2 is unavailable"""
        topic = "AI in healthcare"
        
        # Mock Kimi K2 failure
        integrator.kimi_client.generate_text.side_effect = Exception("API unavailable")
        
        # Should not raise exception, should use fallback
        is_cross_disciplinary, domains = integrator.detect_cross_disciplinary_nature(
            topic, sample_retrieved_info
        )
        
        # Should return reasonable fallback results
        assert isinstance(is_cross_disciplinary, bool)
        assert isinstance(domains, list)
        assert len(domains) > 0
    
    def test_disciplinary_perspective_analysis(self, integrator, sample_retrieved_info):
        """Test analysis of individual disciplinary perspectives"""
        topic = "Machine learning applications"
        domain = ResearchDomain.TECHNOLOGY
        
        # Mock Kimi K2 response
        integrator.kimi_client.generate_text.return_value = KimiK2Response(
            content=json.dumps({
                "key_concepts": ["machine learning", "neural networks", "algorithms"],
                "methodological_approach": "Experimental software development",
                "theoretical_framework": "Computational learning theory",
                "evidence_types": ["benchmarks", "code repositories"],
                "terminology": {"ML": "Machine Learning", "NN": "Neural Network"},
                "confidence": 0.85
            })
        )
        
        perspective = integrator._analyze_disciplinary_perspective(
            topic, domain, sample_retrieved_info
        )
        
        assert isinstance(perspective, DisciplinaryPerspective)
        assert perspective.domain == domain
        assert perspective.confidence == 0.85
        assert "machine learning" in perspective.key_concepts
        assert perspective.methodological_approach == "Experimental software development"
        assert "ML" in perspective.terminology
    
    def test_conflict_detection_between_perspectives(self, integrator, sample_retrieved_info):
        """Test detection of conflicts between disciplinary perspectives"""
        perspective1 = DisciplinaryPerspective(
            domain=ResearchDomain.TECHNOLOGY,
            confidence=0.8,
            key_concepts=["efficiency", "performance"],
            methodological_approach="Quantitative benchmarking",
            theoretical_framework="Computer science theory"
        )
        
        perspective2 = DisciplinaryPerspective(
            domain=ResearchDomain.SCIENCE,
            confidence=0.9,
            key_concepts=["validity", "reliability"],
            methodological_approach="Controlled experiments",
            theoretical_framework="Scientific method"
        )
        
        # Mock conflict detection response
        integrator.kimi_client.generate_text.return_value = KimiK2Response(
            content=json.dumps({
                "has_conflict": True,
                "conflict_type": "methodological",
                "description": "Different approaches to validation and measurement",
                "severity": 0.6,
                "conflicting_aspects": ["validation methods", "success metrics"]
            })
        )
        
        conflict = integrator._detect_perspective_conflict(
            perspective1, perspective2, sample_retrieved_info
        )
        
        assert conflict is not None
        assert isinstance(conflict, CrossDisciplinaryConflict)
        assert conflict.conflict_type == "methodological"
        assert len(conflict.domains_involved) == 2
        assert ResearchDomain.TECHNOLOGY in conflict.domains_involved
        assert ResearchDomain.SCIENCE in conflict.domains_involved
        assert conflict.severity == 0.6
    
    def test_integration_coherence_calculation(self, integrator):
        """Test calculation of integration coherence score"""
        perspectives = [
            DisciplinaryPerspective(domain=ResearchDomain.TECHNOLOGY, confidence=0.8),
            DisciplinaryPerspective(domain=ResearchDomain.SCIENCE, confidence=0.9),
            DisciplinaryPerspective(domain=ResearchDomain.BUSINESS, confidence=0.7)
        ]
        
        resolved_conflicts = [
            CrossDisciplinaryConflict(
                conflict_id="test_conflict",
                domains_involved=[ResearchDomain.TECHNOLOGY, ResearchDomain.SCIENCE],
                conflict_type="methodological",
                description="Test conflict",
                conflicting_information=[],
                severity=0.5,
                resolved=True
            )
        ]
        
        coherence_score = integrator._calculate_integration_coherence(
            perspectives, resolved_conflicts
        )
        
        assert 0.0 <= coherence_score <= 1.0
        # Should be based on average confidence minus conflict penalty
        expected_base = (0.8 + 0.9 + 0.7) / 3  # 0.8
        expected_penalty = len(resolved_conflicts) * 0.1  # 0.1
        expected_score = max(0.0, expected_base - expected_penalty)  # 0.7
        assert abs(coherence_score - expected_score) < 0.01
    
    def test_quality_assessment_methods(self, integrator):
        """Test cross-disciplinary quality assessment methods"""
        integration = CrossDisciplinaryIntegration(
            primary_domains=[ResearchDomain.TECHNOLOGY, ResearchDomain.SCIENCE],
            disciplinary_perspectives=[
                DisciplinaryPerspective(domain=ResearchDomain.TECHNOLOGY, confidence=0.8),
                DisciplinaryPerspective(domain=ResearchDomain.SCIENCE, confidence=0.9)
            ],
            integration_strategy="synthesis",
            conflicts_identified=[
                CrossDisciplinaryConflict(
                    conflict_id="test_conflict",
                    domains_involved=[ResearchDomain.TECHNOLOGY, ResearchDomain.SCIENCE],
                    conflict_type="methodological",
                    description="Test conflict",
                    conflicting_information=[],
                    severity=0.6
                )
            ],
            conflicts_resolved=[
                CrossDisciplinaryConflict(
                    conflict_id="test_conflict",
                    domains_involved=[ResearchDomain.TECHNOLOGY, ResearchDomain.SCIENCE],
                    conflict_type="methodological",
                    description="Test conflict",
                    conflicting_information=[],
                    severity=0.6,
                    resolved=True
                )
            ],
            synthesis_approach="evidence_based_synthesis",
            coherence_score=0.8
        )
        
        # Test disciplinary balance assessment
        balance_score = integrator._assess_disciplinary_balance(integration)
        assert 0.0 <= balance_score <= 1.0
        
        # Test conflict resolution assessment
        resolution_score = integrator._assess_conflict_resolution(integration)
        assert 0.0 <= resolution_score <= 1.0
        assert resolution_score == 1.0  # All conflicts resolved
        
        # Test synthesis quality assessment
        synthesis_score = integrator._assess_synthesis_quality(integration)
        assert 0.0 <= synthesis_score <= 1.0
        
        # Test methodological integration assessment
        method_score = integrator._assess_methodological_integration(integration)
        assert 0.0 <= method_score <= 1.0


class TestCrossDisciplinaryModels:
    """Test cases for cross-disciplinary data models"""
    
    def test_disciplinary_perspective_creation(self):
        """Test creation of DisciplinaryPerspective"""
        perspective = DisciplinaryPerspective(
            domain=ResearchDomain.TECHNOLOGY,
            confidence=0.8,
            key_concepts=["AI", "ML"],
            methodological_approach="Experimental",
            theoretical_framework="Computer Science",
            evidence_types=["benchmarks"],
            terminology={"AI": "Artificial Intelligence"}
        )
        
        assert perspective.domain == ResearchDomain.TECHNOLOGY
        assert perspective.confidence == 0.8
        assert "AI" in perspective.key_concepts
        assert perspective.terminology["AI"] == "Artificial Intelligence"
    
    def test_cross_disciplinary_conflict_creation(self):
        """Test creation of CrossDisciplinaryConflict"""
        conflict = CrossDisciplinaryConflict(
            conflict_id="test_conflict_001",
            domains_involved=[ResearchDomain.TECHNOLOGY, ResearchDomain.SCIENCE],
            conflict_type="methodological",
            description="Different validation approaches",
            conflicting_information=[],
            severity=0.7
        )
        
        assert conflict.conflict_id == "test_conflict_001"
        assert len(conflict.domains_involved) == 2
        assert conflict.conflict_type == "methodological"
        assert conflict.severity == 0.7
        assert not conflict.resolved  # Default value
    
    def test_cross_disciplinary_integration_creation(self):
        """Test creation of CrossDisciplinaryIntegration"""
        integration = CrossDisciplinaryIntegration(
            primary_domains=[ResearchDomain.TECHNOLOGY, ResearchDomain.BUSINESS],
            disciplinary_perspectives=[],
            integration_strategy="synthesis",
            conflicts_identified=[],
            conflicts_resolved=[],
            synthesis_approach="evidence_based",
            coherence_score=0.8
        )
        
        assert len(integration.primary_domains) == 2
        assert integration.integration_strategy == "synthesis"
        assert integration.coherence_score == 0.8
        assert integration.synthesis_approach == "evidence_based"


if __name__ == "__main__":
    pytest.main([__file__])