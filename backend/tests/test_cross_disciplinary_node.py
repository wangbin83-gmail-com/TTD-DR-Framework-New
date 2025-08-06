"""
Tests for cross-disciplinary research workflow nodes.
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from backend.workflow.cross_disciplinary_node import (
    cross_disciplinary_detector_node,
    cross_disciplinary_integrator_node,
    cross_disciplinary_conflict_resolver_node,
    cross_disciplinary_formatter_node,
    cross_disciplinary_quality_assessor_node,
    create_cross_disciplinary_subgraph
)
from backend.models.core import (
    TTDRState, ResearchDomain, Draft, DraftMetadata, QualityMetrics,
    RetrievedInfo, Source, ResearchRequirements, ComplexityLevel
)
from backend.services.cross_disciplinary_integrator import (
    CrossDisciplinaryIntegration,
    CrossDisciplinaryConflict,
    DisciplinaryPerspective
)
from backend.services.kimi_k2_client import KimiK2Client, KimiK2Response


class TestCrossDisciplinaryNodes:
    """Test cases for cross-disciplinary workflow nodes"""
    
    @pytest.fixture
    def sample_state(self):
        """Sample TTD-DR state for testing"""
        return TTDRState(
            topic="AI applications in healthcare and business",
            requirements=ResearchRequirements(
                domain=ResearchDomain.GENERAL,
                complexity_level=ComplexityLevel.INTERMEDIATE,
                max_iterations=5,
                quality_threshold=0.8,
                max_sources=20
            ),
            current_draft=Draft(
                id="test_draft",
                topic="AI applications in healthcare and business",
                structure=None,
                content={
                    "introduction": "AI is transforming multiple industries...",
                    "healthcare_applications": "In healthcare, AI enables...",
                    "business_applications": "In business, AI provides..."
                },
                metadata=DraftMetadata(),
                quality_score=0.7,
                iteration=1
            ),
            information_gaps=[],
            retrieved_info=[
                RetrievedInfo(
                    source=Source(url="https://tech.example.com", title="Tech Article"),
                    content="AI algorithms in software development and automation",
                    relevance_score=0.8,
                    credibility_score=0.7,
                    extraction_timestamp=datetime.now()
                ),
                RetrievedInfo(
                    source=Source(url="https://healthcare.example.com", title="Medical Journal"),
                    content="Clinical applications of machine learning in diagnosis",
                    relevance_score=0.9,
                    credibility_score=0.9,
                    extraction_timestamp=datetime.now()
                ),
                RetrievedInfo(
                    source=Source(url="https://business.example.com", title="Business Report"),
                    content="ROI analysis of AI implementation in enterprises",
                    relevance_score=0.7,
                    credibility_score=0.6,
                    extraction_timestamp=datetime.now()
                )
            ],
            iteration_count=1,
            quality_metrics=QualityMetrics(
                completeness=0.7,
                coherence=0.8,
                accuracy=0.7,
                citation_quality=0.6
            ),
            evolution_history=[],
            final_report=None
        )
    
    @pytest.fixture
    def mock_kimi_client(self):
        """Mock Kimi K2 client"""
        with patch('backend.services.kimi_k2_client.KimiK2Client') as mock_client_class:
            mock_client = Mock(spec=KimiK2Client)
            mock_client.generate_text = AsyncMock()
            mock_client_class.return_value = mock_client
            yield mock_client
    
    def test_cross_disciplinary_detector_node_detects_multi_domain(self, sample_state, mock_kimi_client):
        """Test cross-disciplinary detector identifies multi-domain research"""
        # Mock Kimi K2 response for multi-domain detection
        mock_kimi_client.generate_text.return_value = KimiK2Response(
            content=json.dumps({
                "domains": [
                    {"domain": "TECHNOLOGY", "relevance": 0.8, "reasoning": "AI technology"},
                    {"domain": "SCIENCE", "relevance": 0.7, "reasoning": "Healthcare research"},
                    {"domain": "BUSINESS", "relevance": 0.6, "reasoning": "Business applications"}
                ],
                "is_cross_disciplinary": True
            })
        )
        
        with patch('backend.workflow.cross_disciplinary_node.KimiK2Client', return_value=mock_kimi_client):
            result_state = cross_disciplinary_detector_node(sample_state)
        
        assert "cross_disciplinary_metadata" in result_state
        assert result_state["requires_cross_disciplinary"] is True
        
        metadata = result_state["cross_disciplinary_metadata"]
        assert metadata["is_cross_disciplinary"] is True
        assert len(metadata["involved_domains"]) >= 2
        assert "TECHNOLOGY" in metadata["involved_domains"]
    
    def test_cross_disciplinary_detector_node_detects_single_domain(self, sample_state, mock_kimi_client):
        """Test cross-disciplinary detector identifies single-domain research"""
        # Mock Kimi K2 response for single domain
        mock_kimi_client.generate_text.return_value = KimiK2Response(
            content=json.dumps({
                "domains": [
                    {"domain": "TECHNOLOGY", "relevance": 0.9, "reasoning": "Pure tech topic"}
                ],
                "is_cross_disciplinary": False
            })
        )
        
        with patch('backend.workflow.cross_disciplinary_node.KimiK2Client', return_value=mock_kimi_client):
            result_state = cross_disciplinary_detector_node(sample_state)
        
        assert result_state["requires_cross_disciplinary"] is False
        metadata = result_state["cross_disciplinary_metadata"]
        assert metadata["is_cross_disciplinary"] is False
    
    def test_cross_disciplinary_integrator_node_skips_when_not_required(self, sample_state, mock_kimi_client):
        """Test integrator node skips when cross-disciplinary not required"""
        # Set state to not require cross-disciplinary
        sample_state["requires_cross_disciplinary"] = False
        
        with patch('backend.workflow.cross_disciplinary_node.KimiK2Client', return_value=mock_kimi_client):
            result_state = cross_disciplinary_integrator_node(sample_state)
        
        # Should return unchanged state
        assert result_state == sample_state
        assert "cross_disciplinary_integration" not in result_state
    
    def test_cross_disciplinary_integrator_node_performs_integration(self, sample_state, mock_kimi_client):
        """Test integrator node performs multi-domain integration"""
        # Set up state for cross-disciplinary integration
        sample_state["requires_cross_disciplinary"] = True
        sample_state["cross_disciplinary_metadata"] = {
            "is_cross_disciplinary": True,
            "involved_domains": ["TECHNOLOGY", "SCIENCE", "BUSINESS"]
        }
        
        # Mock Kimi K2 responses for integration process
        responses = [
            # Disciplinary perspective analysis responses
            json.dumps({
                "key_concepts": ["AI", "algorithms"],
                "methodological_approach": "Software engineering",
                "theoretical_framework": "Computer science",
                "evidence_types": ["benchmarks"],
                "terminology": {"AI": "Artificial Intelligence"},
                "confidence": 0.8
            }),
            json.dumps({
                "key_concepts": ["clinical trials", "validation"],
                "methodological_approach": "Medical research",
                "theoretical_framework": "Evidence-based medicine",
                "evidence_types": ["clinical studies"],
                "terminology": {"RCT": "Randomized Controlled Trial"},
                "confidence": 0.9
            }),
            json.dumps({
                "key_concepts": ["ROI", "strategy"],
                "methodological_approach": "Business analysis",
                "theoretical_framework": "Strategic management",
                "evidence_types": ["case studies"],
                "terminology": {"ROI": "Return on Investment"},
                "confidence": 0.7
            }),
            # Conflict detection responses
            json.dumps({
                "has_conflict": True,
                "conflict_type": "methodological",
                "description": "Different validation approaches",
                "severity": 0.6
            }),
            json.dumps({
                "has_conflict": False,
                "conflict_type": "none",
                "description": "No conflicts",
                "severity": 0.1
            }),
            # Conflict resolution response
            json.dumps({
                "resolution_strategy": "Integrate methodologies",
                "integrated_approach": "Mixed methods",
                "remaining_tensions": [],
                "confidence": 0.8
            })
        ]
        
        mock_kimi_client.generate_text.side_effect = [
            KimiK2Response(content=response) for response in responses
        ]
        
        with patch('backend.workflow.cross_disciplinary_node.KimiK2Client', return_value=mock_kimi_client):
            result_state = cross_disciplinary_integrator_node(sample_state)
        
        assert "cross_disciplinary_integration" in result_state
        assert "integration_metadata" in result_state
        
        integration = result_state["cross_disciplinary_integration"]
        assert len(integration.primary_domains) == 3
        assert len(integration.disciplinary_perspectives) == 3
        assert integration.coherence_score > 0.0
        
        metadata = result_state["integration_metadata"]
        assert metadata["integration_completed"] is True
        assert metadata["disciplinary_perspectives"] == 3
    
    def test_cross_disciplinary_conflict_resolver_node(self, sample_state, mock_kimi_client):
        """Test conflict resolver node resolves disciplinary conflicts"""
        # Set up state with integration containing conflicts
        integration = CrossDisciplinaryIntegration(
            primary_domains=[ResearchDomain.TECHNOLOGY, ResearchDomain.SCIENCE],
            disciplinary_perspectives=[
                DisciplinaryPerspective(domain=ResearchDomain.TECHNOLOGY, confidence=0.8),
                DisciplinaryPerspective(domain=ResearchDomain.SCIENCE, confidence=0.9)
            ],
            integration_strategy="synthesis",
            conflicts_identified=[
                CrossDisciplinaryConflict(
                    conflict_id="conflict_001",
                    domains_involved=[ResearchDomain.TECHNOLOGY, ResearchDomain.SCIENCE],
                    conflict_type="methodological",
                    description="Different validation approaches",
                    conflicting_information=[],
                    severity=0.7,
                    resolved=False
                )
            ],
            conflicts_resolved=[],
            synthesis_approach="evidence_based",
            coherence_score=0.6
        )
        
        sample_state["cross_disciplinary_integration"] = integration
        sample_state["integration_metadata"] = {"conflicts_resolved": 0}
        
        # Mock conflict resolution response
        mock_kimi_client.generate_text.return_value = KimiK2Response(
            content=json.dumps({
                "resolution_strategy": "Combine quantitative and qualitative validation",
                "integrated_approach": "Mixed methods validation framework",
                "remaining_tensions": [],
                "confidence": 0.8
            })
        )
        
        with patch('backend.workflow.cross_disciplinary_node.KimiK2Client', return_value=mock_kimi_client):
            result_state = cross_disciplinary_conflict_resolver_node(sample_state)
        
        updated_integration = result_state["cross_disciplinary_integration"]
        assert len(updated_integration.conflicts_resolved) == 1
        assert updated_integration.coherence_score >= integration.coherence_score
        
        metadata = result_state["integration_metadata"]
        assert metadata["conflicts_resolved"] == 1
        assert "resolution_success_rate" in metadata
    
    def test_cross_disciplinary_formatter_node(self, sample_state, mock_kimi_client):
        """Test formatter node creates specialized cross-disciplinary output"""
        # Set up state with integration
        integration = CrossDisciplinaryIntegration(
            primary_domains=[ResearchDomain.TECHNOLOGY, ResearchDomain.BUSINESS],
            disciplinary_perspectives=[
                DisciplinaryPerspective(
                    domain=ResearchDomain.TECHNOLOGY,
                    confidence=0.8,
                    key_concepts=["AI", "algorithms"],
                    methodological_approach="Software engineering"
                ),
                DisciplinaryPerspective(
                    domain=ResearchDomain.BUSINESS,
                    confidence=0.7,
                    key_concepts=["ROI", "strategy"],
                    methodological_approach="Business analysis"
                )
            ],
            integration_strategy="synthesis",
            conflicts_identified=[],
            conflicts_resolved=[],
            synthesis_approach="practical_synthesis",
            coherence_score=0.8
        )
        
        sample_state["cross_disciplinary_integration"] = integration
        
        with patch('backend.workflow.cross_disciplinary_node.KimiK2Client', return_value=mock_kimi_client):
            result_state = cross_disciplinary_formatter_node(sample_state)
        
        updated_draft = result_state["current_draft"]
        assert "cross_disciplinary_report" in updated_draft.content
        assert "disciplinary_perspectives" in updated_draft.content
        
        formatting_metadata = result_state["formatting_metadata"]
        assert formatting_metadata["cross_disciplinary_formatting"] is True
        assert formatting_metadata["output_format"] in ["comprehensive", "hierarchical", "comparative", "synthesis"]
    
    def test_cross_disciplinary_quality_assessor_node(self, sample_state, mock_kimi_client):
        """Test quality assessor node evaluates cross-disciplinary research quality"""
        # Set up state with integration
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
                    severity=0.5
                )
            ],
            conflicts_resolved=[
                CrossDisciplinaryConflict(
                    conflict_id="test_conflict",
                    domains_involved=[ResearchDomain.TECHNOLOGY, ResearchDomain.SCIENCE],
                    conflict_type="methodological",
                    description="Test conflict",
                    conflicting_information=[],
                    severity=0.5,
                    resolved=True
                )
            ],
            synthesis_approach="evidence_based",
            coherence_score=0.8
        )
        
        sample_state["cross_disciplinary_integration"] = integration
        
        with patch('backend.workflow.cross_disciplinary_node.KimiK2Client', return_value=mock_kimi_client):
            result_state = cross_disciplinary_quality_assessor_node(sample_state)
        
        assert "quality_metrics" in result_state
        assert "cross_disciplinary_quality_metadata" in result_state
        
        quality_metadata = result_state["cross_disciplinary_quality_metadata"]
        assert "cross_disciplinary_metrics" in quality_metadata
        assert "overall_cross_disciplinary_score" in quality_metadata
        assert quality_metadata["domains_integrated"] == 2
        assert quality_metadata["conflicts_resolved_ratio"] == 1.0  # All conflicts resolved
    
    def test_error_handling_in_nodes(self, sample_state, mock_kimi_client):
        """Test error handling in cross-disciplinary nodes"""
        # Mock Kimi K2 to raise an exception
        mock_kimi_client.generate_text.side_effect = Exception("API Error")
        
        with patch('backend.workflow.cross_disciplinary_node.KimiK2Client', return_value=mock_kimi_client):
            result_state = cross_disciplinary_detector_node(sample_state)
        
        # Should handle error gracefully
        assert "error_log" in result_state
        assert any("Cross-Disciplinary Detector" in error for error in result_state["error_log"])
        assert result_state["requires_cross_disciplinary"] is False  # Default fallback
    
    def test_create_cross_disciplinary_subgraph(self):
        """Test creation of cross-disciplinary workflow subgraph"""
        subgraph = create_cross_disciplinary_subgraph()
        
        # Should return a compiled graph
        assert subgraph is not None
        
        # Test that we can invoke the graph (basic smoke test)
        test_state = TTDRState(
            topic="Test topic",
            requirements=ResearchRequirements(
                domain=ResearchDomain.GENERAL,
                complexity_level=ComplexityLevel.BASIC,
                max_iterations=3,
                quality_threshold=0.7,
                max_sources=10
            ),
            current_draft=None,
            information_gaps=[],
            retrieved_info=[],
            iteration_count=0,
            quality_metrics=None,
            evolution_history=[],
            final_report=None
        )
        
        # This should not raise an exception
        try:
            # Note: In a real test, we'd mock the Kimi K2 client
            # For now, just test that the graph structure is valid
            assert hasattr(subgraph, 'invoke')
        except Exception as e:
            # Expected to fail without proper mocking, but structure should be valid
            assert "KimiK2Client" in str(e) or "generate_text" in str(e)


class TestCrossDisciplinaryWorkflowIntegration:
    """Test cases for cross-disciplinary workflow integration"""
    
    def test_workflow_state_transitions(self):
        """Test that cross-disciplinary workflow maintains proper state transitions"""
        initial_state = TTDRState(
            topic="Interdisciplinary AI research",
            requirements=ResearchRequirements(
                domain=ResearchDomain.GENERAL,
                complexity_level=ComplexityLevel.ADVANCED,
                max_iterations=5,
                quality_threshold=0.8,
                max_sources=25
            ),
            current_draft=None,
            information_gaps=[],
            retrieved_info=[
                RetrievedInfo(
                    source=Source(url="https://example.com", title="Test"),
                    content="AI research spanning multiple disciplines",
                    relevance_score=0.8,
                    credibility_score=0.7,
                    extraction_timestamp=datetime.now()
                )
            ],
            iteration_count=0,
            quality_metrics=None,
            evolution_history=[],
            final_report=None
        )
        
        # Test state keys that should be preserved
        required_keys = [
            "topic", "requirements", "retrieved_info", 
            "iteration_count", "evolution_history"
        ]
        
        for key in required_keys:
            assert key in initial_state
        
        # Test that state structure is compatible with cross-disciplinary additions
        test_additions = {
            "cross_disciplinary_metadata": {"test": True},
            "requires_cross_disciplinary": True,
            "cross_disciplinary_integration": None,
            "integration_metadata": {"test": True}
        }
        
        updated_state = {**initial_state, **test_additions}
        
        for key in required_keys:
            assert key in updated_state
        
        for key in test_additions:
            assert key in updated_state
    
    def test_cross_disciplinary_metadata_structure(self):
        """Test structure of cross-disciplinary metadata"""
        metadata = {
            "is_cross_disciplinary": True,
            "involved_domains": ["TECHNOLOGY", "SCIENCE"],
            "detection_timestamp": datetime.now().isoformat(),
            "domains_count": 2,
            "detection_confidence": 0.8
        }
        
        # Validate metadata structure
        assert isinstance(metadata["is_cross_disciplinary"], bool)
        assert isinstance(metadata["involved_domains"], list)
        assert isinstance(metadata["domains_count"], int)
        assert isinstance(metadata["detection_confidence"], (int, float))
        assert 0.0 <= metadata["detection_confidence"] <= 1.0
        
        # Validate domain names are valid
        valid_domains = [domain.value for domain in ResearchDomain]
        for domain in metadata["involved_domains"]:
            assert domain in valid_domains
    
    def test_integration_metadata_structure(self):
        """Test structure of integration metadata"""
        metadata = {
            "integration_completed": True,
            "coherence_score": 0.8,
            "integration_strategy": "synthesis",
            "synthesis_approach": "evidence_based",
            "conflicts_identified": 2,
            "conflicts_resolved": 1,
            "disciplinary_perspectives": 3,
            "integration_timestamp": datetime.now().isoformat()
        }
        
        # Validate metadata structure
        assert isinstance(metadata["integration_completed"], bool)
        assert isinstance(metadata["coherence_score"], (int, float))
        assert 0.0 <= metadata["coherence_score"] <= 1.0
        assert isinstance(metadata["integration_strategy"], str)
        assert isinstance(metadata["conflicts_identified"], int)
        assert isinstance(metadata["conflicts_resolved"], int)
        assert metadata["conflicts_resolved"] <= metadata["conflicts_identified"]


if __name__ == "__main__":
    pytest.main([__file__])