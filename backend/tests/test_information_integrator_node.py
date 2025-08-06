"""
Unit tests for information integrator node.
Tests the LangGraph node implementation for information integration.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio

from workflow.information_integrator_node import (
    information_integrator_node, _fallback_integration,
    validate_integration_state, get_integration_statistics
)
from models.core import (
    TTDRState, Draft, RetrievedInfo, InformationGap, Section, 
    ResearchStructure, DraftMetadata, Source, GapType, Priority,
    ComplexityLevel, ResearchDomain, ResearchRequirements
)

class TestInformationIntegratorNode:
    """Test cases for information_integrator_node"""
    
    @pytest.fixture
    def sample_section(self):
        """Create a sample section"""
        return Section(
            id="intro",
            title="Introduction",
            content="Initial content",
            estimated_length=100
        )
    
    @pytest.fixture
    def sample_draft(self, sample_section):
        """Create a sample draft"""
        structure = ResearchStructure(
            sections=[sample_section],
            estimated_length=1000,
            complexity_level=ComplexityLevel.INTERMEDIATE,
            domain=ResearchDomain.TECHNOLOGY
        )
        
        return Draft(
            id="draft_1",
            topic="AI Research",
            structure=structure,
            content={"intro": "Initial content"},
            metadata=DraftMetadata(),
            quality_score=0.5,
            iteration=1
        )
    
    @pytest.fixture
    def sample_source(self):
        """Create a sample source"""
        return Source(
            url="https://example.com/ai",
            title="AI Article",
            domain="example.com",
            credibility_score=0.8
        )
    
    @pytest.fixture
    def sample_retrieved_info(self, sample_source):
        """Create sample retrieved information"""
        return RetrievedInfo(
            source=sample_source,
            content="AI is transforming industries worldwide.",
            relevance_score=0.9,
            credibility_score=0.8,
            gap_id="gap_1"
        )
    
    @pytest.fixture
    def sample_gap(self):
        """Create a sample information gap"""
        return InformationGap(
            id="gap_1",
            section_id="intro",
            gap_type=GapType.CONTENT,
            description="Need more AI information",
            priority=Priority.HIGH
        )
    
    @pytest.fixture
    def sample_state(self, sample_draft, sample_retrieved_info, sample_gap):
        """Create a sample TTDRState"""
        return TTDRState(
            topic="AI Research",
            requirements=ResearchRequirements(),
            current_draft=sample_draft,
            information_gaps=[sample_gap],
            retrieved_info=[sample_retrieved_info],
            iteration_count=1,
            quality_metrics=None,
            evolution_history=[],
            final_report=None,
            error_log=[]
        )
    
    @patch('workflow.information_integrator_node.KimiK2InformationIntegrator')
    def test_information_integrator_node_success(self, mock_integrator_class, sample_state):
        """Test successful information integration"""
        # Mock the integrator
        mock_integrator = Mock()
        mock_integrator_class.return_value = mock_integrator
        mock_integrator.integration_history = [{"test": "history"}]
        
        # Mock the async integration method
        updated_draft = sample_state["current_draft"]
        updated_draft.content["intro"] = "Updated content with AI information"
        updated_draft.iteration = 2
        
        async def mock_integrate(*args, **kwargs):
            return updated_draft
        
        mock_integrator.integrate_information = mock_integrate
        
        # Mock asyncio.run to avoid event loop issues
        with patch('asyncio.run') as mock_run:
            mock_run.return_value = updated_draft
            
            # Test the node
            result = information_integrator_node(sample_state)
        
        # Verify results
        assert result["iteration_count"] == 2
        assert result["current_draft"].content["intro"] == "Updated content with AI information"
        assert "integration_history" in result
        assert len(result["integration_history"]) == 1
    
    def test_information_integrator_node_no_draft(self):
        """Test node with no current draft"""
        state = TTDRState(
            topic="Test",
            requirements=ResearchRequirements(),
            current_draft=None,
            information_gaps=[],
            retrieved_info=[],
            iteration_count=0,
            quality_metrics=None,
            evolution_history=[],
            final_report=None,
            error_log=[]
        )
        
        result = information_integrator_node(state)
        
        # Verify error handling
        assert "No draft available for integration" in result["error_log"]
        assert result["current_draft"] is None
    
    def test_information_integrator_node_no_retrieved_info(self, sample_state):
        """Test node with no retrieved information"""
        sample_state["retrieved_info"] = []
        
        result = information_integrator_node(sample_state)
        
        # Should return unchanged state
        assert result == sample_state
    
    @patch('workflow.information_integrator_node.KimiK2InformationIntegrator')
    def test_information_integrator_node_with_error(self, mock_integrator_class, sample_state):
        """Test node with integration error and fallback"""
        # Mock the integrator to raise an error
        mock_integrator = Mock()
        mock_integrator_class.return_value = mock_integrator
        
        async def mock_integrate_error(*args, **kwargs):
            raise Exception("Integration failed")
        
        mock_integrator.integrate_information = mock_integrate_error
        
        # Mock asyncio.run
        with patch('asyncio.run') as mock_run:
            mock_run.side_effect = Exception("Integration failed")
            
            # Test the node
            result = information_integrator_node(sample_state)
        
        # Verify fallback was used
        assert result["iteration_count"] == 2
        assert "Integration error (fallback used)" in result["error_log"][0]
        assert "transforming industries" in result["current_draft"].content["intro"]
    
    def test_fallback_integration_success(self, sample_draft, sample_retrieved_info, sample_gap):
        """Test fallback integration method"""
        result = _fallback_integration(
            draft=sample_draft,
            retrieved_info=[sample_retrieved_info],
            gaps=[sample_gap]
        )
        
        # Verify integration
        assert result.iteration == sample_draft.iteration + 1
        assert "transforming industries" in result.content["intro"]
        assert "AI Article" in result.content["intro"]
        assert "https://example.com/ai" in result.content["intro"]
    
    def test_fallback_integration_empty_section(self, sample_draft, sample_retrieved_info, sample_gap):
        """Test fallback integration with empty section"""
        # Clear the section content
        sample_draft.content["intro"] = ""
        
        result = _fallback_integration(
            draft=sample_draft,
            retrieved_info=[sample_retrieved_info],
            gaps=[sample_gap]
        )
        
        # Verify new content was added
        assert "transforming industries" in result.content["intro"]
        assert result.content["intro"].startswith("AI is transforming")
    
    def test_fallback_integration_multiple_info(self, sample_draft, sample_retrieved_info, 
                                              sample_gap, sample_source):
        """Test fallback integration with multiple retrieved info items"""
        # Create additional retrieved info
        info2 = RetrievedInfo(
            source=sample_source,
            content="Machine learning is a subset of AI.",
            relevance_score=0.8,
            credibility_score=0.7,
            gap_id="gap_1"
        )
        
        result = _fallback_integration(
            draft=sample_draft,
            retrieved_info=[sample_retrieved_info, info2],
            gaps=[sample_gap]
        )
        
        # Verify both pieces of information were integrated
        assert "transforming industries" in result.content["intro"]
        assert "subset of AI" in result.content["intro"]
        assert result.content["intro"].count("Source:") == 2
    
    def test_validate_integration_state_valid(self, sample_state):
        """Test validation with valid state"""
        errors = validate_integration_state(sample_state)
        assert len(errors) == 0
    
    def test_validate_integration_state_missing_draft(self, sample_state):
        """Test validation with missing draft"""
        sample_state["current_draft"] = None
        errors = validate_integration_state(sample_state)
        assert "No current draft available" in errors
    
    def test_validate_integration_state_missing_info(self, sample_state):
        """Test validation with missing retrieved info"""
        sample_state["retrieved_info"] = []
        errors = validate_integration_state(sample_state)
        assert "No retrieved information available" in errors
    
    def test_validate_integration_state_missing_gaps(self, sample_state):
        """Test validation with missing gaps"""
        sample_state["information_gaps"] = []
        errors = validate_integration_state(sample_state)
        assert "No information gaps defined" in errors
    
    def test_validate_integration_state_unassociated_info(self, sample_state, sample_source):
        """Test validation with unassociated retrieved info"""
        # Add info without gap_id
        unassociated_info = RetrievedInfo(
            source=sample_source,
            content="Unassociated content",
            relevance_score=0.5,
            credibility_score=0.5,
            gap_id=None
        )
        sample_state["retrieved_info"].append(unassociated_info)
        
        errors = validate_integration_state(sample_state)
        assert "1 retrieved items not associated with gaps" in errors
    
    def test_get_integration_statistics_complete(self, sample_state):
        """Test integration statistics with complete state"""
        # Add integration history
        sample_state["integration_history"] = [
            {"source_url": "https://example.com/1"},
            {"source_url": "https://example.com/2"},
            {"source_url": "https://example.com/1"}  # Duplicate
        ]
        
        stats = get_integration_statistics(sample_state)
        
        # Verify statistics
        assert stats["total_retrieved_items"] == 1
        assert stats["total_gaps"] == 1
        assert stats["iteration_count"] == 1
        assert stats["draft_sections"] == 1
        assert stats["populated_sections"] == 1
        assert stats["total_word_count"] == 2  # "Initial content"
        assert stats["integration_operations"] == 3
        assert stats["unique_sources_integrated"] == 2
    
    def test_get_integration_statistics_minimal(self):
        """Test integration statistics with minimal state"""
        minimal_state = TTDRState(
            topic="Test",
            requirements=ResearchRequirements(),
            current_draft=None,
            information_gaps=[],
            retrieved_info=[],
            iteration_count=0,
            quality_metrics=None,
            evolution_history=[],
            final_report=None,
            error_log=[]
        )
        
        stats = get_integration_statistics(minimal_state)
        
        # Verify minimal statistics
        assert stats["total_retrieved_items"] == 0
        assert stats["total_gaps"] == 0
        assert stats["iteration_count"] == 0
        assert stats["draft_sections"] == 0
        assert stats["populated_sections"] == 0
        assert stats["total_word_count"] == 0

class TestIntegrationNodeEdgeCases:
    """Test edge cases for information integrator node"""
    
    def test_integration_with_missing_section(self, sample_state):
        """Test integration when gap references non-existent section"""
        # Create gap with invalid section_id
        invalid_gap = InformationGap(
            id="gap_invalid",
            section_id="nonexistent",
            gap_type=GapType.CONTENT,
            description="Invalid gap",
            priority=Priority.LOW
        )
        
        sample_state["information_gaps"] = [invalid_gap]
        
        # Should not crash, but may not integrate anything
        result = _fallback_integration(
            draft=sample_state["current_draft"],
            retrieved_info=sample_state["retrieved_info"],
            gaps=[invalid_gap]
        )
        
        # Original content should be preserved
        assert result.content["intro"] == "Initial content"
    
    def test_integration_with_empty_content(self, sample_state, sample_source):
        """Test integration with empty retrieved content"""
        empty_info = RetrievedInfo(
            source=sample_source,
            content="",
            relevance_score=0.1,
            credibility_score=0.1,
            gap_id="gap_1"
        )
        
        result = _fallback_integration(
            draft=sample_state["current_draft"],
            retrieved_info=[empty_info],
            gaps=sample_state["information_gaps"]
        )
        
        # Should handle empty content gracefully
        assert result.iteration == sample_state["current_draft"].iteration + 1
    
    def test_integration_with_large_content(self, sample_state, sample_source):
        """Test integration with very large retrieved content"""
        large_content = "Large content. " * 1000  # 2000 words
        
        large_info = RetrievedInfo(
            source=sample_source,
            content=large_content,
            relevance_score=0.9,
            credibility_score=0.8,
            gap_id="gap_1"
        )
        
        result = _fallback_integration(
            draft=sample_state["current_draft"],
            retrieved_info=[large_info],
            gaps=sample_state["information_gaps"]
        )
        
        # Should handle large content
        assert len(result.content["intro"]) > len(sample_state["current_draft"].content["intro"])
        assert result.metadata.word_count > 1000

@pytest.mark.integration
class TestInformationIntegratorNodeIntegration:
    """Integration tests for information integrator node"""
    
    def test_node_in_workflow_context(self):
        """Test node behavior in workflow context"""
        # This would test the node as part of the complete workflow
        pass
    
    def test_state_persistence(self):
        """Test that state changes persist correctly"""
        # This would test state management across node executions
        pass