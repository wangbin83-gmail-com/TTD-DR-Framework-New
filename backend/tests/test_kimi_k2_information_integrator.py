"""
Unit tests for Kimi K2 Information Integrator service.
Tests intelligent content integration, contextual placement, and conflict resolution.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import uuid

from services.kimi_k2_information_integrator import (
    KimiK2InformationIntegrator, IntegrationContext, ConflictResolution
)
from models.core import (
    Draft, RetrievedInfo, InformationGap, Section, ResearchStructure,
    DraftMetadata, Source, GapType, Priority, ComplexityLevel, ResearchDomain
)
from services.kimi_k2_client import KimiK2Response, KimiK2Error

class TestKimiK2InformationIntegrator:
    """Test cases for KimiK2InformationIntegrator"""
    
    @pytest.fixture
    def sample_section(self):
        """Create a sample section for testing"""
        return Section(
            id="intro",
            title="Introduction",
            content="This is the introduction section.",
            estimated_length=100
        )
    
    @pytest.fixture
    def sample_draft(self, sample_section):
        """Create a sample draft for testing"""
        structure = ResearchStructure(
            sections=[sample_section],
            estimated_length=1000,
            complexity_level=ComplexityLevel.INTERMEDIATE,
            domain=ResearchDomain.TECHNOLOGY
        )
        
        return Draft(
            id="draft_1",
            topic="Artificial Intelligence",
            structure=structure,
            content={"intro": "This is the introduction section."},
            metadata=DraftMetadata(),
            quality_score=0.5,
            iteration=1
        )
    
    @pytest.fixture
    def sample_source(self):
        """Create a sample source for testing"""
        return Source(
            url="https://example.com/article",
            title="AI Research Article",
            domain="example.com",
            credibility_score=0.8
        )
    
    @pytest.fixture
    def sample_retrieved_info(self, sample_source):
        """Create sample retrieved information"""
        return RetrievedInfo(
            source=sample_source,
            content="Artificial Intelligence has revolutionized many industries.",
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
            description="Need more information about AI applications",
            priority=Priority.HIGH
        )
    
    @pytest.fixture
    def integrator(self):
        """Create integrator instance with mocked Kimi K2 client"""
        integrator = KimiK2InformationIntegrator()
        integrator.kimi_client = AsyncMock()
        return integrator
    
    @pytest.mark.asyncio
    async def test_integrate_information_success(self, integrator, sample_draft, 
                                               sample_retrieved_info, sample_gap):
        """Test successful information integration"""
        # Mock Kimi K2 response
        mock_response = KimiK2Response(
            content="This is the introduction section.\n\nArtificial Intelligence has revolutionized many industries, transforming how we work and live.\n\nSource: AI Research Article (https://example.com/article)",
            usage={"tokens": 100},
            model="kimi-k2",
            finish_reason="stop"
        )
        integrator.kimi_client.generate_text.return_value = mock_response
        
        # Test integration
        result = await integrator.integrate_information(
            draft=sample_draft,
            retrieved_info=[sample_retrieved_info],
            gaps=[sample_gap]
        )
        
        # Verify results
        assert result.iteration == sample_draft.iteration + 1
        assert result.metadata.updated_at > sample_draft.metadata.updated_at
        assert len(result.content["intro"]) > len(sample_draft.content["intro"])
        assert "revolutionized many industries" in result.content["intro"]
        
        # Verify integration history
        assert len(integrator.integration_history) == 1
        history_entry = integrator.integration_history[0]
        assert history_entry["gap_id"] == "gap_1"
        assert history_entry["section_id"] == "intro"
        assert history_entry["source_url"] == "https://example.com/article"
    
    @pytest.mark.asyncio
    async def test_integrate_information_with_kimi_error(self, integrator, sample_draft,
                                                       sample_retrieved_info, sample_gap):
        """Test integration with Kimi K2 error fallback"""
        # Mock Kimi K2 error
        integrator.kimi_client.generate_text.side_effect = KimiK2Error("API Error")
        
        # Test integration
        result = await integrator.integrate_information(
            draft=sample_draft,
            retrieved_info=[sample_retrieved_info],
            gaps=[sample_gap]
        )
        
        # Verify fallback integration occurred
        assert result.iteration == sample_draft.iteration + 1
        assert "revolutionized many industries" in result.content["intro"]
        assert "Source: AI Research Article" in result.content["intro"]
    
    @pytest.mark.asyncio
    async def test_integrate_single_info_success(self, integrator, sample_retrieved_info, 
                                               sample_gap, sample_section):
        """Test single information integration"""
        # Create integration context
        context = IntegrationContext(
            section=sample_section,
            surrounding_content="Previous context...",
            related_sections=[],
            topic="Artificial Intelligence"
        )
        
        # Mock Kimi K2 response
        mock_response = KimiK2Response(
            content="Enhanced introduction with AI revolutionizing industries.",
            usage={"tokens": 50}
        )
        integrator.kimi_client.generate_text.return_value = mock_response
        
        # Test integration
        result = await integrator._integrate_single_info(
            current_content="Original content.",
            info=sample_retrieved_info,
            gap=sample_gap,
            context=context
        )
        
        # Verify results
        assert result == "Enhanced introduction with AI revolutionizing industries."
        assert integrator.kimi_client.generate_text.called
        
        # Verify integration history
        assert len(integrator.integration_history) == 1
    
    def test_build_integration_prompt(self, integrator, sample_retrieved_info, 
                                    sample_gap, sample_section):
        """Test integration prompt building"""
        context = IntegrationContext(
            section=sample_section,
            surrounding_content="Context content...",
            related_sections=[],
            topic="Artificial Intelligence"
        )
        
        prompt = integrator._build_integration_prompt(
            current_content="Original content",
            info=sample_retrieved_info,
            gap=sample_gap,
            context=context
        )
        
        # Verify prompt contains key elements
        assert "Artificial Intelligence" in prompt
        assert "Introduction" in prompt
        assert "Need more information about AI applications" in prompt
        assert "Original content" in prompt
        assert "revolutionized many industries" in prompt
        assert "https://example.com/article" in prompt
    
    def test_parse_integration_response(self, integrator):
        """Test parsing of Kimi K2 integration response"""
        # Test with clean response
        clean_response = "This is the integrated content."
        result = integrator._parse_integration_response(clean_response)
        assert result == "This is the integrated content."
        
        # Test with meta-commentary
        messy_response = """**Integration Strategy:** Adding new content
        
This is the integrated content.

Note: This addresses the information gap."""
        
        result = integrator._parse_integration_response(messy_response)
        assert "Integration Strategy" not in result
        assert "Note:" not in result
        assert "This is the integrated content." in result
    
    def test_fallback_integration(self, integrator, sample_retrieved_info):
        """Test fallback integration method"""
        # Test with existing content
        result = integrator._fallback_integration(
            current_content="Existing content.",
            info=sample_retrieved_info
        )
        
        assert "Existing content." in result
        assert "revolutionized many industries" in result
        assert "AI Research Article" in result
        assert "https://example.com/article" in result
        
        # Test with empty content
        result = integrator._fallback_integration(
            current_content="",
            info=sample_retrieved_info
        )
        
        assert result.startswith("Artificial Intelligence has revolutionized")
        assert "Source:" in result
    
    @pytest.mark.asyncio
    async def test_resolve_conflicts_success(self, integrator, sample_retrieved_info, 
                                           sample_section):
        """Test conflict resolution with Kimi K2"""
        context = IntegrationContext(
            section=sample_section,
            surrounding_content="",
            related_sections=[],
            topic="AI"
        )
        
        # Mock Kimi K2 response
        mock_response = KimiK2Response(
            content="Resolved content that addresses the conflict.",
            usage={"tokens": 30}
        )
        integrator.kimi_client.generate_text.return_value = mock_response
        
        # Test conflict resolution
        result = await integrator.resolve_conflicts(
            existing_content="Old information about AI.",
            new_info=sample_retrieved_info,
            context=context
        )
        
        # Verify results
        assert isinstance(result, ConflictResolution)
        assert result.original_content == "Old information about AI."
        assert result.new_content == sample_retrieved_info.content
        assert result.resolved_content == "Resolved content that addresses the conflict."
        assert result.resolution_strategy == "kimi_k2_intelligent"
    
    @pytest.mark.asyncio
    async def test_resolve_conflicts_fallback(self, integrator, sample_retrieved_info, 
                                            sample_section):
        """Test conflict resolution fallback"""
        context = IntegrationContext(
            section=sample_section,
            surrounding_content="",
            related_sections=[],
            topic="AI"
        )
        
        # Mock Kimi K2 error
        integrator.kimi_client.generate_text.side_effect = KimiK2Error("API Error")
        
        # Test conflict resolution
        result = await integrator.resolve_conflicts(
            existing_content="Old information about AI.",
            new_info=sample_retrieved_info,
            context=context
        )
        
        # Verify fallback resolution
        assert isinstance(result, ConflictResolution)
        assert result.resolution_strategy.startswith("credibility_based")
        assert len(result.resolved_content) > 0
    
    def test_fallback_conflict_resolution(self, integrator, sample_retrieved_info):
        """Test fallback conflict resolution logic"""
        # Test with high credibility new info
        high_cred_info = RetrievedInfo(
            source=sample_retrieved_info.source,
            content="New high credibility content",
            relevance_score=0.9,
            credibility_score=0.9
        )
        
        result = integrator._fallback_conflict_resolution(
            existing_content="Old content",
            new_info=high_cred_info
        )
        
        assert result.resolution_strategy == "credibility_based_new_preferred"
        assert result.resolved_content.startswith("New high credibility content")
        
        # Test with low credibility new info
        low_cred_info = RetrievedInfo(
            source=sample_retrieved_info.source,
            content="New low credibility content",
            relevance_score=0.5,
            credibility_score=0.3
        )
        
        result = integrator._fallback_conflict_resolution(
            existing_content="Old content",
            new_info=low_cred_info
        )
        
        assert result.resolution_strategy == "credibility_based_existing_preferred"
        assert result.resolved_content.startswith("Old content")
        assert "Additional perspective" in result.resolved_content
    
    @pytest.mark.asyncio
    async def test_ensure_global_coherence(self, integrator, sample_draft):
        """Test global coherence checking"""
        # Mock Kimi K2 structured response
        mock_response = {
            "coherence_issues": ["Transition between sections unclear"],
            "suggested_improvements": ["Add connecting sentences"],
            "overall_coherence_score": 0.6
        }
        integrator.kimi_client.generate_structured_response.return_value = mock_response
        
        # Test coherence check
        result = await integrator._ensure_global_coherence(sample_draft)
        
        # Verify coherence check was performed
        assert integrator.kimi_client.generate_structured_response.called
        assert result.id == sample_draft.id  # Draft should be returned
    
    def test_build_coherence_prompt(self, integrator, sample_draft):
        """Test coherence assessment prompt building"""
        prompt = integrator._build_coherence_prompt(sample_draft)
        
        # Verify prompt contains key elements
        assert sample_draft.topic in prompt
        assert "Introduction" in prompt
        assert "coherence and flow" in prompt
        assert "JSON" in prompt
    
    def test_create_integration_context(self, integrator, sample_draft, sample_section):
        """Test integration context creation"""
        # Add more sections to test surrounding content
        additional_section = Section(
            id="methods",
            title="Methods",
            content="Methods content",
            estimated_length=200
        )
        sample_draft.structure.sections.append(additional_section)
        sample_draft.content["methods"] = "Methods content here"
        
        context = integrator._create_integration_context(sample_draft, sample_section)
        
        # Verify context creation
        assert context.section.id == "intro"
        assert context.topic == "Artificial Intelligence"
        assert len(context.related_sections) > 0
        assert "Methods" in context.surrounding_content
    
    def test_group_info_by_gap(self, integrator, sample_retrieved_info, sample_gap):
        """Test grouping retrieved info by gaps"""
        # Create additional info and gaps
        info2 = RetrievedInfo(
            source=sample_retrieved_info.source,
            content="Additional content",
            relevance_score=0.7,
            credibility_score=0.6,
            gap_id="gap_2"
        )
        
        gap2 = InformationGap(
            id="gap_2",
            section_id="methods",
            gap_type=GapType.EVIDENCE,
            description="Need evidence",
            priority=Priority.MEDIUM
        )
        
        # Test grouping
        result = integrator._group_info_by_gap(
            retrieved_info=[sample_retrieved_info, info2],
            gaps=[sample_gap, gap2]
        )
        
        # Verify grouping
        assert "gap_1" in result
        assert "gap_2" in result
        assert len(result["gap_1"]) == 1
        assert len(result["gap_2"]) == 1
        assert result["gap_1"][0].content == sample_retrieved_info.content
    
    def test_find_section_by_id(self, integrator, sample_draft):
        """Test finding section by ID"""
        # Test existing section
        section = integrator._find_section_by_id(
            sample_draft.structure.sections, 
            "intro"
        )
        assert section is not None
        assert section.id == "intro"
        
        # Test non-existing section
        section = integrator._find_section_by_id(
            sample_draft.structure.sections, 
            "nonexistent"
        )
        assert section is None
    
    def test_copy_draft(self, integrator, sample_draft):
        """Test draft copying"""
        copied_draft = integrator._copy_draft(sample_draft)
        
        # Verify copy
        assert copied_draft.id == sample_draft.id
        assert copied_draft.topic == sample_draft.topic
        assert copied_draft.iteration == sample_draft.iteration + 1
        assert copied_draft.metadata.updated_at > sample_draft.metadata.updated_at
        
        # Verify it's a separate object
        copied_draft.content["intro"] = "Modified content"
        assert sample_draft.content["intro"] != copied_draft.content["intro"]
    
    def test_calculate_word_count(self, integrator, sample_draft):
        """Test word count calculation"""
        # Add more content
        sample_draft.content["intro"] = "This is a test with multiple words."
        sample_draft.content["conclusion"] = "Final thoughts here."
        
        word_count = integrator._calculate_word_count(sample_draft)
        
        # Verify word count (8 + 3 = 11 words)
        assert word_count == 11

class TestIntegrationContext:
    """Test cases for IntegrationContext"""
    
    def test_integration_context_creation(self):
        """Test IntegrationContext creation"""
        section = Section(id="test", title="Test", content="Content")
        related_sections = [Section(id="related", title="Related", content="Related content")]
        
        context = IntegrationContext(
            section=section,
            surrounding_content="Surrounding content",
            related_sections=related_sections,
            topic="Test Topic"
        )
        
        assert context.section.id == "test"
        assert context.surrounding_content == "Surrounding content"
        assert len(context.related_sections) == 1
        assert context.topic == "Test Topic"

class TestConflictResolution:
    """Test cases for ConflictResolution"""
    
    def test_conflict_resolution_creation(self):
        """Test ConflictResolution creation"""
        resolution = ConflictResolution(
            original_content="Original",
            new_content="New",
            resolved_content="Resolved",
            resolution_strategy="test_strategy"
        )
        
        assert resolution.original_content == "Original"
        assert resolution.new_content == "New"
        assert resolution.resolved_content == "Resolved"
        assert resolution.resolution_strategy == "test_strategy"
        assert isinstance(resolution.timestamp, datetime)

@pytest.mark.integration
class TestKimiK2InformationIntegratorIntegration:
    """Integration tests for KimiK2InformationIntegrator"""
    
    @pytest.mark.asyncio
    async def test_full_integration_workflow(self):
        """Test complete integration workflow"""
        # This would test with actual Kimi K2 API if available
        # For now, it's a placeholder for integration testing
        pass
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms"""
        # This would test various error scenarios
        pass