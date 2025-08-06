"""
Unit tests for Kimi K2 Coherence Manager service.
Tests coherence maintenance, citation management, and conflict resolution.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import uuid

from services.kimi_k2_coherence_manager import (
    KimiK2CoherenceManager, Citation, CoherenceIssue, CoherenceReport
)
from models.core import (
    Draft, RetrievedInfo, Section, Source, ResearchStructure, DraftMetadata, ComplexityLevel
)
from services.kimi_k2_client import KimiK2Response


@pytest.fixture
def sample_draft():
    """Create a sample draft for testing"""
    sections = [
        Section(id="intro", title="Introduction", content="", estimated_length=500),
        Section(id="methods", title="Methods", content="", estimated_length=800),
        Section(id="results", title="Results", content="", estimated_length=1000)
    ]
    
    structure = ResearchStructure(
        sections=sections,
        relationships=[],
        estimated_length=2300,
        complexity_level=ComplexityLevel.INTERMEDIATE
    )
    
    content = {
        "intro": "This research explores the impact of AI on education. Recent studies show significant changes.",
        "methods": "We conducted a comprehensive literature review. The methodology involved systematic analysis.",
        "results": "Our findings indicate positive outcomes. However, some challenges remain unresolved."
    }
    
    metadata = DraftMetadata(
        created_at=datetime.now(),
        updated_at=datetime.now(),
        author="test_author",
        version="1.0",
        word_count=150
    )
    
    return Draft(
        id=str(uuid.uuid4()),
        topic="AI Impact on Education",
        structure=structure,
        content=content,
        metadata=metadata,
        quality_score=0.7,
        iteration=1
    )


@pytest.fixture
def sample_retrieved_info():
    """Create sample retrieved information for testing"""
    sources = [
        Source(
            title="AI in Education Research",
            url="https://example.com/ai-education",
            domain="example.com",
            credibility_score=0.8,
            last_accessed=datetime.now()
        ),
        Source(
            title="Educational Technology Trends",
            url="https://example.com/edtech-trends",
            domain="example.com",
            credibility_score=0.7,
            last_accessed=datetime.now()
        )
    ]
    
    return [
        RetrievedInfo(
            source=sources[0],
            content="Artificial intelligence is transforming educational practices worldwide.",
            relevance_score=0.9,
            credibility_score=0.8,
            extraction_timestamp=datetime.now()
        ),
        RetrievedInfo(
            source=sources[1],
            content="Educational technology adoption has accelerated significantly in recent years.",
            relevance_score=0.8,
            credibility_score=0.7,
            extraction_timestamp=datetime.now()
        )
    ]


@pytest.fixture
def coherence_manager():
    """Create a coherence manager instance for testing"""
    return KimiK2CoherenceManager()


class TestKimiK2CoherenceManager:
    """Test cases for Kimi K2 Coherence Manager"""
    
    def test_coherence_manager_creation(self, coherence_manager):
        """Test that coherence manager can be created"""
        assert coherence_manager is not None
        assert hasattr(coherence_manager, 'kimi_client')
        assert hasattr(coherence_manager, 'citations')
        assert hasattr(coherence_manager, 'coherence_history')
    
    @pytest.mark.asyncio
    async def test_maintain_coherence_failure_fallback(self, coherence_manager, sample_draft):
        """Test coherence maintenance with Kimi K2 failure and fallback"""
        
        with patch.object(coherence_manager.kimi_client, 'generate_structured_response', new_callable=AsyncMock) as mock_structured:
            mock_structured.side_effect = Exception("API Error")
            
            # Test coherence maintenance with failure
            updated_draft, report = await coherence_manager.maintain_coherence(sample_draft)
            
            # Verify fallback behavior
            assert isinstance(updated_draft, Draft)
            assert isinstance(report, CoherenceReport)
            assert report.overall_score > 0.5  # Fallback score should be reasonable
            assert updated_draft.id == sample_draft.id
    
    @pytest.mark.asyncio
    async def test_manage_citations_failure_fallback(self, coherence_manager, sample_draft, sample_retrieved_info):
        """Test citation management with failure and fallback"""
        
        with patch.object(coherence_manager.kimi_client, 'generate_structured_response', new_callable=AsyncMock) as mock_structured:
            mock_structured.side_effect = Exception("API Error")
            
            # Test citation management with failure
            updated_draft, citations = await coherence_manager.manage_citations(sample_draft, sample_retrieved_info)
            
            # Verify fallback behavior
            assert isinstance(updated_draft, Draft)
            assert len(citations) == 2  # Citations still extracted
            assert updated_draft.id == sample_draft.id
    
    @pytest.mark.asyncio
    async def test_resolve_citation_conflicts_failure_fallback(self, coherence_manager, sample_retrieved_info):
        """Test citation conflict resolution with failure and fallback"""
        
        with patch.object(coherence_manager.kimi_client, 'generate_text', new_callable=AsyncMock) as mock_text:
            mock_text.side_effect = Exception("API Error")
            
            # Test conflict resolution with failure
            resolution = await coherence_manager.resolve_citation_conflicts(sample_retrieved_info)
            
            # Verify fallback behavior
            assert isinstance(resolution, str)
            assert len(resolution) > 0
            assert "differing perspectives" in resolution.lower()
    
    def test_get_coherence_statistics_empty(self, coherence_manager):
        """Test coherence statistics with no history"""
        
        stats = coherence_manager.get_coherence_statistics()
        
        assert stats["total_operations"] == 0
    
    def test_get_coherence_statistics_with_history(self, coherence_manager):
        """Test coherence statistics with history"""
        
        # Add some history
        coherence_manager.coherence_history = [
            {
                "timestamp": datetime.now(),
                "initial_score": 0.6,
                "final_score": 0.8,
                "issues_resolved": 2,
                "improvements_applied": 3
            },
            {
                "timestamp": datetime.now(),
                "initial_score": 0.7,
                "final_score": 0.9,
                "issues_resolved": 1,
                "improvements_applied": 2
            }
        ]
        
        # Add some citations
        coherence_manager.citations = {"cite_1": Mock(), "cite_2": Mock()}
        
        stats = coherence_manager.get_coherence_statistics()
        
        assert stats["total_operations"] == 2
        assert abs(stats["average_initial_score"] - 0.65) < 0.001
        assert abs(stats["average_final_score"] - 0.85) < 0.001
        assert abs(stats["average_improvement"] - 0.2) < 0.001
        assert stats["total_issues_resolved"] == 3
        assert stats["total_citations_managed"] == 2
    
    def test_extract_citations(self, coherence_manager, sample_retrieved_info):
        """Test citation extraction from retrieved information"""
        
        citations = coherence_manager._extract_citations(sample_retrieved_info)
        
        assert len(citations) == 2
        assert all(isinstance(cite, Citation) for cite in citations)
        assert citations[0].id == "cite_1"
        assert citations[1].id == "cite_2"
        assert len(coherence_manager.citations) == 2
    
    def test_copy_draft(self, coherence_manager, sample_draft):
        """Test draft copying functionality"""
        
        copied_draft = coherence_manager._copy_draft(sample_draft)
        
        assert copied_draft.id == sample_draft.id
        assert copied_draft.topic == sample_draft.topic
        assert copied_draft.content == sample_draft.content
        assert copied_draft is not sample_draft  # Different objects
        assert copied_draft.content is not sample_draft.content  # Different content dict