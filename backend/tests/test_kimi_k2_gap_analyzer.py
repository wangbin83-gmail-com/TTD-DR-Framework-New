"""
Unit tests for Kimi K2 Information Gap Analysis Service.
Tests the gap identification, prioritization, and search query generation functionality.
"""

import pytest
import asyncio
import uuid
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.core import (
    Draft, InformationGap, GapType, Priority, SearchQuery, 
    ResearchDomain, ComplexityLevel, ResearchRequirements,
    DraftMetadata, QualityMetrics
)
from models.research_structure import (
    EnhancedResearchStructure, EnhancedSection, ContentPlaceholder,
    ResearchSectionType, ContentPlaceholderType
)
from services.kimi_k2_gap_analyzer import (
    KimiK2InformationGapAnalyzer, KimiK2SearchQueryGenerator
)
from services.kimi_k2_client import KimiK2Client, KimiK2Response, KimiK2Error

class TestKimiK2InformationGapAnalyzer:
    """Test cases for KimiK2InformationGapAnalyzer"""
    
    @pytest.fixture
    def mock_kimi_client(self):
        """Create a mock Kimi K2 client"""
        client = Mock(spec=KimiK2Client)
        client.generate_structured_response = AsyncMock()
        return client
    
    @pytest.fixture
    def gap_analyzer(self, mock_kimi_client):
        """Create gap analyzer with mock client"""
        return KimiK2InformationGapAnalyzer(mock_kimi_client)
    
    @pytest.fixture
    def sample_draft(self):
        """Create a sample draft for testing"""
        # Create enhanced sections
        section1 = EnhancedSection(
            id="intro",
            title="Introduction",
            estimated_length=500,
            section_type=ResearchSectionType.INTRODUCTION,
            content_placeholders=[
                ContentPlaceholder(
                    id="intro_placeholder",
                    placeholder_type=ContentPlaceholderType.INTRODUCTION,
                    title="Topic Overview",
                    description="Overview of the research topic",
                    estimated_word_count=200,
                    priority=Priority.HIGH
                )
            ]
        )
        
        section2 = EnhancedSection(
            id="analysis",
            title="Analysis",
            estimated_length=1000,
            section_type=ResearchSectionType.ANALYSIS,
            content_placeholders=[
                ContentPlaceholder(
                    id="analysis_placeholder",
                    placeholder_type=ContentPlaceholderType.ANALYSIS,
                    title="Main Analysis",
                    description="Core analysis of the topic",
                    estimated_word_count=500,
                    priority=Priority.CRITICAL
                )
            ]
        )
        
        # Create research structure
        structure = EnhancedResearchStructure(
            sections=[section1, section2],
            estimated_length=1500,
            complexity_level=ComplexityLevel.INTERMEDIATE,
            domain=ResearchDomain.TECHNOLOGY
        )
        
        # Create draft
        draft = Draft(
            id="test_draft",
            topic="Artificial Intelligence in Healthcare",
            structure=structure,
            content={
                "intro": "Brief introduction to AI in healthcare",
                "analysis": "Limited analysis content"
            },
            metadata=DraftMetadata(),
            quality_score=0.6,
            iteration=1
        )
        
        return draft
    
    @pytest.mark.asyncio
    async def test_identify_gaps_success(self, gap_analyzer, sample_draft, mock_kimi_client):
        """Test successful gap identification"""
        # Mock Kimi K2 responses
        section_gap_response = {
            "gaps": [
                {
                    "gap_type": "content",
                    "description": "Missing detailed technical specifications",
                    "priority": "high",
                    "specific_needs": ["Technical details", "Implementation examples"],
                    "suggested_sources": ["Technical documentation", "Research papers"]
                },
                {
                    "gap_type": "evidence",
                    "description": "Lacks supporting case studies",
                    "priority": "medium",
                    "specific_needs": ["Case studies", "Real-world examples"],
                    "suggested_sources": ["Industry reports", "Case study databases"]
                }
            ]
        }
        
        overall_gap_response = {
            "structural_gaps": [
                {
                    "description": "Missing methodology section",
                    "priority": "high",
                    "affected_sections": ["analysis"]
                }
            ],
            "coherence_gaps": [
                {
                    "description": "Weak connection between introduction and analysis",
                    "priority": "medium",
                    "section_connections": ["intro", "analysis"]
                }
            ]
        }
        
        prioritization_response = {
            "prioritized_gaps": [
                {
                    "gap_id": "gap_1",
                    "priority": "critical",
                    "reasoning": "Essential for technical completeness",
                    "impact_score": 0.9
                },
                {
                    "gap_id": "gap_2", 
                    "priority": "high",
                    "reasoning": "Important for credibility",
                    "impact_score": 0.7
                }
            ]
        }
        
        # Configure mock responses
        mock_kimi_client.generate_structured_response.side_effect = [
            section_gap_response,  # First section
            section_gap_response,  # Second section
            overall_gap_response,  # Overall analysis
            prioritization_response  # Prioritization
        ]
        
        # Execute gap identification
        gaps = await gap_analyzer.identify_gaps(sample_draft)
        
        # Verify results
        assert len(gaps) > 0
        assert all(isinstance(gap, InformationGap) for gap in gaps)
        assert any(gap.gap_type == GapType.CONTENT for gap in gaps)
        assert any(gap.priority == Priority.HIGH for gap in gaps)
        
        # Verify Kimi K2 client was called
        assert mock_kimi_client.generate_structured_response.call_count >= 3
    
    @pytest.mark.asyncio
    async def test_identify_gaps_kimi_error(self, gap_analyzer, sample_draft, mock_kimi_client):
        """Test gap identification with Kimi K2 error (fallback behavior)"""
        # Mock Kimi K2 error
        mock_kimi_client.generate_structured_response.side_effect = KimiK2Error("API Error")
        
        # Execute gap identification
        gaps = await gap_analyzer.identify_gaps(sample_draft)
        
        # Should return fallback gaps
        assert len(gaps) > 0
        assert all(isinstance(gap, InformationGap) for gap in gaps)
        
        # Verify fallback behavior
        section_gaps = [gap for gap in gaps if gap.section_id in ["intro", "analysis"]]
        assert len(section_gaps) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_section_gaps(self, gap_analyzer, sample_draft, mock_kimi_client):
        """Test section-specific gap analysis"""
        section = sample_draft.structure.sections[0]  # Introduction section
        content = sample_draft.content["intro"]
        
        # Mock response
        mock_response = {
            "gaps": [
                {
                    "gap_type": "content",
                    "description": "Missing background information",
                    "priority": "high",
                    "specific_needs": ["Historical context", "Current state"],
                    "suggested_sources": ["Academic papers", "Industry reports"]
                }
            ]
        }
        
        mock_kimi_client.generate_structured_response.return_value = mock_response
        
        # Execute section analysis
        gaps = await gap_analyzer._analyze_section_gaps(
            section, content, sample_draft.topic, sample_draft.structure.domain
        )
        
        # Verify results
        assert len(gaps) == 1
        assert gaps[0].section_id == section.id
        assert gaps[0].gap_type == GapType.CONTENT
        assert gaps[0].priority == Priority.HIGH
        assert hasattr(gaps[0], 'specific_needs')
        assert hasattr(gaps[0], 'suggested_sources')
    
    @pytest.mark.asyncio
    async def test_analyze_overall_gaps(self, gap_analyzer, sample_draft, mock_kimi_client):
        """Test overall draft gap analysis"""
        # Mock response
        mock_response = {
            "structural_gaps": [
                {
                    "description": "Missing conclusion section",
                    "priority": "high",
                    "affected_sections": ["analysis"]
                }
            ],
            "coherence_gaps": [
                {
                    "description": "Inconsistent terminology usage",
                    "priority": "medium",
                    "section_connections": ["intro", "analysis"]
                }
            ]
        }
        
        mock_kimi_client.generate_structured_response.return_value = mock_response
        
        # Execute overall analysis
        gaps = await gap_analyzer._analyze_overall_gaps(sample_draft)
        
        # Verify results
        assert len(gaps) == 2
        structural_gaps = [gap for gap in gaps if gap.section_id == "overall_structure"]
        coherence_gaps = [gap for gap in gaps if gap.section_id == "overall_coherence"]
        
        assert len(structural_gaps) == 1
        assert len(coherence_gaps) == 1
        assert structural_gaps[0].gap_type == GapType.CONTENT
        assert coherence_gaps[0].gap_type == GapType.ANALYSIS
    
    @pytest.mark.asyncio
    async def test_prioritize_gaps(self, gap_analyzer, sample_draft, mock_kimi_client):
        """Test gap prioritization"""
        # Create test gaps
        gaps = [
            InformationGap(
                id="gap_1",
                section_id="intro",
                gap_type=GapType.CONTENT,
                description="Missing content",
                priority=Priority.MEDIUM
            ),
            InformationGap(
                id="gap_2",
                section_id="analysis",
                gap_type=GapType.EVIDENCE,
                description="Missing evidence",
                priority=Priority.LOW
            )
        ]
        
        # Mock prioritization response
        mock_response = {
            "prioritized_gaps": [
                {
                    "gap_id": "gap_1",
                    "priority": "critical",
                    "reasoning": "Essential for completeness",
                    "impact_score": 0.9
                },
                {
                    "gap_id": "gap_2",
                    "priority": "high",
                    "reasoning": "Important for credibility",
                    "impact_score": 0.7
                }
            ]
        }
        
        mock_kimi_client.generate_structured_response.return_value = mock_response
        
        # Execute prioritization
        prioritized_gaps = await gap_analyzer._prioritize_gaps(gaps, sample_draft)
        
        # Verify results
        assert len(prioritized_gaps) == 2
        assert prioritized_gaps[0].priority == Priority.CRITICAL
        assert prioritized_gaps[1].priority == Priority.HIGH
        assert hasattr(prioritized_gaps[0], 'impact_score')
        assert prioritized_gaps[0].impact_score == 0.9
    
    def test_generate_fallback_gaps(self, gap_analyzer, sample_draft):
        """Test fallback gap generation"""
        gaps = gap_analyzer._generate_fallback_gaps(sample_draft)
        
        # Should generate gaps for underdeveloped sections
        assert len(gaps) > 0
        assert all(isinstance(gap, InformationGap) for gap in gaps)
        
        # Check for content gaps in short sections
        content_gaps = [gap for gap in gaps if gap.gap_type == GapType.CONTENT]
        assert len(content_gaps) > 0
        
        # Check for citation gaps
        citation_gaps = [gap for gap in gaps if gap.gap_type == GapType.CITATION]
        assert len(citation_gaps) > 0

class TestKimiK2SearchQueryGenerator:
    """Test cases for KimiK2SearchQueryGenerator"""
    
    @pytest.fixture
    def mock_kimi_client(self):
        """Create a mock Kimi K2 client"""
        client = Mock(spec=KimiK2Client)
        client.generate_structured_response = AsyncMock()
        return client
    
    @pytest.fixture
    def query_generator(self, mock_kimi_client):
        """Create query generator with mock client"""
        return KimiK2SearchQueryGenerator(mock_kimi_client)
    
    @pytest.fixture
    def sample_gap(self):
        """Create a sample information gap"""
        gap = InformationGap(
            id="test_gap",
            section_id="analysis",
            gap_type=GapType.EVIDENCE,
            description="Missing case studies for AI implementation",
            priority=Priority.HIGH
        )
        gap.specific_needs = ["Case studies", "Implementation examples"]
        gap.suggested_sources = ["Industry reports", "Academic papers"]
        return gap
    
    @pytest.mark.asyncio
    async def test_generate_search_queries_success(self, query_generator, sample_gap, mock_kimi_client):
        """Test successful search query generation"""
        # Mock response
        mock_response = {
            "queries": [
                {
                    "query": "AI healthcare implementation case studies",
                    "priority": "high",
                    "expected_results": 15,
                    "search_strategy": "Find real-world implementation examples"
                },
                {
                    "query": "artificial intelligence medical diagnosis examples",
                    "priority": "medium",
                    "expected_results": 10,
                    "search_strategy": "Focus on diagnostic applications"
                },
                {
                    "query": "healthcare AI success stories industry reports",
                    "priority": "medium",
                    "expected_results": 12,
                    "search_strategy": "Industry perspective on AI adoption"
                }
            ]
        }
        
        mock_kimi_client.generate_structured_response.return_value = mock_response
        
        # Execute query generation
        queries = await query_generator.generate_search_queries(
            gap=sample_gap,
            topic="AI in Healthcare",
            domain=ResearchDomain.TECHNOLOGY,
            max_queries=3
        )
        
        # Verify results
        assert len(queries) == 3
        assert all(isinstance(query, SearchQuery) for query in queries)
        assert queries[0].priority == Priority.HIGH
        assert queries[0].expected_results == 15
        assert hasattr(queries[0], 'search_strategy')
        
        # Verify Kimi K2 client was called
        mock_kimi_client.generate_structured_response.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_search_queries_kimi_error(self, query_generator, sample_gap, mock_kimi_client):
        """Test query generation with Kimi K2 error (fallback behavior)"""
        # Mock Kimi K2 error
        mock_kimi_client.generate_structured_response.side_effect = KimiK2Error("API Error")
        
        # Execute query generation
        queries = await query_generator.generate_search_queries(
            gap=sample_gap,
            topic="AI in Healthcare",
            domain=ResearchDomain.TECHNOLOGY
        )
        
        # Should return fallback queries
        assert len(queries) > 0
        assert all(isinstance(query, SearchQuery) for query in queries)
        
        # Verify fallback queries contain topic and gap information
        query_texts = [q.query.lower() for q in queries]
        assert any("ai" in text or "healthcare" in text for text in query_texts)
    
    @pytest.mark.asyncio
    async def test_optimize_queries(self, query_generator):
        """Test query optimization"""
        # Create test queries
        queries = [
            SearchQuery(query="machine learning", priority=Priority.MEDIUM),
            SearchQuery(query="AI applications", priority=Priority.HIGH)
        ]
        
        # Execute optimization
        optimized = await query_generator._optimize_queries(queries, ResearchDomain.TECHNOLOGY)
        
        # Verify optimization
        assert len(optimized) == len(queries)
        
        # Check that domain context was added where appropriate
        optimized_texts = [q.query.lower() for q in optimized]
        assert any("technology" in text for text in optimized_texts)
    
    def test_generate_fallback_queries(self, query_generator, sample_gap):
        """Test fallback query generation"""
        queries = query_generator._generate_fallback_queries(sample_gap, "AI in Healthcare")
        
        # Should generate basic queries
        assert len(queries) > 0
        assert all(isinstance(query, SearchQuery) for query in queries)
        
        # Verify queries contain relevant terms
        query_texts = [q.query.lower() for q in queries]
        assert any("ai" in text or "healthcare" in text for text in query_texts)
        assert any("evidence" in text for text in query_texts)  # Based on gap type
    
    @pytest.mark.asyncio
    async def test_validate_and_refine_queries(self, query_generator, sample_gap, mock_kimi_client):
        """Test query validation and refinement"""
        # Create test queries
        queries = [
            SearchQuery(query="AI healthcare", priority=Priority.MEDIUM),
            SearchQuery(query="machine learning medical", priority=Priority.HIGH)
        ]
        
        # Mock refinement response
        mock_response = {
            "refined_queries": [
                {
                    "original_query": "AI healthcare",
                    "refined_query": "artificial intelligence healthcare applications case studies",
                    "improvement_reason": "More specific and targeted",
                    "effectiveness_score": 0.8
                },
                {
                    "original_query": "machine learning medical",
                    "refined_query": "machine learning medical diagnosis implementation",
                    "improvement_reason": "Added specific application area",
                    "effectiveness_score": 0.7
                }
            ]
        }
        
        mock_kimi_client.generate_structured_response.return_value = mock_response
        
        # Execute refinement
        refined_queries = await query_generator.validate_and_refine_queries(
            queries, sample_gap, "AI in Healthcare"
        )
        
        # Verify refinement
        assert len(refined_queries) == 2
        assert refined_queries[0].query == "artificial intelligence healthcare applications case studies"
        assert refined_queries[1].query == "machine learning medical diagnosis implementation"
        assert hasattr(refined_queries[0], 'effectiveness_score')
        assert refined_queries[0].effectiveness_score == 0.8

class TestGapAnalyzerIntegration:
    """Integration tests for gap analyzer components"""
    
    @pytest.fixture
    def sample_draft_with_gaps(self):
        """Create a draft that should have identifiable gaps"""
        # Create sections with minimal content
        section1 = EnhancedSection(
            id="intro",
            title="Introduction",
            estimated_length=1000,
            section_type=ResearchSectionType.INTRODUCTION
        )
        
        section2 = EnhancedSection(
            id="methods",
            title="Methodology", 
            estimated_length=800,
            section_type=ResearchSectionType.METHODOLOGY
        )
        
        structure = EnhancedResearchStructure(
            sections=[section1, section2],
            estimated_length=1800,
            complexity_level=ComplexityLevel.ADVANCED,
            domain=ResearchDomain.SCIENCE
        )
        
        # Create draft with insufficient content
        draft = Draft(
            id="gap_test_draft",
            topic="Climate Change Impact on Marine Ecosystems",
            structure=structure,
            content={
                "intro": "Climate change affects marine life.",  # Very brief
                "methods": ""  # Empty
            },
            quality_score=0.3,
            iteration=0
        )
        
        return draft
    
    @pytest.mark.asyncio
    async def test_end_to_end_gap_analysis(self, sample_draft_with_gaps):
        """Test complete gap analysis workflow"""
        # Create analyzer with real client (will use fallback if no API key)
        analyzer = KimiK2InformationGapAnalyzer()
        query_generator = KimiK2SearchQueryGenerator()
        
        # Execute gap identification
        gaps = await analyzer.identify_gaps(sample_draft_with_gaps)
        
        # Should identify gaps in underdeveloped sections
        assert len(gaps) > 0
        
        # Should have content gaps for short/empty sections
        content_gaps = [gap for gap in gaps if gap.gap_type == GapType.CONTENT]
        assert len(content_gaps) > 0
        
        # Generate queries for first gap
        if gaps:
            queries = await query_generator.generate_search_queries(
                gap=gaps[0],
                topic=sample_draft_with_gaps.topic,
                domain=sample_draft_with_gaps.structure.domain
            )
            
            assert len(queries) > 0
            assert all(isinstance(query, SearchQuery) for query in queries)
            
            # Queries should be relevant to the topic
            query_texts = [q.query.lower() for q in queries]
            assert any("climate" in text or "marine" in text for text in query_texts)

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])