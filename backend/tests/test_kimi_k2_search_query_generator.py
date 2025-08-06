"""
Unit tests for Kimi K2 Search Query Generation Service.
Tests the query generation, optimization, and validation functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.core import (
    InformationGap, GapType, Priority, SearchQuery, ResearchDomain
)
from services.kimi_k2_search_query_generator import (
    KimiK2SearchQueryGenerator, SearchQueryTemplate
)
from services.kimi_k2_client import KimiK2Client, KimiK2Response, KimiK2Error

class TestSearchQueryTemplate:
    """Test cases for SearchQueryTemplate"""
    
    def test_template_creation(self):
        """Test creating a search query template"""
        template = SearchQueryTemplate("test_template", ResearchDomain.TECHNOLOGY, GapType.CONTENT)
        
        assert template.template_id == "test_template"
        assert template.domain == ResearchDomain.TECHNOLOGY
        assert template.gap_type == GapType.CONTENT
        assert template.query_patterns == []
        assert template.optimization_hints == []
        assert template.validation_criteria == []
    
    def test_add_pattern(self):
        """Test adding query patterns to template"""
        template = SearchQueryTemplate("test", ResearchDomain.TECHNOLOGY, GapType.CONTENT)
        
        template.add_pattern("{topic} technical specifications", Priority.HIGH)
        template.add_pattern("{topic} implementation guide", Priority.MEDIUM)
        
        assert len(template.query_patterns) == 2
        assert template.query_patterns[0]["pattern"] == "{topic} technical specifications"
        assert template.query_patterns[0]["priority"] == Priority.HIGH
        assert "topic" in template.query_patterns[0]["variables"]
    
    def test_generate_queries_from_template(self):
        """Test generating queries from template patterns"""
        template = SearchQueryTemplate("test", ResearchDomain.TECHNOLOGY, GapType.CONTENT)
        template.add_pattern("{topic} {domain} overview")
        template.add_pattern("{topic} best practices")
        
        variables = {"topic": "AI", "domain": "technology"}
        queries = template.generate_queries(variables, max_queries=2)
        
        assert len(queries) == 2
        assert "AI technology overview" in queries
        assert "AI best practices" in queries
    
    def test_generate_queries_missing_variables(self):
        """Test handling missing variables in template generation"""
        template = SearchQueryTemplate("test", ResearchDomain.TECHNOLOGY, GapType.CONTENT)
        template.add_pattern("{topic} {missing_var} overview")
        
        variables = {"topic": "AI"}
        queries = template.generate_queries(variables, max_queries=1)
        
        # Should handle missing variables gracefully
        assert len(queries) == 0  # Pattern with missing variable should be skipped

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
    
    def test_template_initialization(self, query_generator):
        """Test that templates are properly initialized"""
        templates = query_generator.query_templates
        
        # Should have templates for different domains and gap types
        assert "tech_content" in templates
        assert "tech_evidence" in templates
        assert "science_content" in templates
        assert "business_content" in templates
        assert "citation" in templates
        
        # Check template structure
        tech_template = templates["tech_content"]
        assert tech_template.domain == ResearchDomain.TECHNOLOGY
        assert tech_template.gap_type == GapType.CONTENT
        assert len(tech_template.query_patterns) > 0
        assert len(tech_template.optimization_hints) > 0
    
    @pytest.mark.asyncio
    async def test_generate_search_queries_success(self, query_generator, sample_gap, mock_kimi_client):
        """Test successful search query generation with Kimi K2"""
        # Mock Kimi K2 response
        mock_response = {
            "queries": [
                {
                    "query": "AI healthcare implementation case studies",
                    "priority": "high",
                    "expected_results": 15,
                    "search_strategy": "Find real-world implementation examples",
                    "reasoning": "Targets specific implementation examples"
                },
                {
                    "query": "artificial intelligence medical diagnosis examples",
                    "priority": "medium",
                    "expected_results": 10,
                    "search_strategy": "Focus on diagnostic applications",
                    "reasoning": "Covers diagnostic use cases"
                }
            ]
        }
        
        # Mock validation response
        validation_response = {
            "validated_queries": [
                {
                    "original_query": "AI healthcare implementation case studies",
                    "refined_query": "AI healthcare implementation case studies",
                    "effectiveness_score": 0.9,
                    "validation_notes": "Well-targeted query",
                    "keep_query": True
                },
                {
                    "original_query": "artificial intelligence medical diagnosis examples",
                    "refined_query": "AI medical diagnosis examples",
                    "effectiveness_score": 0.8,
                    "validation_notes": "Simplified for better results",
                    "keep_query": True
                }
            ]
        }
        
        mock_kimi_client.generate_structured_response.side_effect = [
            mock_response,  # Query generation
            validation_response  # Query validation
        ]
        
        # Execute query generation
        queries = await query_generator.generate_search_queries(
            gap=sample_gap,
            topic="AI in Healthcare",
            domain=ResearchDomain.TECHNOLOGY,
            max_queries=3
        )
        
        # Verify results
        assert len(queries) >= 2
        assert all(isinstance(query, SearchQuery) for query in queries)
        
        # Check that queries were refined during validation
        refined_query = next((q for q in queries if "medical diagnosis" in q.query), None)
        assert refined_query is not None
        # Query should contain the refined content (may have additional optimization)
        assert "AI medical diagnosis" in refined_query.query or "artificial intelligence medical diagnosis" in refined_query.query
        assert refined_query.effectiveness_score == 0.8
        
        # Verify Kimi K2 client was called for both generation and validation
        assert mock_kimi_client.generate_structured_response.call_count == 2
    
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
        
        # Verify fallback queries contain relevant terms
        query_texts = [q.query.lower() for q in queries]
        assert any("ai" in text or "healthcare" in text for text in query_texts)
        assert any("evidence" in text for text in query_texts)  # Based on gap type
    
    @pytest.mark.asyncio
    async def test_generate_with_kimi_k2(self, query_generator, sample_gap, mock_kimi_client):
        """Test direct Kimi K2 query generation"""
        mock_response = {
            "queries": [
                {
                    "query": "AI implementation case studies healthcare",
                    "priority": "high",
                    "expected_results": 12,
                    "search_strategy": "Implementation focus",
                    "reasoning": "Targets implementation examples"
                }
            ]
        }
        
        mock_kimi_client.generate_structured_response.return_value = mock_response
        
        # Execute Kimi K2 generation
        queries = await query_generator._generate_with_kimi_k2(
            sample_gap, "AI in Healthcare", ResearchDomain.TECHNOLOGY, 3
        )
        
        # Verify results
        assert len(queries) == 1
        assert queries[0].query == "AI implementation case studies healthcare"
        assert queries[0].priority == Priority.HIGH
        assert queries[0].expected_results == 12
        assert queries[0].search_strategy == "Implementation focus"
        assert hasattr(queries[0], 'reasoning')
    
    @pytest.mark.asyncio
    async def test_optimize_queries_with_templates(self, query_generator, sample_gap):
        """Test query optimization using templates"""
        # Create initial queries
        initial_queries = [
            SearchQuery(query="AI healthcare", priority=Priority.MEDIUM),
            SearchQuery(query="machine learning medical", priority=Priority.HIGH)
        ]
        
        # Execute optimization
        optimized_queries = await query_generator._optimize_queries_with_templates(
            initial_queries, sample_gap, "AI in Healthcare", ResearchDomain.TECHNOLOGY
        )
        
        # Should have more queries (original + template-generated)
        assert len(optimized_queries) >= len(initial_queries)
        
        # Check that domain-specific optimizations were applied
        optimized_texts = [q.query.lower() for q in optimized_queries]
        
        # Should include technology-related terms (check for domain optimization)
        # The optimization should add domain-specific terms to existing queries
        assert any("technical" in text or "implementation" in text or "system" in text for text in optimized_texts)
    
    def test_apply_domain_optimization(self, query_generator):
        """Test domain-specific query optimization"""
        query = SearchQuery(query="AI healthcare", priority=Priority.MEDIUM)
        
        # Apply technology domain optimization
        optimized_query = query_generator._apply_domain_optimization(
            query, ResearchDomain.TECHNOLOGY, GapType.EVIDENCE
        )
        
        # Should add domain and gap-type specific terms
        assert "technical" in optimized_query.query.lower() or "implementation" in optimized_query.query.lower()
        assert "examples" in optimized_query.query.lower() or "case studies" in optimized_query.query.lower()
    
    @pytest.mark.asyncio
    async def test_validate_and_refine_queries(self, query_generator, sample_gap, mock_kimi_client):
        """Test query validation and refinement"""
        # Create test queries
        queries = [
            SearchQuery(query="AI healthcare implementation", priority=Priority.HIGH),
            SearchQuery(query="machine learning medical applications", priority=Priority.MEDIUM)
        ]
        
        # Mock validation response
        mock_response = {
            "validated_queries": [
                {
                    "original_query": "AI healthcare implementation",
                    "refined_query": "AI healthcare implementation case studies",
                    "effectiveness_score": 0.9,
                    "validation_notes": "Added 'case studies' for better targeting",
                    "keep_query": True
                },
                {
                    "original_query": "machine learning medical applications",
                    "refined_query": "machine learning medical applications",
                    "effectiveness_score": 0.7,
                    "validation_notes": "Query is already well-formed",
                    "keep_query": True
                }
            ]
        }
        
        mock_kimi_client.generate_structured_response.return_value = mock_response
        
        # Execute validation
        validated_queries = await query_generator._validate_and_refine_queries(
            queries, sample_gap, "AI in Healthcare", ResearchDomain.TECHNOLOGY
        )
        
        # Verify refinement
        assert len(validated_queries) == 2
        assert validated_queries[0].query == "AI healthcare implementation case studies"
        assert validated_queries[0].effectiveness_score == 0.9
        assert hasattr(validated_queries[0], 'validation_notes')
    
    def test_generate_fallback_queries(self, query_generator, sample_gap):
        """Test fallback query generation"""
        queries = query_generator._generate_fallback_queries(
            sample_gap, "AI in Healthcare", ResearchDomain.TECHNOLOGY, 3
        )
        
        # Should generate queries
        assert len(queries) > 0
        assert len(queries) <= 3
        assert all(isinstance(query, SearchQuery) for query in queries)
        
        # Verify queries contain relevant terms
        query_texts = [q.query.lower() for q in queries]
        assert any("ai" in text or "healthcare" in text for text in query_texts)
        
        # Should include evidence-specific query (based on gap type)
        assert any("evidence" in text or "case study" in text for text in query_texts)
    
    @pytest.mark.asyncio
    async def test_batch_generate_queries(self, query_generator, mock_kimi_client):
        """Test batch query generation for multiple gaps"""
        # Create multiple gaps
        gaps = [
            InformationGap(
                id="gap_1",
                section_id="intro",
                gap_type=GapType.CONTENT,
                description="Missing introduction content",
                priority=Priority.HIGH
            ),
            InformationGap(
                id="gap_2",
                section_id="analysis",
                gap_type=GapType.EVIDENCE,
                description="Missing evidence",
                priority=Priority.MEDIUM
            )
        ]
        
        # Mock responses for each gap
        mock_kimi_client.generate_structured_response.side_effect = [
            {"queries": [{"query": "AI healthcare overview", "priority": "high"}]},  # Gap 1 generation
            {"validated_queries": [{"original_query": "AI healthcare overview", "refined_query": "AI healthcare overview", "effectiveness_score": 0.8, "keep_query": True}]},  # Gap 1 validation
            {"queries": [{"query": "AI healthcare evidence", "priority": "medium"}]},  # Gap 2 generation
            {"validated_queries": [{"original_query": "AI healthcare evidence", "refined_query": "AI healthcare evidence", "effectiveness_score": 0.7, "keep_query": True}]}  # Gap 2 validation
        ]
        
        # Execute batch generation
        results = await query_generator.batch_generate_queries(
            gaps, "AI in Healthcare", ResearchDomain.TECHNOLOGY, max_queries_per_gap=2
        )
        
        # Verify results
        assert len(results) == 2
        assert "gap_1" in results
        assert "gap_2" in results
        assert len(results["gap_1"]) > 0
        assert len(results["gap_2"]) > 0
    
    def test_get_query_statistics(self, query_generator):
        """Test query statistics generation"""
        queries = [
            SearchQuery(query="AI healthcare", priority=Priority.HIGH, effectiveness_score=0.9, search_strategy="kimi_generated"),
            SearchQuery(query="machine learning medical", priority=Priority.MEDIUM, effectiveness_score=0.7, search_strategy="template_generated"),
            SearchQuery(query="artificial intelligence diagnosis", priority=Priority.HIGH, effectiveness_score=0.8, search_strategy="kimi_generated")
        ]
        
        stats = query_generator.get_query_statistics(queries)
        
        # Verify statistics
        assert stats["total"] == 3
        assert stats["priority_distribution"]["high"] == 2
        assert stats["priority_distribution"]["medium"] == 1
        assert abs(stats["average_effectiveness"] - 0.8) < 0.001  # (0.9 + 0.7 + 0.8) / 3
        assert len(stats["query_lengths"]) == 3
        assert "kimi_generated" in stats["strategies"]
        assert "template_generated" in stats["strategies"]
    
    def test_get_query_statistics_empty(self, query_generator):
        """Test query statistics with empty query list"""
        stats = query_generator.get_query_statistics([])
        
        assert stats["total"] == 0

class TestQueryGenerationIntegration:
    """Integration tests for query generation system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_query_generation(self):
        """Test complete query generation workflow"""
        # Create generator with real client (will use fallback if no API key)
        generator = KimiK2SearchQueryGenerator()
        
        # Create test gap
        gap = InformationGap(
            id="integration_test_gap",
            section_id="methodology",
            gap_type=GapType.CONTENT,
            description="Missing research methodology information",
            priority=Priority.HIGH
        )
        
        # Execute query generation
        queries = await generator.generate_search_queries(
            gap=gap,
            topic="Machine Learning in Medical Diagnosis",
            domain=ResearchDomain.SCIENCE,
            max_queries=3
        )
        
        # Verify results
        assert len(queries) > 0
        assert len(queries) <= 3
        assert all(isinstance(query, SearchQuery) for query in queries)
        
        # Queries should be relevant to the topic and domain
        query_texts = [q.query.lower() for q in queries]
        assert any("machine learning" in text or "medical" in text or "diagnosis" in text for text in query_texts)
        
        # Should include methodology-related terms (based on gap description)
        assert any("methodology" in text or "research" in text for text in query_texts)
    
    @pytest.mark.asyncio
    async def test_domain_specific_generation(self):
        """Test that different domains generate appropriate queries"""
        generator = KimiK2SearchQueryGenerator()
        
        # Test technology domain
        tech_gap = InformationGap(
            id="tech_gap",
            section_id="implementation",
            gap_type=GapType.EVIDENCE,
            description="Missing implementation examples",
            priority=Priority.HIGH
        )
        
        tech_queries = await generator.generate_search_queries(
            gap=tech_gap,
            topic="Blockchain Technology",
            domain=ResearchDomain.TECHNOLOGY,
            max_queries=2
        )
        
        # Test business domain
        business_gap = InformationGap(
            id="business_gap",
            section_id="market_analysis",
            gap_type=GapType.CONTENT,
            description="Missing market analysis",
            priority=Priority.HIGH
        )
        
        business_queries = await generator.generate_search_queries(
            gap=business_gap,
            topic="Blockchain Technology",
            domain=ResearchDomain.BUSINESS,
            max_queries=2
        )
        
        # Verify domain-specific differences
        tech_texts = [q.query.lower() for q in tech_queries]
        business_texts = [q.query.lower() for q in business_queries]
        
        # Technology queries should include technical terms
        assert any("technical" in text or "implementation" in text for text in tech_texts)
        
        # Business queries should include business terms
        assert any("business" in text or "market" in text for text in business_texts)

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])