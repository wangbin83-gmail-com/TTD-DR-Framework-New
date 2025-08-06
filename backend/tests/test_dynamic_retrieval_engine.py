"""
Unit tests for Dynamic Retrieval Engine.
Tests content processing, credibility scoring, and integration functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.dynamic_retrieval_engine import (
    DynamicRetrievalEngine, SourceCredibilityScorer, ContentProcessor,
    DuplicateDetector, CredibilityFactors
)
from services.google_search_client import GoogleSearchResult, GoogleSearchResponse
from models.core import (
    InformationGap, GapType, Priority, SearchQuery, RetrievedInfo, Source
)

class TestSourceCredibilityScorer:
    """Test source credibility scoring functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.scorer = SourceCredibilityScorer()
    
    def test_score_domain_authority_academic(self):
        """Test domain authority scoring for academic sources"""
        assert self.scorer.score_domain_authority("https://scholar.google.com/paper") == 0.95
        assert self.scorer.score_domain_authority("https://university.edu/research") == 0.9
        assert self.scorer.score_domain_authority("https://arxiv.org/abs/123") == 0.9
    
    def test_score_domain_authority_government(self):
        """Test domain authority scoring for government sources"""
        assert self.scorer.score_domain_authority("https://cdc.gov/health") == 0.9
        assert self.scorer.score_domain_authority("https://agency.gov/report") == 0.9
        assert self.scorer.score_domain_authority("https://military.mil/info") == 0.85
    
    def test_score_domain_authority_news(self):
        """Test domain authority scoring for news sources"""
        assert self.scorer.score_domain_authority("https://reuters.com/article") == 0.85
        assert self.scorer.score_domain_authority("https://bbc.com/news") == 0.85
        assert self.scorer.score_domain_authority("https://nytimes.com/article") == 0.8
    
    def test_score_domain_authority_low_credibility(self):
        """Test domain authority scoring for low credibility sources"""
        assert self.scorer.score_domain_authority("https://myblog.wordpress.com") == 0.4
        assert self.scorer.score_domain_authority("https://random.blogspot.com") == 0.4
        assert self.scorer.score_domain_authority("https://unknown.com") == 0.6
    
    def test_score_search_ranking(self):
        """Test search ranking scoring"""
        assert self.scorer.score_search_ranking(1, 100) == 0.9
        assert self.scorer.score_search_ranking(5, 100) == 0.8
        assert self.scorer.score_search_ranking(15, 100) == 0.7
        assert self.scorer.score_search_ranking(50, 100) == 0.5
    
    def test_score_content_quality(self):
        """Test content quality scoring"""
        # High quality result
        high_quality_result = GoogleSearchResult(
            title="Research Study on Machine Learning",
            link="https://example.com",
            snippet="This comprehensive research study analyzes machine learning algorithms with statistical data and methodology. The findings show significant evidence of improved performance."
        )
        
        score = self.scorer.score_content_quality(high_quality_result)
        assert score > 0.7
        
        # Low quality result
        low_quality_result = GoogleSearchResult(
            title="Blog Post",
            link="https://example.com",
            snippet="Short snippet"
        )
        
        score = self.scorer.score_content_quality(low_quality_result)
        assert score < 0.7
    
    def test_determine_source_type(self):
        """Test source type determination"""
        assert self.scorer.determine_source_type("https://university.edu", Mock()) == "academic"
        assert self.scorer.determine_source_type("https://agency.gov", Mock()) == "government"
        assert self.scorer.determine_source_type("https://reuters.com", Mock()) == "news"
        assert self.scorer.determine_source_type("https://wikipedia.org", Mock()) == "reference"
        assert self.scorer.determine_source_type("https://stackoverflow.com", Mock()) == "technical"
        assert self.scorer.determine_source_type("https://myblog.wordpress.com", Mock()) == "blog"
        assert self.scorer.determine_source_type("https://example.com", Mock()) == "general"
    
    def test_calculate_credibility_score(self):
        """Test overall credibility score calculation"""
        result = GoogleSearchResult(
            title="Research Study",
            link="https://university.edu/study",
            snippet="Comprehensive research with data and analysis methodology"
        )
        
        factors = self.scorer.calculate_credibility_score(result, 1, 100)
        
        assert isinstance(factors, CredibilityFactors)
        assert factors.domain_authority > 0.8  # Academic domain
        assert factors.search_ranking == 0.9  # First position
        assert factors.content_quality > 0.6  # Good content indicators
        assert factors.source_type == "academic"

class TestContentProcessor:
    """Test content processing functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.processor = ContentProcessor()
    
    def test_extract_relevant_content(self):
        """Test content extraction from search results"""
        result = GoogleSearchResult(
            title="Test Article",
            link="https://example.com",
            snippet="This is the main snippet content",
            html_snippet="This is <b>HTML</b> snippet content",
            page_map={"article": ["Additional structured data"]}
        )
        
        gap = InformationGap(
            id="gap1",
            section_id="section1",
            gap_type=GapType.CONTENT,
            description="Need information about testing",
            priority=Priority.MEDIUM
        )
        
        content = self.processor.extract_relevant_content(result, gap)
        
        assert "main snippet content" in content
        assert "HTML snippet content" in content
        assert "Additional structured data" in content
    
    def test_calculate_relevance_score(self):
        """Test relevance score calculation"""
        gap = InformationGap(
            id="gap1",
            section_id="section1",
            gap_type=GapType.CONTENT,
            description="machine learning algorithms",
            priority=Priority.MEDIUM
        )
        
        query = SearchQuery(
            query="machine learning neural networks",
            priority=Priority.MEDIUM
        )
        
        # High relevance content
        high_relevance_content = "This article discusses machine learning algorithms and neural networks in detail. The research covers various machine learning approaches and their applications."
        
        score = self.processor.calculate_relevance_score(high_relevance_content, gap, query)
        assert score > 0.7
        
        # Low relevance content
        low_relevance_content = "This is about cooking recipes and has nothing to do with technology."
        
        score = self.processor.calculate_relevance_score(low_relevance_content, gap, query)
        assert score < 0.6
    
    @pytest.mark.asyncio
    async def test_enhance_content_with_kimi_success(self):
        """Test content enhancement with Kimi K2"""
        gap = InformationGap(
            id="gap1",
            section_id="section1",
            gap_type=GapType.CONTENT,
            description="Need information about AI",
            priority=Priority.MEDIUM
        )
        
        original_content = "AI is a broad field of computer science."
        
        with patch.object(self.processor.kimi_client, 'generate_text', new_callable=AsyncMock) as mock_generate:
            mock_response = Mock()
            mock_response.content = "Enhanced content: AI is a broad field that includes machine learning, natural language processing, and computer vision."
            mock_generate.return_value = mock_response
            
            enhanced_content = await self.processor.enhance_content_with_kimi(original_content, gap)
            
            assert "Enhanced content:" in enhanced_content
            assert len(enhanced_content) > len(original_content)
    
    @pytest.mark.asyncio
    async def test_enhance_content_with_kimi_failure(self):
        """Test content enhancement failure handling"""
        gap = InformationGap(
            id="gap1",
            section_id="section1",
            gap_type=GapType.CONTENT,
            description="Need information about AI",
            priority=Priority.MEDIUM
        )
        
        original_content = "AI is a broad field of computer science."
        
        with patch.object(self.processor.kimi_client, 'generate_text', new_callable=AsyncMock) as mock_generate:
            mock_generate.side_effect = Exception("API error")
            
            enhanced_content = await self.processor.enhance_content_with_kimi(original_content, gap)
            
            # Should return original content on failure
            assert enhanced_content == original_content

class TestDuplicateDetector:
    """Test duplicate detection functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.detector = DuplicateDetector()
    
    def test_is_duplicate_url(self):
        """Test URL duplicate detection"""
        url1 = "https://example.com/article"
        url2 = "https://example.com/article/"  # Same with trailing slash
        url3 = "https://example.com/different"
        
        # First URL should not be duplicate
        assert not self.detector.is_duplicate_url(url1)
        
        # Same URL (normalized) should be duplicate
        assert self.detector.is_duplicate_url(url2)
        
        # Different URL should not be duplicate
        assert not self.detector.is_duplicate_url(url3)
    
    def test_is_duplicate_content(self):
        """Test content duplicate detection"""
        content1 = "This is some unique content about machine learning."
        content2 = "This is some unique content about machine learning."  # Exact duplicate
        content3 = "This is different content about deep learning."
        
        # First content should not be duplicate
        assert not self.detector.is_duplicate_content(content1)
        
        # Same content should be duplicate
        assert self.detector.is_duplicate_content(content2)
        
        # Different content should not be duplicate
        assert not self.detector.is_duplicate_content(content3)

class TestDynamicRetrievalEngine:
    """Test the main dynamic retrieval engine"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engine = DynamicRetrievalEngine()
    
    @pytest.mark.asyncio
    async def test_retrieve_information_success(self):
        """Test successful information retrieval"""
        # Create test gaps
        gaps = [
            InformationGap(
                id="gap1",
                section_id="section1",
                gap_type=GapType.CONTENT,
                description="Need information about AI",
                priority=Priority.HIGH,
                search_queries=[
                    SearchQuery(query="artificial intelligence overview", priority=Priority.HIGH)
                ]
            )
        ]
        
        # Mock Google search response
        mock_search_response = GoogleSearchResponse(
            items=[
                GoogleSearchResult(
                    title="AI Overview",
                    link="https://example.com/ai",
                    snippet="Artificial intelligence is a field of computer science..."
                )
            ],
            total_results=100,
            search_time=0.3
        )
        
        with patch.object(self.engine.google_client, 'search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_search_response
            
            retrieved_info = await self.engine.retrieve_information(gaps, max_results_per_gap=5)
            
            assert len(retrieved_info) == 1
            assert retrieved_info[0].gap_id == "gap1"
            assert retrieved_info[0].source.url == "https://example.com/ai"
            assert retrieved_info[0].relevance_score > 0
            assert retrieved_info[0].credibility_score > 0
    
    @pytest.mark.asyncio
    async def test_retrieve_information_empty_gaps(self):
        """Test retrieval with empty gaps list"""
        retrieved_info = await self.engine.retrieve_information([], max_results_per_gap=5)
        assert len(retrieved_info) == 0
    
    @pytest.mark.asyncio
    async def test_retrieve_information_search_failure(self):
        """Test handling of search failures"""
        gaps = [
            InformationGap(
                id="gap1",
                section_id="section1",
                gap_type=GapType.CONTENT,
                description="Need information about AI",
                priority=Priority.HIGH,
                search_queries=[
                    SearchQuery(query="artificial intelligence", priority=Priority.HIGH)
                ]
            )
        ]
        
        with patch.object(self.engine.google_client, 'search', new_callable=AsyncMock) as mock_search:
            from services.google_search_client import GoogleSearchError
            mock_search.side_effect = GoogleSearchError("API error")
            
            # Should handle error gracefully and return empty list
            retrieved_info = await self.engine.retrieve_information(gaps, max_results_per_gap=5)
            assert len(retrieved_info) == 0
    
    @pytest.mark.asyncio
    async def test_search_and_process_with_duplicates(self):
        """Test search processing with duplicate filtering"""
        gap = InformationGap(
            id="gap1",
            section_id="section1",
            gap_type=GapType.CONTENT,
            description="Need information about AI",
            priority=Priority.HIGH
        )
        
        query = SearchQuery(query="artificial intelligence", priority=Priority.HIGH)
        
        # Mock search response with duplicates
        mock_search_response = GoogleSearchResponse(
            items=[
                GoogleSearchResult(
                    title="AI Article 1",
                    link="https://example.com/ai1",
                    snippet="First article about AI"
                ),
                GoogleSearchResult(
                    title="AI Article 2",
                    link="https://example.com/ai1",  # Duplicate URL
                    snippet="Second article about AI"
                ),
                GoogleSearchResult(
                    title="AI Article 3",
                    link="https://example.com/ai3",
                    snippet="First article about AI"  # Duplicate content
                )
            ],
            total_results=100
        )
        
        with patch.object(self.engine.google_client, 'search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_search_response
            
            retrieved_info = await self.engine._search_and_process(query, gap, 10)
            
            # Should filter out duplicates, leaving only 1 unique result
            assert len(retrieved_info) == 1
            assert retrieved_info[0].source.url == "https://example.com/ai1"
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check functionality"""
        with patch.object(self.engine.google_client, 'health_check', new_callable=AsyncMock) as mock_google_health:
            with patch.object(self.engine.content_processor.kimi_client, 'health_check', new_callable=AsyncMock) as mock_kimi_health:
                mock_google_health.return_value = True
                mock_kimi_health.return_value = True
                
                health_status = await self.engine.health_check()
                
                assert health_status["google_search"] is True
                assert health_status["kimi_k2"] is True

@pytest.mark.integration
class TestDynamicRetrievalEngineIntegration:
    """Integration tests for the dynamic retrieval engine"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_retrieval(self):
        """Test end-to-end retrieval process"""
        engine = DynamicRetrievalEngine()
        
        # Create realistic test gap
        gaps = [
            InformationGap(
                id="gap1",
                section_id="introduction",
                gap_type=GapType.CONTENT,
                description="Need current information about machine learning trends",
                priority=Priority.HIGH,
                search_queries=[
                    SearchQuery(
                        query="machine learning trends 2024",
                        priority=Priority.HIGH
                    )
                ]
            )
        ]
        
        # Mock the entire search process
        mock_results = [
            GoogleSearchResult(
                title="Machine Learning Trends 2024",
                link="https://techreview.com/ml-trends",
                snippet="The latest trends in machine learning include transformer models, federated learning, and automated ML pipelines."
            )
        ]
        
        mock_response = GoogleSearchResponse(
            items=mock_results,
            total_results=50,
            search_time=0.4
        )
        
        with patch.object(engine.google_client, 'search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_response
            
            retrieved_info = await engine.retrieve_information(gaps, max_results_per_gap=3)
            
            assert len(retrieved_info) > 0
            
            info = retrieved_info[0]
            assert info.gap_id == "gap1"
            assert info.source.title == "Machine Learning Trends 2024"
            assert info.relevance_score > 0
            assert info.credibility_score > 0
            assert "machine learning" in info.content.lower()

if __name__ == "__main__":
    pytest.main([__file__])