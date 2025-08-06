"""
Unit tests for Content Filtering System.
Tests quality assessment, duplicate detection, and filtering pipeline.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.content_filter import (
    ContentQualityAssessor, AdvancedDuplicateDetector, ContentFilteringPipeline,
    ContentQualityMetrics, DuplicationSignature
)
from services.google_search_client import GoogleSearchResult
from models.core import InformationGap, GapType, Priority, Source

class TestContentQualityAssessor:
    """Test content quality assessment functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.assessor = ContentQualityAssessor()
        
        self.test_gap = InformationGap(
            id="gap1",
            section_id="intro",
            gap_type=GapType.CONTENT,
            description="machine learning algorithms research",
            priority=Priority.HIGH
        )
        
        self.test_source = Source(
            url="https://example.com/research",
            title="Research Article",
            domain="example.com",
            credibility_score=0.8
        )
    
    def test_assess_readability_good_content(self):
        """Test readability assessment for well-structured content"""
        good_content = """
        This is a well-structured article about machine learning. 
        It contains multiple sentences with appropriate length. 
        The content is organized in clear paragraphs.
        
        Each paragraph discusses a specific aspect of the topic.
        The writing style is professional and informative.
        """
        
        score = self.assessor._assess_readability(good_content)
        assert score > 0.7
    
    def test_assess_readability_poor_content(self):
        """Test readability assessment for poorly structured content"""
        poor_content = "THIS IS ALL CAPS AND VERY HARD TO READ WITH NO PROPER STRUCTURE OR PUNCTUATION"
        
        score = self.assessor._assess_readability(poor_content)
        assert score < 0.5
    
    def test_assess_readability_empty_content(self):
        """Test readability assessment for empty content"""
        score = self.assessor._assess_readability("")
        assert score == 0.0
    
    def test_assess_information_density_high(self):
        """Test information density assessment for information-rich content"""
        dense_content = """
        The research study analyzed data from 10,000 participants over 5 years.
        Results showed a 25% increase in accuracy with the new algorithm.
        Statistical analysis revealed significant correlations between variables.
        The findings provide evidence for the proposed hypothesis.
        """
        
        score = self.assessor._assess_information_density(dense_content)
        assert score > 0.7
    
    def test_assess_information_density_low(self):
        """Test information density assessment for low-information content"""
        sparse_content = "This is just some random text without any real information or data."
        
        score = self.assessor._assess_information_density(sparse_content)
        assert score < 0.5
    
    def test_assess_factual_indicators_academic(self):
        """Test factual indicators assessment for academic content"""
        academic_content = """
        This peer-reviewed research study presents findings from a controlled experiment.
        The methodology involved statistical analysis of collected data.
        Results demonstrate significant evidence supporting the hypothesis.
        """
        
        score = self.assessor._assess_factual_indicators(academic_content, self.test_gap)
        assert score > 0.6
    
    def test_assess_factual_indicators_low_quality(self):
        """Test factual indicators assessment for low-quality content"""
        low_quality_content = """
        Click here for amazing results! You won't believe this incredible discovery!
        Subscribe now for more shocking content that doctors hate!
        """
        
        score = self.assessor._assess_factual_indicators(low_quality_content, self.test_gap)
        assert score < 0.4
    
    def test_assess_recency_recent_source(self):
        """Test recency assessment for recent sources"""
        recent_source = Source(
            url="https://example.com",
            title="Recent Article",
            domain="example.com",
            credibility_score=0.8,
            last_accessed=datetime.now() - timedelta(days=10)
        )
        
        score = self.assessor._assess_recency(recent_source)
        assert score == 1.0
    
    def test_assess_recency_old_source(self):
        """Test recency assessment for old sources"""
        old_source = Source(
            url="https://example.com",
            title="Old Article",
            domain="example.com",
            credibility_score=0.8,
            last_accessed=datetime.now() - timedelta(days=800)
        )
        
        score = self.assessor._assess_recency(old_source)
        assert score < 0.5
    
    def test_assess_relevance_high(self):
        """Test relevance assessment for highly relevant content"""
        relevant_content = """
        This article discusses machine learning algorithms in detail.
        It covers various research approaches and algorithmic implementations.
        The content focuses on machine learning research methodologies.
        """
        
        score = self.assessor._assess_relevance(relevant_content, self.test_gap)
        assert score > 0.7
    
    def test_assess_relevance_low(self):
        """Test relevance assessment for low relevance content"""
        irrelevant_content = "This article is about cooking recipes and has nothing to do with technology."
        
        score = self.assessor._assess_relevance(irrelevant_content, self.test_gap)
        assert score < 0.3
    
    def test_determine_content_type_academic(self):
        """Test content type determination for academic content"""
        academic_content = "This peer-reviewed research study presents findings from experiments."
        
        content_type = self.assessor._determine_content_type(academic_content, self.test_gap)
        assert content_type == "academic"
    
    def test_determine_content_type_technical(self):
        """Test content type determination for technical content"""
        technical_content = "This implementation guide covers algorithm optimization and framework architecture."
        
        content_type = self.assessor._determine_content_type(technical_content, self.test_gap)
        assert content_type == "technical"
    
    def test_assess_content_quality_comprehensive(self):
        """Test comprehensive content quality assessment"""
        quality_content = """
        This peer-reviewed research study analyzed machine learning algorithms over 2 years.
        The methodology involved statistical analysis of performance data from 1000 experiments.
        Results showed significant improvements in accuracy and efficiency.
        
        The findings provide strong evidence supporting the proposed algorithmic approach.
        Data visualization and detailed analysis are included in the appendix.
        """
        
        metrics = self.assessor.assess_content_quality(quality_content, self.test_source, self.test_gap)
        
        assert metrics.overall_score > 0.6
        assert metrics.readability_score > 0.5
        assert metrics.information_density > 0.5
        assert metrics.factual_indicators > 0.5
        assert metrics.relevance_score > 0.5

class TestAdvancedDuplicateDetector:
    """Test advanced duplicate detection functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.detector = AdvancedDuplicateDetector(similarity_threshold=0.8)
    
    def test_hash_content_normalization(self):
        """Test content hashing with normalization"""
        content1 = "This is a test content with   extra spaces."
        content2 = "This is a test content with extra spaces."
        content3 = "This is completely different content."
        
        hash1 = self.detector._hash_content(content1)
        hash2 = self.detector._hash_content(content2)
        hash3 = self.detector._hash_content(content3)
        
        assert hash1 == hash2  # Should be same after normalization
        assert hash1 != hash3  # Should be different
    
    def test_normalize_url(self):
        """Test URL normalization"""
        url1 = "https://example.com/path?param=value#fragment"
        url2 = "https://example.com/path/"
        url3 = "https://example.com/path"
        
        normalized1 = self.detector._normalize_url(url1)
        normalized2 = self.detector._normalize_url(url2)
        normalized3 = self.detector._normalize_url(url3)
        
        assert normalized1 == "example.com/path"
        assert normalized2 == "example.com/path"
        assert normalized3 == "example.com/path"
    
    def test_duplicate_detection_exact_match(self):
        """Test duplicate detection for exact content matches"""
        content = "This is test content for duplicate detection."
        title = "Test Article"
        url = "https://example.com/test"
        
        # First occurrence should not be duplicate
        is_dup1 = self.detector.is_duplicate(content, title, url)
        assert not is_dup1
        
        # Second occurrence should be duplicate
        is_dup2 = self.detector.is_duplicate(content, title, url)
        assert is_dup2
    
    def test_duplicate_detection_url_pattern(self):
        """Test duplicate detection based on URL patterns"""
        content1 = "Different content 1"
        content2 = "Different content 2"
        title1 = "Article 1"
        title2 = "Article 2"
        url1 = "https://example.com/article?id=1"
        url2 = "https://example.com/article?id=2"
        
        # First URL should not be duplicate
        is_dup1 = self.detector.is_duplicate(content1, title1, url1)
        assert not is_dup1
        
        # Different URL pattern should not be duplicate
        is_dup2 = self.detector.is_duplicate(content2, title2, url2)
        assert not is_dup2
    
    def test_duplicate_detection_same_domain_similar_title(self):
        """Test duplicate detection for same domain with similar titles"""
        content1 = "Content about machine learning algorithms"
        content2 = "Different content about neural networks"
        title1 = "Machine Learning Research"
        title2 = "Machine Learning Research"  # Same title
        url1 = "https://example.com/ml-research-1"
        url2 = "https://example.com/ml-research-2"
        
        # First should not be duplicate
        is_dup1 = self.detector.is_duplicate(content1, title1, url1)
        assert not is_dup1
        
        # Same domain, same title should be considered duplicate
        is_dup2 = self.detector.is_duplicate(content2, title2, url2)
        assert is_dup2

class TestContentFilteringPipeline:
    """Test the main content filtering pipeline"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.pipeline = ContentFilteringPipeline()
        
        self.test_gaps = [
            InformationGap(
                id="gap1",
                section_id="intro",
                gap_type=GapType.CONTENT,
                description="machine learning algorithms research",
                priority=Priority.HIGH
            ),
            InformationGap(
                id="gap2",
                section_id="methods",
                gap_type=GapType.EVIDENCE,
                description="neural network performance data",
                priority=Priority.MEDIUM
            )
        ]
        
        # Mock Google search results
        self.mock_results = [
            GoogleSearchResult(
                title="High Quality ML Research",
                link="https://academic.edu/ml-research",
                snippet="This peer-reviewed study analyzes machine learning algorithms with comprehensive data analysis and statistical validation.",
                html_snippet="<b>Machine learning</b> research with detailed <b>algorithms</b> analysis.",
                page_map={}
            ),
            GoogleSearchResult(
                title="Click Here for Amazing ML Tricks!",
                link="https://spam-site.com/ml-tricks",
                snippet="You won't believe these incredible machine learning tricks! Click here for amazing results!",
                html_snippet="<b>Amazing</b> ML tricks that <b>doctors hate</b>!",
                page_map={}
            ),
            GoogleSearchResult(
                title="Neural Network Performance Study",
                link="https://research.org/nn-performance",
                snippet="Comprehensive analysis of neural network performance across multiple datasets with statistical significance testing.",
                html_snippet="<b>Neural network</b> performance with <b>statistical</b> analysis.",
                page_map={}
            )
        ]
    
    def test_filter_search_results_quality_threshold(self):
        """Test filtering based on quality threshold"""
        filtered_results = self.pipeline.filter_search_results(
            self.mock_results, self.test_gaps, min_quality_score=0.5
        )
        
        # Should filter out low quality results
        assert len(filtered_results) < len(self.mock_results)
        
        # All remaining results should meet quality threshold
        for result, metrics in filtered_results:
            assert metrics.overall_score >= 0.5
    
    def test_filter_search_results_sorting(self):
        """Test that results are sorted by quality score"""
        filtered_results = self.pipeline.filter_search_results(
            self.mock_results, self.test_gaps, min_quality_score=0.3
        )
        
        # Results should be sorted by quality score (descending)
        scores = [metrics.overall_score for _, metrics in filtered_results]
        assert scores == sorted(scores, reverse=True)
    
    def test_filter_search_results_duplicate_removal(self):
        """Test duplicate removal in filtering"""
        # Add duplicate result
        duplicate_result = GoogleSearchResult(
            title="High Quality ML Research",  # Same title
            link="https://academic.edu/ml-research",  # Same URL
            snippet="This peer-reviewed study analyzes machine learning algorithms with comprehensive data analysis and statistical validation.",
            html_snippet="<b>Machine learning</b> research with detailed <b>algorithms</b> analysis.",
            page_map={}
        )
        
        results_with_duplicate = self.mock_results + [duplicate_result]
        
        filtered_results = self.pipeline.filter_search_results(
            results_with_duplicate, self.test_gaps, min_quality_score=0.3
        )
        
        # Should have removed the duplicate
        assert len(filtered_results) <= len(self.mock_results)
    
    def test_get_filtering_statistics(self):
        """Test filtering statistics collection"""
        # Process some results first
        self.pipeline.filter_search_results(
            self.mock_results, self.test_gaps, min_quality_score=0.3
        )
        
        stats = self.pipeline.get_filtering_statistics()
        
        assert "total_signatures" in stats
        assert "unique_domains" in stats
        assert "content_hashes" in stats
        assert "url_patterns" in stats
        assert all(isinstance(v, int) for v in stats.values())

class TestIntegrationWithGoogleSearchResults:
    """Test integration with actual Google Search result structures"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.pipeline = ContentFilteringPipeline()
        self.test_gap = InformationGap(
            id="gap1",
            section_id="intro",
            gap_type=GapType.CONTENT,
            description="artificial intelligence research",
            priority=Priority.HIGH
        )
    
    def test_process_google_search_result_structure(self):
        """Test processing of Google Search API result structure"""
        # Simulate realistic Google Search result
        google_result = GoogleSearchResult(
            title="Artificial Intelligence Research: Current Trends and Future Directions",
            link="https://ai-research-journal.org/current-trends-2024",
            snippet="This comprehensive review examines current trends in artificial intelligence research, including machine learning, deep learning, and neural networks. The study analyzes recent developments and identifies future research directions based on systematic literature review of 500+ papers.",
            html_snippet="<b>Artificial intelligence</b> research trends including <b>machine learning</b> and <b>deep learning</b> developments.",
            page_map={
                "metatags": [{"name": "author", "content": "Dr. Jane Smith"}],
                "cse_thumbnail": [{"src": "https://example.com/thumb.jpg"}]
            }
        )
        
        filtered_results = self.pipeline.filter_search_results(
            [google_result], [self.test_gap], min_quality_score=0.4
        )
        
        assert len(filtered_results) == 1
        result, metrics = filtered_results[0]
        
        # Verify result structure is preserved
        assert result.title == google_result.title
        assert result.link == google_result.link
        assert result.snippet == google_result.snippet
        
        # Verify quality metrics are calculated
        assert metrics.overall_score > 0.4
        assert metrics.relevance_score > 0.5  # Should be relevant to AI research
    
    def test_handle_missing_optional_fields(self):
        """Test handling of Google Search results with missing optional fields"""
        minimal_result = GoogleSearchResult(
            title="AI Research",
            link="https://example.com/ai",
            snippet="Basic information about AI research.",
            html_snippet=None,  # Missing
            page_map=None  # Missing
        )
        
        # Should not raise exceptions
        filtered_results = self.pipeline.filter_search_results(
            [minimal_result], [self.test_gap], min_quality_score=0.1
        )
        
        assert len(filtered_results) <= 1  # May be filtered out due to low quality
    
    def test_empty_results_handling(self):
        """Test handling of empty search results"""
        filtered_results = self.pipeline.filter_search_results(
            [], [self.test_gap], min_quality_score=0.5
        )
        
        assert filtered_results == []
        
        # Statistics should still work
        stats = self.pipeline.get_filtering_statistics()
        assert all(isinstance(v, int) for v in stats.values())

if __name__ == "__main__":
    pytest.main([__file__])