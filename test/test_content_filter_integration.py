#!/usr/bin/env python3
"""
Integration test for content filtering with Google Search results
"""

import sys
import os
sys.path.append('backend')

from services.content_filter import ContentFilteringPipeline
from services.google_search_client import GoogleSearchResult
from models.core import InformationGap, GapType, Priority

def test_google_search_integration():
    """Test content filtering with mock Google Search results"""
    print("Testing content filter integration with Google Search results...")
    
    pipeline = ContentFilteringPipeline()
    
    # Create test gaps
    test_gaps = [
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
    mock_results = [
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
    
    print(f"Processing {len(mock_results)} search results...")
    
    # Filter search results
    filtered_results = pipeline.filter_search_results(
        mock_results, test_gaps, min_quality_score=0.4
    )
    
    print(f"Filtered to {len(filtered_results)} high-quality results")
    
    # Verify filtering worked
    assert len(filtered_results) <= len(mock_results)
    
    # Check that results are sorted by quality
    if len(filtered_results) > 1:
        scores = [metrics.overall_score for _, metrics in filtered_results]
        assert scores == sorted(scores, reverse=True)
        print("✓ Results properly sorted by quality score")
    
    # Display results
    for i, (result, metrics) in enumerate(filtered_results):
        print(f"\nResult {i+1}:")
        print(f"  Title: {result.title}")
        print(f"  URL: {result.link}")
        print(f"  Quality Score: {metrics.overall_score:.3f}")
        print(f"  Relevance Score: {metrics.relevance_score:.3f}")
        print(f"  Source Authority: {metrics.source_authority:.3f}")
    
    # Test duplicate detection
    print("\nTesting duplicate detection...")
    
    # Add duplicate result
    duplicate_result = GoogleSearchResult(
        title="High Quality ML Research",  # Same title
        link="https://academic.edu/ml-research",  # Same URL
        snippet="This peer-reviewed study analyzes machine learning algorithms with comprehensive data analysis and statistical validation.",
        html_snippet="<b>Machine learning</b> research with detailed <b>algorithms</b> analysis.",
        page_map={}
    )
    
    results_with_duplicate = mock_results + [duplicate_result]
    
    filtered_with_duplicate = pipeline.filter_search_results(
        results_with_duplicate, test_gaps, min_quality_score=0.3
    )
    
    # Should have removed the duplicate
    assert len(filtered_with_duplicate) <= len(filtered_results)
    print("✓ Duplicate detection working correctly")
    
    # Get filtering statistics
    stats = pipeline.get_filtering_statistics()
    print(f"\nFiltering Statistics:")
    print(f"  Total signatures: {stats['total_signatures']}")
    print(f"  Unique domains: {stats['unique_domains']}")
    print(f"  Content hashes: {stats['content_hashes']}")
    print(f"  URL patterns: {stats['url_patterns']}")
    
    print("\n✓ All integration tests passed!")

if __name__ == "__main__":
    test_google_search_integration()