#!/usr/bin/env python3
"""
Simple test to verify content filter functionality
"""

import sys
import os
sys.path.append('backend')

from services.content_filter import ContentQualityAssessor, ContentFilteringPipeline
from models.core import InformationGap, GapType, Priority, Source

def test_basic_functionality():
    """Test basic content filter functionality"""
    print("Testing content filter basic functionality...")
    
    # Create test objects
    assessor = ContentQualityAssessor()
    pipeline = ContentFilteringPipeline()
    
    # Create test gap
    test_gap = InformationGap(
        id="gap1",
        section_id="intro",
        gap_type=GapType.CONTENT,
        description="machine learning algorithms research",
        priority=Priority.HIGH
    )
    
    # Create test source
    test_source = Source(
        url="https://example.com/research",
        title="Research Article",
        domain="example.com",
        credibility_score=0.8
    )
    
    # Test content quality assessment
    test_content = "This is a research study about machine learning algorithms with comprehensive analysis."
    
    metrics = assessor.assess_content_quality(test_content, test_source, test_gap)
    
    print(f"Overall quality score: {metrics.overall_score}")
    print(f"Readability score: {metrics.readability_score}")
    print(f"Information density: {metrics.information_density}")
    print(f"Relevance score: {metrics.relevance_score}")
    
    assert metrics.overall_score > 0.0
    assert metrics.readability_score >= 0.0
    assert metrics.information_density >= 0.0
    assert metrics.relevance_score >= 0.0
    
    print("✓ Content quality assessment working correctly")
    
    # Test filtering statistics
    stats = pipeline.get_filtering_statistics()
    print(f"Filtering statistics: {stats}")
    
    assert isinstance(stats, dict)
    assert all(isinstance(v, int) for v in stats.values())
    
    print("✓ Filtering statistics working correctly")
    print("All tests passed!")

if __name__ == "__main__":
    test_basic_functionality()