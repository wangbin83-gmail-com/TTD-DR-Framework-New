#!/usr/bin/env python3
"""
Simple test to verify domain adapter functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.services.domain_adapter import DomainAdapter
from backend.models.core import ResearchDomain, ComplexityLevel, ResearchRequirements
from backend.services.kimi_k2_client import KimiK2Client
from unittest.mock import Mock
import json

def test_domain_detection():
    """Test basic domain detection"""
    print("Testing domain detection...")
    
    # Mock Kimi K2 client - we'll mock the synchronous wrapper method
    mock_kimi_client = Mock(spec=KimiK2Client)
    
    # Create domain adapter
    adapter = DomainAdapter(mock_kimi_client)
    
    # Test detection
    result = adapter.detect_domain("Machine Learning Algorithms")
    
    print(f"Detected domain: {result.primary_domain}")
    print(f"Confidence: {result.confidence}")
    print(f"Keywords found: {result.keywords_found}")
    
    assert result.primary_domain == ResearchDomain.TECHNOLOGY
    assert result.confidence > 0.1  # Lower threshold for keyword-based detection
    
    print("✓ Domain detection test passed")

def test_requirements_adaptation():
    """Test requirements adaptation"""
    print("Testing requirements adaptation...")
    
    mock_kimi_client = Mock(spec=KimiK2Client)
    adapter = DomainAdapter(mock_kimi_client)
    
    # Create sample requirements
    requirements = ResearchRequirements(
        domain=ResearchDomain.GENERAL,
        complexity_level=ComplexityLevel.INTERMEDIATE,
        max_iterations=5,
        quality_threshold=0.8,
        max_sources=20
    )
    
    # Mock domain result
    from backend.services.domain_adapter import DomainDetectionResult
    domain_result = DomainDetectionResult(
        primary_domain=ResearchDomain.TECHNOLOGY,
        confidence=0.9,
        secondary_domains=[],
        detection_method="test",
        keywords_found=["software"],
        reasoning="Test"
    )
    
    # Test adaptation
    adapted = adapter.adapt_research_requirements(requirements, domain_result)
    
    print(f"Original domain: {requirements.domain}")
    print(f"Adapted domain: {adapted.domain}")
    print(f"Quality threshold: {adapted.quality_threshold}")
    
    assert adapted.domain == ResearchDomain.TECHNOLOGY
    assert adapted.quality_threshold >= requirements.quality_threshold
    
    print("✓ Requirements adaptation test passed")

def test_terminology_handling():
    """Test terminology handling"""
    print("Testing terminology handling...")
    
    mock_kimi_client = Mock(spec=KimiK2Client)
    adapter = DomainAdapter(mock_kimi_client)
    
    # Test terminology handlers
    tech_handler = adapter.terminology_handlers[ResearchDomain.TECHNOLOGY]
    
    print(f"Technology abbreviations: {len(tech_handler.abbreviations)}")
    print(f"AI expansion: {tech_handler.abbreviations.get('AI', 'Not found')}")
    
    assert len(tech_handler.abbreviations) > 0
    assert "AI" in tech_handler.abbreviations
    assert tech_handler.abbreviations["AI"] == "Artificial Intelligence"
    
    print("✓ Terminology handling test passed")

if __name__ == "__main__":
    try:
        test_domain_detection()
        test_requirements_adaptation()
        test_terminology_handling()
        print("\n✅ All domain adapter tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()