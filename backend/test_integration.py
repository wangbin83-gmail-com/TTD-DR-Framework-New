#!/usr/bin/env python3
"""
Integration test to verify the TTD-DR framework setup
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_core_models():
    """Test core data models"""
    print("Testing core models...")
    
    from models.core import (
        Source, Section, ResearchStructure, Draft, InformationGap,
        QualityMetrics, ResearchRequirements, TTDRState,
        GapType, Priority, ComplexityLevel, ResearchDomain
    )
    
    # Test Source model
    source = Source(
        url="https://example.com",
        title="Test Article",
        domain="example.com",
        credibility_score=0.8
    )
    assert source.url == "https://example.com"
    print("✓ Source model works")
    
    # Test Section model
    section = Section(
        id="intro",
        title="Introduction",
        content="Test content",
        estimated_length=500
    )
    assert section.id == "intro"
    print("✓ Section model works")
    
    # Test ResearchStructure model
    structure = ResearchStructure(
        sections=[section],
        estimated_length=2000,
        complexity_level=ComplexityLevel.INTERMEDIATE,
        domain=ResearchDomain.TECHNOLOGY
    )
    assert len(structure.sections) == 1
    print("✓ ResearchStructure model works")
    
    # Test Draft model
    draft = Draft(
        id="draft_1",
        topic="AI in Healthcare",
        structure=structure,
        content={"intro": "Introduction content"},
        quality_score=0.7,
        iteration=1
    )
    assert draft.topic == "AI in Healthcare"
    print("✓ Draft model works")
    
    # Test InformationGap model
    gap = InformationGap(
        id="gap_1",
        section_id="intro",
        gap_type=GapType.CONTENT,
        description="Missing background information",
        priority=Priority.HIGH
    )
    assert gap.gap_type == GapType.CONTENT
    print("✓ InformationGap model works")
    
    # Test QualityMetrics model
    metrics = QualityMetrics(
        completeness=0.8,
        coherence=0.7,
        accuracy=0.9,
        citation_quality=0.6
    )
    # Overall score should be calculated automatically
    expected_overall = (0.8 + 0.7 + 0.9 + 0.6) / 4
    assert abs(metrics.overall_score - expected_overall) < 0.001
    print("✓ QualityMetrics model works")
    
    print("All core models working correctly!")

def test_validation():
    """Test validation utilities"""
    print("\nTesting validation utilities...")
    
    from models.validation import DataValidator, TTDRStateValidator
    from models.core import Source, ResearchRequirements, ResearchDomain, ComplexityLevel
    
    # Test model validation
    source_data = {
        "url": "https://example.com",
        "title": "Test Article",
        "domain": "example.com",
        "credibility_score": 0.8
    }
    
    result = DataValidator.validate_model(Source, source_data)
    assert result.is_valid
    assert result.data.url == "https://example.com"
    print("✓ Model validation works")
    
    # Test serialization
    source = Source(**source_data)
    serialized = DataValidator.serialize_model(source)
    assert serialized['url'] == "https://example.com"
    print("✓ Model serialization works")
    
    # Test TTDRState creation
    requirements = ResearchRequirements(
        domain=ResearchDomain.TECHNOLOGY,
        complexity_level=ComplexityLevel.INTERMEDIATE
    )
    
    state = TTDRStateValidator.create_initial_state("AI in Healthcare", requirements)
    assert state['topic'] == "AI in Healthcare"
    assert state['iteration_count'] == 0
    print("✓ TTDRState creation works")
    
    print("All validation utilities working correctly!")

def test_kimi_k2_client():
    """Test Kimi K2 client setup"""
    print("\nTesting Kimi K2 client...")
    
    from services.kimi_k2_client import KimiK2Client, KimiK2Response, KimiK2Error, RateLimiter
    
    # Test client initialization
    client = KimiK2Client()
    assert client.model == "moonshot-v1-8k"
    assert client.max_tokens == 4000
    print("✓ Kimi K2 client initialization works")
    
    # Test rate limiter
    limiter = RateLimiter(max_requests=5, time_window=60)
    assert limiter.max_requests == 5
    print("✓ Rate limiter initialization works")
    
    # Test response model
    response = KimiK2Response(
        content="Test response",
        usage={"prompt_tokens": 10, "completion_tokens": 5},
        model="moonshot-v1-8k",
        finish_reason="stop"
    )
    assert response.content == "Test response"
    print("✓ KimiK2Response model works")
    
    print("Kimi K2 client setup working correctly!")

def test_configuration():
    """Test configuration setup"""
    print("\nTesting configuration...")
    
    from config.settings import settings
    
    # Test default values
    assert settings.kimi_k2_base_url == "https://api.moonshot.cn/v1"
    assert settings.kimi_k2_model == "moonshot-v1-8k"
    assert settings.kimi_k2_max_tokens == 4000
    assert settings.kimi_k2_temperature == 0.7
    print("✓ Configuration defaults work")
    
    print("Configuration working correctly!")

def main():
    """Run all integration tests"""
    print("=" * 60)
    print("TTD-DR Framework Integration Test")
    print("=" * 60)
    
    try:
        test_core_models()
        test_validation()
        test_kimi_k2_client()
        test_configuration()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! TTD-DR Framework setup is working correctly!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)