#!/usr/bin/env python3
"""
Simple test to verify domain evaluation functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.services.domain_adapter import DomainAdapter
from backend.services.domain_evaluation import DomainEvaluator, create_domain_evaluation_report
from backend.models.core import ResearchDomain
from backend.services.kimi_k2_client import KimiK2Client
from unittest.mock import Mock
import json

def test_domain_evaluation():
    """Test domain evaluation system"""
    print("Testing domain evaluation system...")
    
    # Mock Kimi K2 client
    mock_kimi_client = Mock(spec=KimiK2Client)
    
    # Create domain adapter
    adapter = DomainAdapter(mock_kimi_client)
    
    # Create evaluator
    evaluator = DomainEvaluator(adapter)
    
    # Test detection accuracy
    test_cases = [
        ("Machine Learning Algorithms", ResearchDomain.TECHNOLOGY),
        ("Clinical Trial Results", ResearchDomain.SCIENCE),
        ("Market Analysis Report", ResearchDomain.BUSINESS),
        ("Academic Research Paper", ResearchDomain.ACADEMIC),
        ("General Information", ResearchDomain.GENERAL)
    ]
    
    accuracy = evaluator.evaluate_detection_accuracy(test_cases)
    print(f"Detection accuracy: {accuracy:.2f}")
    
    # Test terminology consistency
    tech_content = "The API framework uses AI for ML processing. The SDK provides API documentation."
    consistency = evaluator.evaluate_terminology_consistency(tech_content, ResearchDomain.TECHNOLOGY)
    print(f"Terminology consistency: {consistency:.2f}")
    
    # Test format compliance
    compliance = evaluator.evaluate_format_compliance(tech_content, ResearchDomain.TECHNOLOGY)
    print(f"Format compliance: {compliance:.2f}")
    
    # Test comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation(test_cases)
    print(f"Overall evaluation score: {results['metrics']['overall_score']:.2f}")
    
    assert 0.0 <= accuracy <= 1.0
    assert 0.0 <= consistency <= 1.0
    assert 0.0 <= compliance <= 1.0
    assert 0.0 <= results['metrics']['overall_score'] <= 1.0
    
    print("✓ Domain evaluation test passed")

def test_evaluation_report():
    """Test evaluation report generation"""
    print("Testing evaluation report generation...")
    
    # Mock Kimi K2 client
    mock_kimi_client = Mock(spec=KimiK2Client)
    
    # Create domain adapter
    adapter = DomainAdapter(mock_kimi_client)
    
    # Generate evaluation report
    report = create_domain_evaluation_report(adapter)
    
    print(f"Report generated with {report['total_test_cases']} test cases")
    print(f"System supports {len(report['system_info']['domains_supported'])} domains")
    print(f"Overall system score: {report['metrics']['overall_score']:.2f}")
    print(f"Recommendations: {len(report['recommendations'])}")
    
    # Verify report structure
    assert "evaluation_timestamp" in report
    assert "metrics" in report
    assert "system_info" in report
    assert "recommendations" in report
    
    # Verify metrics
    metrics = report["metrics"]
    assert "detection_accuracy" in metrics
    assert "overall_score" in metrics
    assert 0.0 <= metrics["overall_score"] <= 1.0
    
    print("✓ Evaluation report test passed")

def test_domain_specific_evaluation():
    """Test domain-specific evaluation features"""
    print("Testing domain-specific evaluation...")
    
    # Mock Kimi K2 client
    mock_kimi_client = Mock(spec=KimiK2Client)
    
    # Create domain adapter
    adapter = DomainAdapter(mock_kimi_client)
    evaluator = DomainEvaluator(adapter)
    
    # Test each domain
    domain_scores = {}
    for domain in ResearchDomain:
        test_cases = [(f"Sample {domain.value} research topic", domain)]
        accuracy = evaluator.evaluate_detection_accuracy(test_cases)
        domain_scores[domain.value] = accuracy
        print(f"{domain.value} domain accuracy: {accuracy:.2f}")
    
    # Verify all domains were tested
    assert len(domain_scores) == len(ResearchDomain)
    
    # Verify all scores are valid
    for domain, score in domain_scores.items():
        assert 0.0 <= score <= 1.0
    
    print("✓ Domain-specific evaluation test passed")

if __name__ == "__main__":
    try:
        test_domain_evaluation()
        test_evaluation_report()
        test_domain_specific_evaluation()
        print("\n✅ All domain evaluation tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()