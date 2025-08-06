"""
Offline test for quality assessment system.
Tests the core functionality without any API calls.
"""

from unittest.mock import Mock, patch
from backend.models.core import (
    Draft, ResearchStructure, Section, ComplexityLevel, 
    ResearchDomain, DraftMetadata, ResearchRequirements, TTDRState, QualityMetrics
)
from backend.services.kimi_k2_quality_assessor import KimiK2QualityAssessor
from backend.workflow.quality_assessor_node import (
    quality_check_fallback, get_quality_summary, get_quality_grade, assess_improvement_potential
)

def test_quality_assessor_fallback_methods():
    """Test the fallback methods of quality assessor"""
    
    print("Testing quality assessor fallback methods...")
    
    # Create test data
    structure = ResearchStructure(
        sections=[
            Section(id="intro", title="Introduction"),
            Section(id="methods", title="Methods"),
            Section(id="results", title="Results")
        ],
        estimated_length=2000,
        complexity_level=ComplexityLevel.INTERMEDIATE,
        domain=ResearchDomain.TECHNOLOGY
    )
    
    draft = Draft(
        id="offline-test",
        topic="AI in Healthcare Offline Test",
        structure=structure,
        content={
            "intro": "This research explores AI applications in healthcare with comprehensive analysis of current trends and future possibilities in medical diagnosis and treatment...",
            "methods": "We employed systematic review methodology with rigorous selection criteria and comprehensive database searches across multiple medical databases...",
            "results": "Results demonstrate significant improvements in diagnostic accuracy, efficiency, and patient outcomes across multiple domains including radiology, pathology, and clinical decision support..."
        },
        metadata=DraftMetadata(),
        quality_score=0.0,
        iteration=1
    )
    
    # Test fallback methods directly
    assessor = KimiK2QualityAssessor()
    
    # Test content preparation
    content = assessor._prepare_draft_content(draft)
    print(f"✓ Content preparation: {len(content)} characters")
    assert draft.topic in content
    assert "Introduction" in content
    
    # Test fallback completeness assessment
    completeness = assessor._fallback_completeness_assessment(draft, content)
    print(f"✓ Fallback completeness: {completeness:.3f}")
    assert 0.0 <= completeness <= 1.0
    
    # Test fallback coherence assessment
    coherence = assessor._fallback_coherence_assessment(draft, content)
    print(f"✓ Fallback coherence: {coherence:.3f}")
    assert 0.0 <= coherence <= 1.0
    
    # Test fallback accuracy assessment
    accuracy = assessor._fallback_accuracy_assessment(draft, content)
    print(f"✓ Fallback accuracy: {accuracy:.3f}")
    assert 0.0 <= accuracy <= 1.0
    
    # Test fallback citation assessment
    citation_quality = assessor._fallback_citation_assessment(draft, content)
    print(f"✓ Fallback citation quality: {citation_quality:.3f}")
    assert 0.0 <= citation_quality <= 1.0
    
    print("✓ All fallback methods working correctly!")
    return True

def test_quality_utility_functions():
    """Test quality utility functions"""
    
    print("\nTesting quality utility functions...")
    
    # Test quality metrics
    quality_metrics = QualityMetrics(
        completeness=0.85,
        coherence=0.78,
        accuracy=0.82,
        citation_quality=0.65,
        overall_score=0.775
    )
    
    # Test quality summary
    summary = get_quality_summary(quality_metrics)
    print(f"✓ Quality summary: {summary}")
    assert summary["overall_score"] == 0.775
    assert summary["quality_grade"] in ["A", "B", "C", "D", "F"]
    
    # Test quality grades
    grades = {
        0.95: "A",
        0.85: "B", 
        0.75: "C",
        0.65: "D",
        0.45: "F"
    }
    
    for score, expected_grade in grades.items():
        grade = get_quality_grade(score)
        print(f"✓ Score {score} -> Grade {grade}")
        assert grade == expected_grade
    
    # Test improvement potential
    analysis = assess_improvement_potential(quality_metrics)
    print(f"✓ Improvement analysis: {analysis}")
    assert "current_score" in analysis
    assert "improvement_areas" in analysis
    assert "strengths" in analysis
    
    # Test with previous metrics
    previous_metrics = QualityMetrics(overall_score=0.7)
    analysis_with_trend = assess_improvement_potential(quality_metrics, previous_metrics)
    print(f"✓ Improvement trend: {analysis_with_trend['improvement_trend']}")
    assert analysis_with_trend["improvement_trend"] > 0
    assert analysis_with_trend["is_improving"] is True
    
    print("✓ All utility functions working correctly!")
    return True

def test_quality_check_logic():
    """Test quality check decision logic"""
    
    print("\nTesting quality check decision logic...")
    
    requirements = ResearchRequirements(
        quality_threshold=0.8,
        max_iterations=5
    )
    
    # Test scenarios
    scenarios = [
        {
            "name": "Continue - Low quality, iterations remaining",
            "quality_score": 0.6,
            "iteration": 2,
            "expected": "gap_analyzer"
        },
        {
            "name": "Stop - High quality",
            "quality_score": 0.85,
            "iteration": 2,
            "expected": "self_evolution_enhancer"
        },
        {
            "name": "Stop - Max iterations",
            "quality_score": 0.6,
            "iteration": 5,
            "expected": "self_evolution_enhancer"
        },
        {
            "name": "Continue - Moderate quality, early iteration",
            "quality_score": 0.7,
            "iteration": 1,
            "expected": "gap_analyzer"
        }
    ]
    
    for scenario in scenarios:
        quality_metrics = QualityMetrics(overall_score=scenario["quality_score"])
        state = {
            "quality_metrics": quality_metrics,
            "requirements": requirements,
            "iteration_count": scenario["iteration"]
        }
        
        decision = quality_check_fallback(state)
        print(f"✓ {scenario['name']}: {decision}")
        assert decision == scenario["expected"], f"Expected {scenario['expected']}, got {decision}"
    
    print("✓ All quality check scenarios working correctly!")
    return True

def test_error_handling():
    """Test error handling in quality assessment"""
    
    print("\nTesting error handling...")
    
    assessor = KimiK2QualityAssessor()
    
    # Test with empty draft
    empty_structure = ResearchStructure(
        sections=[],
        estimated_length=0,
        complexity_level=ComplexityLevel.BASIC,
        domain=ResearchDomain.GENERAL
    )
    
    empty_draft = Draft(
        id="empty-test",
        topic="Empty Test",
        structure=empty_structure,
        content={},
        metadata=DraftMetadata()
    )
    
    # Test fallback methods with empty content
    content = assessor._prepare_draft_content(empty_draft)
    completeness = assessor._fallback_completeness_assessment(empty_draft, content)
    print(f"✓ Empty draft completeness: {completeness:.3f}")
    assert completeness == 0.0
    
    # Test handle assessment result with exception
    exception = Exception("Test error")
    result = assessor._handle_assessment_result(exception, "test_metric", 0.5)
    print(f"✓ Exception handling: {result}")
    assert result == 0.5
    
    # Test with out-of-bounds values
    result = assessor._handle_assessment_result(1.5, "test_metric", 0.5)
    assert result == 1.0
    
    result = assessor._handle_assessment_result(-0.5, "test_metric", 0.5)
    assert result == 0.0
    
    print("✓ Error handling working correctly!")
    return True

if __name__ == "__main__":
    try:
        print("🧪 Running offline quality assessment tests...\n")
        
        test1 = test_quality_assessor_fallback_methods()
        test2 = test_quality_utility_functions()
        test3 = test_quality_check_logic()
        test4 = test_error_handling()
        
        if all([test1, test2, test3, test4]):
            print("\n🎉 All offline quality assessment tests passed!")
            print("✅ Quality assessment system is working correctly!")
        else:
            print("\n❌ Some tests failed!")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()