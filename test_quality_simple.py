"""
Simple test for quality assessment system without API calls.
Tests the fallback logic and basic functionality.
"""

from backend.models.core import (
    Draft, ResearchStructure, Section, ComplexityLevel, 
    ResearchDomain, DraftMetadata, ResearchRequirements, TTDRState, QualityMetrics
)
from backend.workflow.quality_assessor_node import (
    quality_assessor_node, quality_check_node, quality_check_fallback,
    get_quality_summary, get_quality_grade, assess_improvement_potential
)

def test_quality_assessment_fallback():
    """Test quality assessment with fallback logic"""
    
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
        id="fallback-test",
        topic="AI in Healthcare Fallback Test",
        structure=structure,
        content={
            "intro": "This research explores AI applications in healthcare with comprehensive analysis of current trends and future possibilities...",
            "methods": "We employed systematic review methodology with rigorous selection criteria and comprehensive database searches...",
            "results": "Results demonstrate significant improvements in diagnostic accuracy, efficiency, and patient outcomes across multiple domains..."
        },
        metadata=DraftMetadata(),
        quality_score=0.0,
        iteration=1
    )
    
    requirements = ResearchRequirements(
        domain=ResearchDomain.TECHNOLOGY,
        complexity_level=ComplexityLevel.INTERMEDIATE,
        max_iterations=5,
        quality_threshold=0.8,
        max_sources=20
    )
    
    # Create initial state
    initial_state: TTDRState = {
        "topic": "AI in Healthcare Fallback Test",
        "requirements": requirements,
        "current_draft": draft,
        "information_gaps": [],
        "retrieved_info": [],
        "iteration_count": 1,
        "quality_metrics": None,
        "evolution_history": [],
        "final_report": None,
        "error_log": []
    }
    
    print("Testing quality assessment fallback logic...")
    
    # Test quality assessment (will use fallback since no API key)
    assessed_state = quality_assessor_node(initial_state)
    
    print(f"✓ Quality assessment completed with fallback")
    print(f"  Overall score: {assessed_state['quality_metrics'].overall_score:.3f}")
    print(f"  Completeness: {assessed_state['quality_metrics'].completeness:.3f}")
    print(f"  Coherence: {assessed_state['quality_metrics'].coherence:.3f}")
    print(f"  Accuracy: {assessed_state['quality_metrics'].accuracy:.3f}")
    print(f"  Citation quality: {assessed_state['quality_metrics'].citation_quality:.3f}")
    
    # Test quality check decision
    decision = quality_check_node(assessed_state)
    print(f"✓ Quality check decision: {decision}")
    
    # Test utility functions
    summary = get_quality_summary(assessed_state['quality_metrics'])
    print(f"✓ Quality summary: Grade {summary['quality_grade']}")
    
    grade = get_quality_grade(assessed_state['quality_metrics'].overall_score)
    print(f"✓ Quality grade: {grade}")
    
    improvement_analysis = assess_improvement_potential(assessed_state['quality_metrics'])
    print(f"✓ Improvement areas: {improvement_analysis['improvement_areas']}")
    print(f"✓ Strengths: {improvement_analysis['strengths']}")
    
    # Verify results
    assert assessed_state["quality_metrics"] is not None
    assert 0.0 <= assessed_state["quality_metrics"].overall_score <= 1.0
    assert decision in ["gap_analyzer", "self_evolution_enhancer"]
    assert grade in ["A", "B", "C", "D", "F"]
    
    print("✓ All tests passed!")
    return True

def test_quality_check_scenarios():
    """Test different quality check scenarios"""
    
    print("\nTesting quality check scenarios...")
    
    requirements = ResearchRequirements(
        quality_threshold=0.8,
        max_iterations=5
    )
    
    # Test 1: Quality below threshold, iterations remaining
    low_quality_metrics = QualityMetrics(overall_score=0.6)
    state_low_quality = {
        "quality_metrics": low_quality_metrics,
        "requirements": requirements,
        "iteration_count": 2
    }
    
    decision = quality_check_fallback(state_low_quality)
    print(f"✓ Low quality scenario: {decision} (should continue)")
    assert decision == "gap_analyzer"
    
    # Test 2: Quality above threshold
    high_quality_metrics = QualityMetrics(overall_score=0.85)
    state_high_quality = {
        "quality_metrics": high_quality_metrics,
        "requirements": requirements,
        "iteration_count": 2
    }
    
    decision = quality_check_fallback(state_high_quality)
    print(f"✓ High quality scenario: {decision} (should stop)")
    assert decision == "self_evolution_enhancer"
    
    # Test 3: Max iterations reached
    state_max_iterations = {
        "quality_metrics": low_quality_metrics,
        "requirements": requirements,
        "iteration_count": 5
    }
    
    decision = quality_check_fallback(state_max_iterations)
    print(f"✓ Max iterations scenario: {decision} (should stop)")
    assert decision == "self_evolution_enhancer"
    
    print("✓ All quality check scenarios passed!")
    return True

if __name__ == "__main__":
    try:
        success1 = test_quality_assessment_fallback()
        success2 = test_quality_check_scenarios()
        
        if success1 and success2:
            print("\n🎉 Quality assessment system tests successful!")
        else:
            print("\n❌ Some tests failed!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()