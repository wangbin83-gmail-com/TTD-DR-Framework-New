"""
Simple integration test for quality assessment system.
Tests the complete workflow integration.
"""

import asyncio
from backend.models.core import (
    Draft, ResearchStructure, Section, ComplexityLevel, 
    ResearchDomain, DraftMetadata, ResearchRequirements, TTDRState
)
from backend.workflow.quality_assessor_node import quality_assessor_node, quality_check_node
from backend.services.kimi_k2_quality_assessor import KimiK2QualityAssessor

def test_quality_assessment_integration():
    """Test complete quality assessment integration"""
    
    # Create test data
    structure = ResearchStructure(
        sections=[
            Section(id="intro", title="Introduction", content="Introduction content"),
            Section(id="methods", title="Methods", content="Methods content"),
            Section(id="results", title="Results", content="Results content")
        ],
        estimated_length=2000,
        complexity_level=ComplexityLevel.INTERMEDIATE,
        domain=ResearchDomain.TECHNOLOGY
    )
    
    draft = Draft(
        id="integration-test",
        topic="AI in Healthcare Integration Test",
        structure=structure,
        content={
            "intro": "This research explores AI applications in healthcare with comprehensive analysis...",
            "methods": "We employed systematic review methodology with rigorous selection criteria...",
            "results": "Results demonstrate significant improvements in diagnostic accuracy and efficiency..."
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
        "topic": "AI in Healthcare Integration Test",
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
    
    print("Testing quality assessment integration...")
    
    # Test quality assessment
    try:
        assessed_state = quality_assessor_node(initial_state)
        
        print(f"✓ Quality assessment completed")
        print(f"  Overall score: {assessed_state['quality_metrics'].overall_score:.3f}")
        print(f"  Completeness: {assessed_state['quality_metrics'].completeness:.3f}")
        print(f"  Coherence: {assessed_state['quality_metrics'].coherence:.3f}")
        print(f"  Accuracy: {assessed_state['quality_metrics'].accuracy:.3f}")
        print(f"  Citation quality: {assessed_state['quality_metrics'].citation_quality:.3f}")
        
        # Test quality check decision
        decision = quality_check_node(assessed_state)
        print(f"✓ Quality check decision: {decision}")
        
        # Verify results
        assert assessed_state["quality_metrics"] is not None
        assert 0.0 <= assessed_state["quality_metrics"].overall_score <= 1.0
        assert decision in ["gap_analyzer", "self_evolution_enhancer"]
        
        print("✓ Integration test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_quality_assessment_integration()
    if success:
        print("\n🎉 Quality assessment system integration test successful!")
    else:
        print("\n❌ Quality assessment system integration test failed!")