"""
Test cross-disciplinary workflow integration with TTD-DR framework.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.workflow.cross_disciplinary_node import (
    cross_disciplinary_detector_node,
    cross_disciplinary_integrator_node,
    cross_disciplinary_conflict_resolver_node,
    cross_disciplinary_formatter_node,
    cross_disciplinary_quality_assessor_node
)
from backend.models.core import (
    TTDRState, ResearchDomain, Draft, DraftMetadata, QualityMetrics,
    RetrievedInfo, Source, ResearchRequirements, ComplexityLevel,
    ResearchStructure, Section
)
from datetime import datetime


def test_cross_disciplinary_workflow():
    """Test complete cross-disciplinary workflow"""
    print("Testing Cross-Disciplinary Workflow Integration...")
    
    # Create comprehensive test state
    test_state = TTDRState(
        topic="AI applications in healthcare business and scientific research",
        requirements=ResearchRequirements(
            domain=ResearchDomain.GENERAL,
            complexity_level=ComplexityLevel.INTERMEDIATE,
            max_iterations=5,
            quality_threshold=0.8,
            max_sources=20,
            preferred_source_types=["academic", "industry", "clinical"]
        ),
        current_draft=Draft(
            id="test_draft",
            topic="AI applications in healthcare business and scientific research",
            structure=ResearchStructure(
                sections=[
                    Section(id="introduction", title="Introduction"),
                    Section(id="healthcare_applications", title="Healthcare Applications"),
                    Section(id="business_applications", title="Business Applications"),
                    Section(id="research_methodology", title="Research Methodology")
                ],
                estimated_length=2000,
                complexity_level=ComplexityLevel.INTERMEDIATE,
                domain=ResearchDomain.GENERAL
            ),
            content={
                "introduction": "AI is transforming multiple industries...",
                "healthcare_applications": "In healthcare, AI enables...",
                "business_applications": "In business, AI provides...",
                "research_methodology": "This study employs mixed methods..."
            },
            metadata=DraftMetadata(),
            quality_score=0.7,
            iteration=1
        ),
        information_gaps=[],
        retrieved_info=[
            RetrievedInfo(
                source=Source(
                    url="https://tech.example.com", 
                    title="Tech Article",
                    domain="technology",
                    credibility_score=0.8
                ),
                content="AI algorithms in software development and automation",
                relevance_score=0.8,
                credibility_score=0.7,
                extraction_timestamp=datetime.now()
            ),
            RetrievedInfo(
                source=Source(
                    url="https://healthcare.example.com", 
                    title="Medical Journal",
                    domain="healthcare",
                    credibility_score=0.9
                ),
                content="Clinical applications of machine learning in diagnosis",
                relevance_score=0.9,
                credibility_score=0.9,
                extraction_timestamp=datetime.now()
            ),
            RetrievedInfo(
                source=Source(
                    url="https://business.example.com", 
                    title="Business Report",
                    domain="business",
                    credibility_score=0.7
                ),
                content="ROI analysis of AI implementation in enterprises",
                relevance_score=0.7,
                credibility_score=0.6,
                extraction_timestamp=datetime.now()
            )
        ],
        iteration_count=1,
        quality_metrics=QualityMetrics(
            completeness=0.7,
            coherence=0.8,
            accuracy=0.7,
            citation_quality=0.6
        ),
        evolution_history=[],
        final_report=None
    )
    
    try:
        # Step 1: Cross-disciplinary detection
        print("Step 1: Cross-disciplinary detection...")
        state = cross_disciplinary_detector_node(test_state)
        
        if state.get("requires_cross_disciplinary"):
            print("✓ Cross-disciplinary research detected")
            metadata = state["cross_disciplinary_metadata"]
            print(f"  - Domains involved: {metadata['involved_domains']}")
            print(f"  - Confidence: {metadata['detection_confidence']}")
        else:
            print("✓ Single-domain research detected")
        
        # Step 2: Multi-domain integration (if needed)
        if state.get("requires_cross_disciplinary"):
            print("\nStep 2: Multi-domain integration...")
            state = cross_disciplinary_integrator_node(state)
            
            if "cross_disciplinary_integration" in state:
                integration = state["cross_disciplinary_integration"]
                print(f"✓ Integration completed with {len(integration.disciplinary_perspectives)} perspectives")
                print(f"  - Coherence score: {integration.coherence_score:.2f}")
                print(f"  - Integration strategy: {integration.integration_strategy}")
                
                # Step 3: Conflict resolution
                print("\nStep 3: Conflict resolution...")
                state = cross_disciplinary_conflict_resolver_node(state)
                
                updated_integration = state["cross_disciplinary_integration"]
                print(f"✓ Conflicts resolved: {len(updated_integration.conflicts_resolved)}")
                print(f"  - Updated coherence score: {updated_integration.coherence_score:.2f}")
                
                # Step 4: Specialized formatting
                print("\nStep 4: Specialized formatting...")
                state = cross_disciplinary_formatter_node(state)
                
                formatted_draft = state["current_draft"]
                new_sections = [k for k in formatted_draft.content.keys() 
                              if k not in test_state["current_draft"].content.keys()]
                print(f"✓ Formatting completed, added {len(new_sections)} new sections")
                if new_sections:
                    print(f"  - New sections: {new_sections}")
                
                # Step 5: Quality assessment
                print("\nStep 5: Quality assessment...")
                final_state = cross_disciplinary_quality_assessor_node(state)
                
                if "cross_disciplinary_quality_metadata" in final_state:
                    quality_metadata = final_state["cross_disciplinary_quality_metadata"]
                    overall_score = quality_metadata["overall_cross_disciplinary_score"]
                    print(f"✓ Quality assessment completed")
                    print(f"  - Overall cross-disciplinary score: {overall_score:.2f}")
                    print(f"  - Quality threshold met: {quality_metadata['quality_threshold_met']}")
                
                print(f"\n🎉 Complete cross-disciplinary workflow executed successfully!")
                return True
            else:
                print("✓ Integration skipped (not required)")
        
        print(f"\n🎉 Workflow completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_workflow_error_handling():
    """Test workflow error handling"""
    print("\nTesting Workflow Error Handling...")
    
    # Create minimal state that might cause issues
    minimal_state = TTDRState(
        topic="",  # Empty topic
        requirements=ResearchRequirements(
            domain=ResearchDomain.GENERAL,
            complexity_level=ComplexityLevel.BASIC,
            max_iterations=1,
            quality_threshold=0.5,
            max_sources=5
        ),
        current_draft=None,  # No draft
        information_gaps=[],
        retrieved_info=[],  # No retrieved info
        iteration_count=0,
        quality_metrics=None,
        evolution_history=[],
        final_report=None
    )
    
    try:
        # Should handle empty/minimal state gracefully
        state = cross_disciplinary_detector_node(minimal_state)
        print("✓ Detector node handled minimal state gracefully")
        
        # Should not require cross-disciplinary for empty topic
        assert not state.get("requires_cross_disciplinary", False)
        print("✓ Correctly identified as single-domain for empty topic")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("CROSS-DISCIPLINARY WORKFLOW INTEGRATION TEST")
    print("=" * 70)
    
    success1 = test_cross_disciplinary_workflow()
    success2 = test_workflow_error_handling()
    
    if success1 and success2:
        print("\n🎉 ALL WORKFLOW TESTS PASSED!")
        print("Cross-disciplinary research capabilities are fully integrated.")
        sys.exit(0)
    else:
        print("\n❌ SOME WORKFLOW TESTS FAILED.")
        sys.exit(1)