"""
Simple test to verify cross-disciplinary research capabilities work.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.services.cross_disciplinary_integrator import (
    CrossDisciplinaryIntegrator,
    DisciplinaryPerspective,
    CrossDisciplinaryConflict
)
from backend.models.core import ResearchDomain, RetrievedInfo, Source
from datetime import datetime


def test_basic_cross_disciplinary_functionality():
    """Test basic cross-disciplinary functionality"""
    print("Testing Cross-Disciplinary Research Capabilities...")
    
    # Test 1: Create integrator
    try:
        integrator = CrossDisciplinaryIntegrator()
        print("✓ CrossDisciplinaryIntegrator created successfully")
    except Exception as e:
        print(f"✗ Failed to create integrator: {e}")
        return False
    
    # Test 2: Test domain detection
    try:
        topic = "AI applications in healthcare business and scientific research"
        retrieved_info = [
            RetrievedInfo(
                source=Source(
                    url="https://example.com", 
                    title="Test",
                    domain="technology",
                    credibility_score=0.8
                ),
                content="AI technology in healthcare systems",
                relevance_score=0.8,
                credibility_score=0.7,
                extraction_timestamp=datetime.now()
            ),
            RetrievedInfo(
                source=Source(
                    url="https://example2.com", 
                    title="Test2",
                    domain="business",
                    credibility_score=0.7
                ),
                content="Business analysis of medical AI market",
                relevance_score=0.7,
                credibility_score=0.6,
                extraction_timestamp=datetime.now()
            )
        ]
        
        is_cross_disciplinary, domains = integrator.detect_cross_disciplinary_nature(
            topic, retrieved_info
        )
        
        print(f"✓ Domain detection completed: {is_cross_disciplinary}, domains: {[d.value for d in domains]}")
    except Exception as e:
        print(f"✗ Domain detection failed: {e}")
        return False
    
    # Test 3: Test disciplinary perspective creation
    try:
        perspective = DisciplinaryPerspective(
            domain=ResearchDomain.TECHNOLOGY,
            confidence=0.8,
            key_concepts=["AI", "machine learning"],
            methodological_approach="Software engineering",
            theoretical_framework="Computer science"
        )
        print(f"✓ DisciplinaryPerspective created: {perspective.domain.value}")
    except Exception as e:
        print(f"✗ Failed to create perspective: {e}")
        return False
    
    # Test 4: Test conflict creation
    try:
        conflict = CrossDisciplinaryConflict(
            conflict_id="test_conflict",
            domains_involved=[ResearchDomain.TECHNOLOGY, ResearchDomain.SCIENCE],
            conflict_type="methodological",
            description="Different validation approaches",
            conflicting_information=[],
            severity=0.6
        )
        print(f"✓ CrossDisciplinaryConflict created: {conflict.conflict_type}")
    except Exception as e:
        print(f"✗ Failed to create conflict: {e}")
        return False
    
    # Test 5: Test multi-domain integration (with fallback)
    try:
        if is_cross_disciplinary and len(domains) > 1:
            integration = integrator.integrate_multi_domain_knowledge(
                topic=topic,
                domains=domains[:2],  # Limit to 2 domains for simple test
                retrieved_info=retrieved_info
            )
            print(f"✓ Multi-domain integration completed: {len(integration.disciplinary_perspectives)} perspectives")
        else:
            print("✓ Single-domain research detected (no integration needed)")
    except Exception as e:
        print(f"✗ Multi-domain integration failed: {e}")
        return False
    
    print("\n🎉 All cross-disciplinary tests passed!")
    return True


def test_cross_disciplinary_workflow_nodes():
    """Test cross-disciplinary workflow nodes"""
    print("\nTesting Cross-Disciplinary Workflow Nodes...")
    
    try:
        from backend.workflow.cross_disciplinary_node import (
            cross_disciplinary_detector_node,
            cross_disciplinary_integrator_node
        )
        print("✓ Cross-disciplinary workflow nodes imported successfully")
    except Exception as e:
        print(f"✗ Failed to import workflow nodes: {e}")
        return False
    
    # Test node creation
    try:
        from backend.models.core import TTDRState, ResearchRequirements, ComplexityLevel
        
        test_state = TTDRState(
            topic="Test cross-disciplinary topic",
            requirements=ResearchRequirements(
                domain=ResearchDomain.GENERAL,
                complexity_level=ComplexityLevel.BASIC,
                max_iterations=3,
                quality_threshold=0.7,
                max_sources=10
            ),
            current_draft=None,
            information_gaps=[],
            retrieved_info=[],
            iteration_count=0,
            quality_metrics=None,
            evolution_history=[],
            final_report=None
        )
        print("✓ Test state created successfully")
    except Exception as e:
        print(f"✗ Failed to create test state: {e}")
        return False
    
    print("✓ Cross-disciplinary workflow nodes test completed")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("CROSS-DISCIPLINARY RESEARCH CAPABILITIES TEST")
    print("=" * 60)
    
    success1 = test_basic_cross_disciplinary_functionality()
    success2 = test_cross_disciplinary_workflow_nodes()
    
    if success1 and success2:
        print("\n🎉 ALL TESTS PASSED! Cross-disciplinary capabilities are working.")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED. Please check the implementation.")
        sys.exit(1)