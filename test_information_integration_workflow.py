"""
Integration test for information integration workflow.
Tests the complete information integration process with Kimi K2.
"""

import asyncio
from datetime import datetime

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from models.core import (
    TTDRState, Draft, RetrievedInfo, InformationGap, Section, 
    ResearchStructure, DraftMetadata, Source, GapType, Priority,
    ComplexityLevel, ResearchDomain, ResearchRequirements
)
from workflow.information_integrator_node import information_integrator_node

def test_information_integration_workflow():
    """Test the complete information integration workflow"""
    
    # Create test data
    section = Section(
        id="intro",
        title="Introduction to AI",
        content="Artificial Intelligence is a broad field.",
        estimated_length=200
    )
    
    structure = ResearchStructure(
        sections=[section],
        estimated_length=1000,
        complexity_level=ComplexityLevel.INTERMEDIATE,
        domain=ResearchDomain.TECHNOLOGY
    )
    
    draft = Draft(
        id="test_draft",
        topic="Artificial Intelligence Applications",
        structure=structure,
        content={"intro": "Artificial Intelligence is a broad field."},
        metadata=DraftMetadata(),
        quality_score=0.4,
        iteration=1
    )
    
    source = Source(
        url="https://example.com/ai-applications",
        title="AI Applications in Industry",
        domain="example.com",
        credibility_score=0.85
    )
    
    retrieved_info = RetrievedInfo(
        source=source,
        content="AI applications span across healthcare, finance, transportation, and manufacturing. Machine learning algorithms enable predictive analytics and automation.",
        relevance_score=0.9,
        credibility_score=0.85,
        gap_id="gap_1"
    )
    
    gap = InformationGap(
        id="gap_1",
        section_id="intro",
        gap_type=GapType.CONTENT,
        description="Need specific examples of AI applications",
        priority=Priority.HIGH
    )
    
    # Create initial state
    state = TTDRState(
        topic="Artificial Intelligence Applications",
        requirements=ResearchRequirements(
            domain=ResearchDomain.TECHNOLOGY,
            complexity_level=ComplexityLevel.INTERMEDIATE,
            quality_threshold=0.8
        ),
        current_draft=draft,
        information_gaps=[gap],
        retrieved_info=[retrieved_info],
        iteration_count=1,
        quality_metrics=None,
        evolution_history=[],
        final_report=None,
        error_log=[]
    )
    
    print("=== Information Integration Workflow Test ===")
    print(f"Initial draft content: {draft.content['intro']}")
    print(f"Retrieved info: {retrieved_info.content[:100]}...")
    print(f"Gap to address: {gap.description}")
    
    # Execute information integration
    print("\n--- Executing Information Integration ---")
    result_state = information_integrator_node(state)
    
    # Verify results
    print("\n--- Results ---")
    print(f"Integration successful: {result_state['current_draft'] is not None}")
    print(f"Iteration count: {result_state['iteration_count']}")
    
    if result_state['current_draft']:
        updated_content = result_state['current_draft'].content['intro']
        print(f"Updated content length: {len(updated_content)} chars")
        print(f"Content includes retrieved info: {'healthcare' in updated_content}")
        print(f"Content includes source attribution: {'example.com' in updated_content}")
        
        print(f"\nUpdated content preview:")
        print(f"{updated_content[:300]}...")
    
    # Check for errors
    if result_state.get('error_log'):
        print(f"\nErrors encountered: {result_state['error_log']}")
    
    # Check integration history if available
    if result_state.get('integration_history'):
        print(f"\nIntegration operations: {len(result_state['integration_history'])}")
    
    print("\n=== Test Complete ===")
    
    # Basic assertions
    assert result_state['current_draft'] is not None
    assert result_state['iteration_count'] > state['iteration_count']
    assert len(result_state['current_draft'].content['intro']) > len(draft.content['intro'])

if __name__ == "__main__":
    test_information_integration_workflow()