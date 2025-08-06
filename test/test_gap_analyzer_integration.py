#!/usr/bin/env python3

import sys
import os
sys.path.append('backend')

from backend.workflow.graph import gap_analyzer_node
from backend.models.core import (
    TTDRState, Draft, ResearchRequirements, ResearchDomain, ComplexityLevel, DraftMetadata
)
from backend.models.research_structure import (
    EnhancedResearchStructure, EnhancedSection, ContentPlaceholder,
    ResearchSectionType, ContentPlaceholderType, Priority
)

# Create sample draft manually
section1 = EnhancedSection(
    id="intro",
    title="Introduction",
    estimated_length=500,
    section_type=ResearchSectionType.INTRODUCTION,
    content_placeholders=[
        ContentPlaceholder(
            id="intro_placeholder",
            placeholder_type=ContentPlaceholderType.INTRODUCTION,
            title="Topic Overview",
            description="Overview of the research topic",
            estimated_word_count=200,
            priority=Priority.HIGH
        )
    ]
)

section2 = EnhancedSection(
    id="analysis",
    title="Analysis",
    estimated_length=1000,
    section_type=ResearchSectionType.ANALYSIS,
    content_placeholders=[
        ContentPlaceholder(
            id="analysis_placeholder",
            placeholder_type=ContentPlaceholderType.ANALYSIS,
            title="Main Analysis",
            description="Core analysis of the topic",
            estimated_word_count=500,
            priority=Priority.CRITICAL
        )
    ]
)

structure = EnhancedResearchStructure(
    sections=[section1, section2],
    estimated_length=1500,
    complexity_level=ComplexityLevel.INTERMEDIATE,
    domain=ResearchDomain.TECHNOLOGY
)

sample_draft = Draft(
    id="test_draft",
    topic="Artificial Intelligence in Healthcare",
    structure=structure,
    content={
        "intro": "Brief introduction to AI in healthcare",
        "analysis": "Limited analysis content"
    },
    metadata=DraftMetadata(),
    quality_score=0.6,
    iteration=1
)

# Create TTD-DR state
state = TTDRState(
    topic="Artificial Intelligence in Healthcare",
    requirements=ResearchRequirements(
        domain=ResearchDomain.TECHNOLOGY,
        complexity_level=ComplexityLevel.INTERMEDIATE,
        max_iterations=5,
        quality_threshold=0.8
    ),
    current_draft=sample_draft,
    information_gaps=[],
    retrieved_info=[],
    iteration_count=1,
    quality_metrics=None,
    evolution_history=[],
    final_report=None,
    error_log=[]
)

print("Testing gap analyzer node integration...")
print(f"Initial state - gaps: {len(state['information_gaps'])}")

# Execute gap analyzer node
try:
    updated_state = gap_analyzer_node(state)
    
    print(f"After gap analysis - gaps: {len(updated_state['information_gaps'])}")
    
    for i, gap in enumerate(updated_state['information_gaps']):
        print(f"Gap {i+1}:")
        print(f"  Type: {gap.gap_type.value}")
        print(f"  Section: {gap.section_id}")
        print(f"  Priority: {gap.priority.value}")
        print(f"  Description: {gap.description}")
        print(f"  Search queries: {len(gap.search_queries)}")
        if gap.search_queries:
            for j, query in enumerate(gap.search_queries[:2]):  # Show first 2 queries
                print(f"    Query {j+1}: {query.query}")
        print()
    
    print("Gap analyzer node integration test completed successfully!")
    
except Exception as e:
    print(f"Error during gap analysis: {e}")
    import traceback
    traceback.print_exc()