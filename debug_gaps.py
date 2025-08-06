#!/usr/bin/env python3

import sys
import os
sys.path.append('backend')

from services.kimi_k2_gap_analyzer import KimiK2InformationGapAnalyzer
from backend.models.core import (
    Draft, ResearchDomain, ComplexityLevel, DraftMetadata
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

gap_analyzer = KimiK2InformationGapAnalyzer()

print('Draft content:')
for section_id, content in sample_draft.content.items():
    print(f'  {section_id}: "{content}" (len={len(content)})')

gaps = gap_analyzer._generate_fallback_gaps(sample_draft)
print(f'\nGenerated {len(gaps)} gaps:')
for gap in gaps:
    print(f'  {gap.gap_type.value}: {gap.description}')

# Check citation detection logic
print('\nCitation detection:')
for section_id, content in sample_draft.content.items():
    has_citations = any(marker in content for marker in ["http", "[", "(", "doi:", "www.", ".com", ".org"])
    print(f'  {section_id}: has_citations={has_citations}, content="{content}"')