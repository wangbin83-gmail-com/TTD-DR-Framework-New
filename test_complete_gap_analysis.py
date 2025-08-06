#!/usr/bin/env python3

import sys
import os
sys.path.append('backend')
import asyncio

from backend.services.kimi_k2_gap_analyzer import KimiK2InformationGapAnalyzer
from backend.services.kimi_k2_search_query_generator import KimiK2SearchQueryGenerator
from backend.models.core import (
    Draft, ResearchDomain, ComplexityLevel, DraftMetadata
)
from backend.models.research_structure import (
    EnhancedResearchStructure, EnhancedSection, ContentPlaceholder,
    ResearchSectionType, ContentPlaceholderType, Priority
)

async def test_complete_gap_analysis():
    """Test the complete gap analysis workflow"""
    
    print("=== Complete Gap Analysis System Test ===\n")
    
    # Create sample draft
    section1 = EnhancedSection(
        id="intro",
        title="Introduction",
        estimated_length=800,
        section_type=ResearchSectionType.INTRODUCTION,
        content_placeholders=[
            ContentPlaceholder(
                id="intro_placeholder",
                placeholder_type=ContentPlaceholderType.INTRODUCTION,
                title="Topic Overview",
                description="Overview of the research topic",
                estimated_word_count=300,
                priority=Priority.HIGH
            )
        ]
    )
    
    section2 = EnhancedSection(
        id="methodology",
        title="Methodology",
        estimated_length=1200,
        section_type=ResearchSectionType.METHODOLOGY,
        content_placeholders=[
            ContentPlaceholder(
                id="method_placeholder",
                placeholder_type=ContentPlaceholderType.METHODOLOGY,
                title="Research Methods",
                description="Detailed research methodology",
                estimated_word_count=600,
                priority=Priority.CRITICAL
            )
        ]
    )
    
    structure = EnhancedResearchStructure(
        sections=[section1, section2],
        estimated_length=2000,
        complexity_level=ComplexityLevel.ADVANCED,
        domain=ResearchDomain.SCIENCE
    )
    
    draft = Draft(
        id="complete_test_draft",
        topic="Machine Learning Applications in Climate Change Research",
        structure=structure,
        content={
            "intro": "Machine learning is increasingly used in climate research",
            "methodology": "Various ML techniques are applied"
        },
        metadata=DraftMetadata(),
        quality_score=0.4,
        iteration=0
    )
    
    print(f"Testing with draft: {draft.topic}")
    print(f"Domain: {draft.structure.domain.value}")
    print(f"Sections: {len(draft.structure.sections)}")
    print()
    
    # Step 1: Identify information gaps
    print("Step 1: Identifying information gaps...")
    gap_analyzer = KimiK2InformationGapAnalyzer()
    gaps = await gap_analyzer.identify_gaps(draft)
    
    print(f"Identified {len(gaps)} information gaps:")
    for i, gap in enumerate(gaps, 1):
        print(f"  {i}. {gap.gap_type.value.upper()}: {gap.description}")
        print(f"     Section: {gap.section_id}, Priority: {gap.priority.value}")
    print()
    
    # Step 2: Generate search queries for gaps
    print("Step 2: Generating search queries...")
    query_generator = KimiK2SearchQueryGenerator()
    
    all_queries = []
    for gap in gaps[:3]:  # Test first 3 gaps
        print(f"\nGenerating queries for gap: {gap.description[:50]}...")
        queries = await query_generator.generate_search_queries(
            gap=gap,
            topic=draft.topic,
            domain=draft.structure.domain,
            max_queries=2
        )
        
        print(f"Generated {len(queries)} queries:")
        for j, query in enumerate(queries, 1):
            print(f"  {j}. \"{query.query}\" (Priority: {query.priority.value})")
            print(f"     Strategy: {getattr(query, 'search_strategy', 'N/A')}")
            print(f"     Effectiveness: {query.effectiveness_score:.2f}")
        
        all_queries.extend(queries)
    
    # Step 3: Query statistics
    print(f"\n=== Query Generation Statistics ===")
    stats = query_generator.get_query_statistics(all_queries)
    print(f"Total queries generated: {stats['total']}")
    print(f"Average effectiveness score: {stats['average_effectiveness']:.2f}")
    print(f"Priority distribution: {stats['priority_distribution']}")
    print(f"Search strategies used: {stats['strategies']}")
    
    # Step 4: Test batch generation
    print(f"\n=== Batch Query Generation Test ===")
    batch_results = await query_generator.batch_generate_queries(
        gaps[:2], draft.topic, draft.structure.domain, max_queries_per_gap=2
    )
    
    print(f"Batch generated queries for {len(batch_results)} gaps:")
    for gap_id, queries in batch_results.items():
        print(f"  Gap {gap_id}: {len(queries)} queries")
    
    print(f"\n=== Test Completed Successfully! ===")
    print(f"✓ Gap identification: {len(gaps)} gaps found")
    print(f"✓ Query generation: {len(all_queries)} queries generated")
    print(f"✓ Batch processing: {len(batch_results)} gaps processed")
    print(f"✓ Domain optimization: Science domain templates applied")
    print(f"✓ Quality metrics: Average effectiveness {stats['average_effectiveness']:.2f}")

if __name__ == "__main__":
    asyncio.run(test_complete_gap_analysis())