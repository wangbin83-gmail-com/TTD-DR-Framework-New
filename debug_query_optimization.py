#!/usr/bin/env python3

import sys
import os
sys.path.append('backend')
import asyncio

from backend.services.kimi_k2_search_query_generator import KimiK2SearchQueryGenerator
from backend.models.core import InformationGap, GapType, Priority, SearchQuery, ResearchDomain

# Create test data
sample_gap = InformationGap(
    id="test_gap",
    section_id="analysis",
    gap_type=GapType.EVIDENCE,
    description="Missing case studies for AI implementation",
    priority=Priority.HIGH
)

initial_queries = [
    SearchQuery(query="AI healthcare", priority=Priority.MEDIUM),
    SearchQuery(query="machine learning medical", priority=Priority.HIGH)
]

async def debug_optimization():
    generator = KimiK2SearchQueryGenerator()
    
    print("Initial queries:")
    for q in initial_queries:
        print(f"  {q.query}")
    
    print(f"\nTemplate keys available: {list(generator.query_templates.keys())}")
    
    # Check template lookup
    template_key = f"{ResearchDomain.TECHNOLOGY.value}_{GapType.EVIDENCE.value}"
    print(f"Looking for template key: {template_key}")
    
    if template_key not in generator.query_templates:
        template_key = GapType.EVIDENCE.value
        print(f"Fallback to gap type key: {template_key}")
    
    template = generator.query_templates.get(template_key)
    print(f"Found template: {template is not None}")
    
    if template:
        print(f"Template patterns: {[p['pattern'] for p in template.query_patterns]}")
    
    # Execute optimization
    optimized_queries = await generator._optimize_queries_with_templates(
        initial_queries, sample_gap, "AI in Healthcare", ResearchDomain.TECHNOLOGY
    )
    
    print(f"\nOptimized queries ({len(optimized_queries)}):")
    for q in optimized_queries:
        print(f"  {q.query} (strategy: {getattr(q, 'search_strategy', 'none')})")

if __name__ == "__main__":
    asyncio.run(debug_optimization())