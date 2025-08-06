"""
Kimi K2 Information Gap Analysis Service.
This service provides functionality to identify information gaps in research drafts
and generate search queries using Kimi K2 AI capabilities.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import uuid

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.core import (
    Draft, InformationGap, GapType, Priority, SearchQuery, ResearchDomain
)
from models.research_structure import EnhancedSection, ContentPlaceholder
from services.kimi_k2_client import KimiK2Client, KimiK2Error

logger = logging.getLogger(__name__)

class KimiK2InformationGapAnalyzer:
    """Analyzer for identifying information gaps using Kimi K2 AI"""
    
    def __init__(self, kimi_client: Optional[KimiK2Client] = None):
        self.kimi_client = kimi_client or KimiK2Client()
        self.gap_analysis_cache = {}  # Cache for repeated analysis
    
    async def identify_gaps(self, draft: Draft) -> List[InformationGap]:
        """
        Identify information gaps in the current draft using Kimi K2
        
        Args:
            draft: The research draft to analyze
            
        Returns:
            List of identified information gaps
        """
        try:
            logger.info(f"Starting gap analysis for draft: {draft.id}")
            
            # Analyze each section for gaps
            all_gaps = []
            
            for section in draft.structure.sections:
                section_content = draft.content.get(section.id, "")
                section_gaps = await self._analyze_section_gaps(
                    section, section_content, draft.topic, draft.structure.domain
                )
                all_gaps.extend(section_gaps)
            
            # Analyze overall draft coherence and completeness
            overall_gaps = await self._analyze_overall_gaps(draft)
            all_gaps.extend(overall_gaps)
            
            # Prioritize gaps using Kimi K2
            prioritized_gaps = await self._prioritize_gaps(all_gaps, draft)
            
            logger.info(f"Identified {len(prioritized_gaps)} information gaps")
            return prioritized_gaps
            
        except Exception as e:
            logger.error(f"Failed to identify gaps: {e}")
            return self._generate_fallback_gaps(draft)
    
    async def _analyze_section_gaps(
        self,
        section: EnhancedSection,
        content: str,
        topic: str,
        domain: ResearchDomain
    ) -> List[InformationGap]:
        """Analyze gaps in a specific section"""
        
        prompt = self._build_section_gap_analysis_prompt(section, content, topic, domain)
        
        try:
            # Define schema for structured response
            schema = {
                "type": "object",
                "properties": {
                    "gaps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "gap_type": {
                                    "type": "string",
                                    "enum": ["content", "evidence", "citation", "analysis"]
                                },
                                "description": {"type": "string"},
                                "priority": {
                                    "type": "string",
                                    "enum": ["low", "medium", "high", "critical"]
                                },
                                "specific_needs": {"type": "array", "items": {"type": "string"}},
                                "suggested_sources": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["gap_type", "description", "priority"]
                        }
                    }
                }
            }
            
            response = await self.kimi_client.generate_structured_response(prompt, schema)
            
            # Convert response to InformationGap objects
            gaps = []
            for gap_data in response.get("gaps", []):
                gap = InformationGap(
                    id=str(uuid.uuid4()),
                    section_id=section.id,
                    gap_type=GapType(gap_data["gap_type"]),
                    description=gap_data["description"],
                    priority=Priority(gap_data["priority"]),
                    search_queries=[]  # Will be generated separately
                )
                
                # Store additional metadata for query generation
                gap.specific_needs = gap_data.get("specific_needs", [])
                gap.suggested_sources = gap_data.get("suggested_sources", [])
                
                gaps.append(gap)
            
            return gaps
            
        except Exception as e:
            logger.error(f"Failed to analyze section gaps: {e}")
            return self._generate_fallback_section_gaps(section, content)
    
    def _build_section_gap_analysis_prompt(
        self,
        section: EnhancedSection,
        content: str,
        topic: str,
        domain: ResearchDomain
    ) -> str:
        """Build prompt for section gap analysis"""
        
        return f"""
Analyze the following research section for information gaps:

Research Topic: {topic}
Domain: {domain.value}
Section Title: {section.title}
Section Type: {section.section_type if isinstance(section.section_type, str) else section.section_type.value}
Expected Length: {section.estimated_length} words

Current Content:
{content[:2000]}  # Limit content length

Content Placeholders Expected:
{self._format_placeholders(section.content_placeholders)}

Please identify information gaps in this section. For each gap, provide:

1. Gap Type: 
   - "content": Missing substantial content or information
   - "evidence": Lacks supporting evidence or examples
   - "citation": Missing citations or references
   - "analysis": Lacks analysis or interpretation

2. Description: Clear description of what's missing

3. Priority: 
   - "critical": Essential for section completeness
   - "high": Important for quality
   - "medium": Would improve the section
   - "low": Nice to have

4. Specific Needs: What specific information is needed

5. Suggested Sources: Types of sources that might help fill the gap

Focus on gaps that would significantly improve the section's quality, completeness, and credibility.

Return your analysis as a JSON object with the specified structure.
"""
    
    def _format_placeholders(self, placeholders: List[ContentPlaceholder]) -> str:
        """Format content placeholders for prompt"""
        if not placeholders:
            return "No specific placeholders defined"
        
        formatted = []
        for placeholder in placeholders:
            formatted.append(f"- {placeholder.title}: {placeholder.description}")
        
        return "\n".join(formatted)
    
    async def _analyze_overall_gaps(self, draft: Draft) -> List[InformationGap]:
        """Analyze overall draft for structural and coherence gaps"""
        
        prompt = self._build_overall_gap_analysis_prompt(draft)
        
        try:
            schema = {
                "type": "object",
                "properties": {
                    "structural_gaps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "priority": {"type": "string"},
                                "affected_sections": {"type": "array", "items": {"type": "string"}}
                            }
                        }
                    },
                    "coherence_gaps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "priority": {"type": "string"},
                                "section_connections": {"type": "array", "items": {"type": "string"}}
                            }
                        }
                    }
                }
            }
            
            response = await self.kimi_client.generate_structured_response(prompt, schema)
            
            gaps = []
            
            # Process structural gaps
            for gap_data in response.get("structural_gaps", []):
                gap = InformationGap(
                    id=str(uuid.uuid4()),
                    section_id="overall_structure",
                    gap_type=GapType.CONTENT,
                    description=f"Structural gap: {gap_data['description']}",
                    priority=Priority(gap_data.get("priority", "medium")),
                    search_queries=[]
                )
                gap.affected_sections = gap_data.get("affected_sections", [])
                gaps.append(gap)
            
            # Process coherence gaps
            for gap_data in response.get("coherence_gaps", []):
                gap = InformationGap(
                    id=str(uuid.uuid4()),
                    section_id="overall_coherence",
                    gap_type=GapType.ANALYSIS,
                    description=f"Coherence gap: {gap_data['description']}",
                    priority=Priority(gap_data.get("priority", "medium")),
                    search_queries=[]
                )
                gap.section_connections = gap_data.get("section_connections", [])
                gaps.append(gap)
            
            return gaps
            
        except Exception as e:
            logger.error(f"Failed to analyze overall gaps: {e}")
            return []
    
    def _build_overall_gap_analysis_prompt(self, draft: Draft) -> str:
        """Build prompt for overall draft gap analysis"""
        
        # Prepare section summaries
        section_summaries = []
        for section in draft.structure.sections:
            content = draft.content.get(section.id, "[No content]")
            summary = content[:200] + "..." if len(content) > 200 else content
            section_summaries.append(f"- {section.title}: {summary}")
        
        return f"""
Analyze the overall structure and coherence of this research draft:

Research Topic: {draft.topic}
Domain: {draft.structure.domain.value}
Target Length: {draft.structure.estimated_length} words
Current Iteration: {draft.iteration}

Section Overview:
{chr(10).join(section_summaries)}

Please identify:

1. Structural Gaps:
   - Missing sections that are typical for this domain
   - Sections that are too brief or underdeveloped
   - Logical ordering issues
   - Missing connections between sections

2. Coherence Gaps:
   - Inconsistent arguments or themes
   - Missing transitions between sections
   - Contradictory information
   - Unclear relationships between ideas

For each gap, specify:
- Description: What's missing or problematic
- Priority: critical, high, medium, or low
- Affected sections or connections

Focus on gaps that impact the overall quality and readability of the research.
"""
    
    async def _prioritize_gaps(
        self,
        gaps: List[InformationGap],
        draft: Draft
    ) -> List[InformationGap]:
        """Prioritize gaps using Kimi K2 intelligence"""
        
        if not gaps:
            return gaps
        
        prompt = self._build_gap_prioritization_prompt(gaps, draft)
        
        try:
            schema = {
                "type": "object",
                "properties": {
                    "prioritized_gaps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "gap_id": {"type": "string"},
                                "priority": {"type": "string"},
                                "reasoning": {"type": "string"},
                                "impact_score": {"type": "number", "minimum": 0, "maximum": 1}
                            }
                        }
                    }
                }
            }
            
            response = await self.kimi_client.generate_structured_response(prompt, schema)
            
            # Update gap priorities based on Kimi K2 analysis
            priority_map = {}
            impact_map = {}
            
            for gap_data in response.get("prioritized_gaps", []):
                gap_id = gap_data["gap_id"]
                priority_map[gap_id] = Priority(gap_data["priority"])
                impact_map[gap_id] = gap_data.get("impact_score", 0.5)
            
            # Update gaps with new priorities
            for gap in gaps:
                if gap.id in priority_map:
                    gap.priority = priority_map[gap.id]
                    gap.impact_score = impact_map[gap.id]
            
            # Sort by priority and impact score
            priority_order = {
                Priority.CRITICAL: 4,
                Priority.HIGH: 3,
                Priority.MEDIUM: 2,
                Priority.LOW: 1
            }
            
            gaps.sort(
                key=lambda g: (
                    priority_order.get(g.priority, 0),
                    getattr(g, 'impact_score', 0.5)
                ),
                reverse=True
            )
            
            return gaps
            
        except Exception as e:
            logger.error(f"Failed to prioritize gaps: {e}")
            return gaps  # Return original gaps if prioritization fails
    
    def _build_gap_prioritization_prompt(
        self,
        gaps: List[InformationGap],
        draft: Draft
    ) -> str:
        """Build prompt for gap prioritization"""
        
        gap_descriptions = []
        for i, gap in enumerate(gaps):
            gap_descriptions.append(
                f"Gap {i+1} (ID: {gap.id}):\n"
                f"  Type: {gap.gap_type.value}\n"
                f"  Section: {gap.section_id}\n"
                f"  Description: {gap.description}\n"
                f"  Current Priority: {gap.priority.value}"
            )
        
        return f"""
Prioritize the following information gaps for a research draft:

Research Topic: {draft.topic}
Domain: {draft.structure.domain.value}
Current Quality Score: {draft.quality_score}
Iteration: {draft.iteration}

Identified Gaps:
{chr(10).join(gap_descriptions)}

Please prioritize these gaps considering:
1. Impact on overall research quality
2. Importance for the specific domain
3. Effort required to address the gap
4. Dependencies between gaps

For each gap, provide:
- gap_id: The ID of the gap
- priority: critical, high, medium, or low
- reasoning: Why this priority was assigned
- impact_score: 0.0 to 1.0 indicating potential impact on quality

Focus on gaps that will provide the most significant improvement to the research.
"""
    
    def _generate_fallback_gaps(self, draft: Draft) -> List[InformationGap]:
        """Generate fallback gaps when Kimi K2 is unavailable"""
        
        logger.warning("Using fallback gap identification")
        
        gaps = []
        
        for section in draft.structure.sections:
            content = draft.content.get(section.id, "")
            
            # Simple heuristic-based gap identification
            if len(content) < section.estimated_length * 0.5:
                gap = InformationGap(
                    id=str(uuid.uuid4()),
                    section_id=section.id,
                    gap_type=GapType.CONTENT,
                    description=f"Section '{section.title}' appears underdeveloped. Current length: {len(content)} characters, expected: ~{section.estimated_length * 5} characters.",
                    priority=Priority.HIGH if len(content) < section.estimated_length * 0.2 else Priority.MEDIUM,
                    search_queries=[]
                )
                gaps.append(gap)
            
            # Check for missing citations (simple heuristic) - check all sections with content
            if len(content) > 20:  # Even lower threshold for testing
                has_citations = any(marker in content for marker in ["http", "[", "(", "doi:", "www.", ".com", ".org"])
                if not has_citations:
                    gap = InformationGap(
                        id=str(uuid.uuid4()),
                        section_id=section.id,
                        gap_type=GapType.CITATION,
                        description=f"Section '{section.title}' lacks citations or references.",
                        priority=Priority.MEDIUM,
                        search_queries=[]
                    )
                    gaps.append(gap)
        
        return gaps
    
    def _generate_fallback_section_gaps(
        self,
        section: EnhancedSection,
        content: str
    ) -> List[InformationGap]:
        """Generate fallback gaps for a specific section"""
        
        gaps = []
        
        # Check content length
        if len(content) < section.estimated_length * 0.5:
            gap = InformationGap(
                id=str(uuid.uuid4()),
                section_id=section.id,
                gap_type=GapType.CONTENT,
                description=f"Section needs more detailed content",
                priority=Priority.HIGH,
                search_queries=[]
            )
            gaps.append(gap)
        
        # Check for placeholders that haven't been filled
        for placeholder in section.content_placeholders:
            if placeholder.title.lower() not in content.lower():
                gap = InformationGap(
                    id=str(uuid.uuid4()),
                    section_id=section.id,
                    gap_type=GapType.CONTENT,
                    description=f"Missing content for: {placeholder.description}",
                    priority=placeholder.priority,
                    search_queries=[]
                )
                gaps.append(gap)
        
        return gaps

class KimiK2SearchQueryGenerator:
    """Generator for search queries using Kimi K2 AI"""
    
    def __init__(self, kimi_client: Optional[KimiK2Client] = None):
        self.kimi_client = kimi_client or KimiK2Client()
    
    async def generate_search_queries(
        self,
        gap: InformationGap,
        topic: str,
        domain: ResearchDomain,
        max_queries: int = 3
    ) -> List[SearchQuery]:
        """
        Generate search queries for an information gap using Kimi K2
        
        Args:
            gap: The information gap to generate queries for
            topic: The research topic
            domain: The research domain
            max_queries: Maximum number of queries to generate
            
        Returns:
            List of optimized search queries
        """
        try:
            prompt = self._build_query_generation_prompt(gap, topic, domain, max_queries)
            
            schema = {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "priority": {"type": "string"},
                                "expected_results": {"type": "integer"},
                                "search_strategy": {"type": "string"}
                            },
                            "required": ["query", "priority"]
                        }
                    }
                }
            }
            
            response = await self.kimi_client.generate_structured_response(prompt, schema)
            
            # Convert to SearchQuery objects
            search_queries = []
            for query_data in response.get("queries", []):
                query = SearchQuery(
                    query=query_data["query"],
                    priority=Priority(query_data.get("priority", "medium")),
                    expected_results=query_data.get("expected_results", 10)
                )
                query.search_strategy = query_data.get("search_strategy", "general")
                search_queries.append(query)
            
            # Validate and optimize queries
            optimized_queries = await self._optimize_queries(search_queries, domain)
            
            logger.info(f"Generated {len(optimized_queries)} search queries for gap: {gap.id}")
            return optimized_queries
            
        except Exception as e:
            logger.error(f"Failed to generate search queries: {e}")
            return self._generate_fallback_queries(gap, topic)
    
    def _build_query_generation_prompt(
        self,
        gap: InformationGap,
        topic: str,
        domain: ResearchDomain,
        max_queries: int
    ) -> str:
        """Build prompt for search query generation"""
        
        return f"""
Generate effective search queries to fill the following information gap:

Research Topic: {topic}
Domain: {domain.value}
Gap Type: {gap.gap_type.value}
Gap Description: {gap.description}
Section: {gap.section_id}
Priority: {gap.priority.value}

Additional Context:
{getattr(gap, 'specific_needs', [])}
{getattr(gap, 'suggested_sources', [])}

Generate up to {max_queries} search queries that would help find relevant information to fill this gap.

For each query, consider:
1. Keywords that are likely to return relevant results
2. Domain-specific terminology
3. Different perspectives or approaches to the topic
4. Authoritative sources for this domain
5. Google Search optimization (use natural language, avoid overly complex boolean operators)

For each query, provide:
- query: The actual search query string (optimized for Google Search)
- priority: critical, high, medium, or low (based on how likely it is to find relevant information)
- expected_results: Number of results to retrieve (5-20)
- search_strategy: Brief description of what this query is trying to find

Focus on queries that are:
- Specific enough to return relevant results
- Broad enough to capture different sources
- Optimized for the research domain
- Likely to find authoritative information
"""
    
    async def _optimize_queries(
        self,
        queries: List[SearchQuery],
        domain: ResearchDomain
    ) -> List[SearchQuery]:
        """Optimize search queries for better results"""
        
        # Simple optimization - could be enhanced with Kimi K2
        optimized = []
        
        for query in queries:
            # Add domain-specific terms if not present
            domain_terms = {
                ResearchDomain.TECHNOLOGY: ["technology", "technical", "innovation"],
                ResearchDomain.SCIENCE: ["research", "study", "scientific"],
                ResearchDomain.BUSINESS: ["business", "market", "industry"],
                ResearchDomain.ACADEMIC: ["academic", "scholarly", "peer-reviewed"]
            }
            
            query_text = query.query.lower()
            domain_keywords = domain_terms.get(domain, [])
            
            # Add domain context if not present
            if not any(term in query_text for term in domain_keywords):
                if domain != ResearchDomain.GENERAL:
                    query.query = f"{query.query} {domain.value}"
            
            optimized.append(query)
        
        return optimized
    
    def _generate_fallback_queries(
        self,
        gap: InformationGap,
        topic: str
    ) -> List[SearchQuery]:
        """Generate fallback queries when Kimi K2 is unavailable"""
        
        # Simple fallback query generation
        base_query = f"{topic} {gap.section_id.replace('_', ' ')}"
        
        queries = [
            SearchQuery(
                query=base_query,
                priority=Priority.MEDIUM,
                expected_results=10
            ),
            SearchQuery(
                query=f"{topic} {gap.gap_type.value}",
                priority=Priority.MEDIUM,
                expected_results=10
            )
        ]
        
        # Add specific query based on gap type
        if gap.gap_type == GapType.EVIDENCE:
            queries.append(SearchQuery(
                query=f"{topic} evidence examples case study",
                priority=Priority.HIGH,
                expected_results=15
            ))
        elif gap.gap_type == GapType.CITATION:
            queries.append(SearchQuery(
                query=f"{topic} research papers academic sources",
                priority=Priority.HIGH,
                expected_results=15
            ))
        
        return queries

    async def validate_and_refine_queries(
        self,
        queries: List[SearchQuery],
        gap: InformationGap,
        topic: str
    ) -> List[SearchQuery]:
        """Validate and refine search queries using Kimi K2"""
        
        try:
            prompt = f"""
Evaluate and refine the following search queries for effectiveness:

Research Topic: {topic}
Information Gap: {gap.description}
Gap Type: {gap.gap_type.value}

Current Queries:
{chr(10).join([f"- {q.query}" for q in queries])}

Please evaluate each query and suggest improvements:
1. Are the queries likely to return relevant results?
2. Are they optimized for Google Search?
3. Do they cover different aspects of the information gap?
4. Are there better keyword combinations?

Provide refined versions of the queries or suggest new ones if needed.
Return as JSON with improved queries.
"""
            
            schema = {
                "type": "object",
                "properties": {
                    "refined_queries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "original_query": {"type": "string"},
                                "refined_query": {"type": "string"},
                                "improvement_reason": {"type": "string"},
                                "effectiveness_score": {"type": "number"}
                            }
                        }
                    }
                }
            }
            
            response = await self.kimi_client.generate_structured_response(prompt, schema)
            
            # Apply refinements
            refined_queries = []
            refinements = {r["original_query"]: r for r in response.get("refined_queries", [])}
            
            for query in queries:
                if query.query in refinements:
                    refinement = refinements[query.query]
                    query.query = refinement["refined_query"]
                    query.effectiveness_score = refinement.get("effectiveness_score", 0.7)
                
                refined_queries.append(query)
            
            return refined_queries
            
        except Exception as e:
            logger.error(f"Failed to refine queries: {e}")
            return queries  # Return original queries if refinement fails