"""
Kimi K2 Search Query Generation Service.
This service provides specialized functionality for generating, optimizing,
and validating search queries using Kimi K2 AI capabilities.
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
    InformationGap, GapType, Priority, SearchQuery, ResearchDomain
)
from services.kimi_k2_client import KimiK2Client, KimiK2Error

logger = logging.getLogger(__name__)

class SearchQueryTemplate:
    """Template for generating domain-specific search queries"""
    
    def __init__(self, template_id: str, domain: ResearchDomain, gap_type: GapType):
        self.template_id = template_id
        self.domain = domain
        self.gap_type = gap_type
        self.query_patterns = []
        self.optimization_hints = []
        self.validation_criteria = []
    
    def add_pattern(self, pattern: str, priority: Priority = Priority.MEDIUM):
        """Add a query pattern to the template"""
        self.query_patterns.append({
            "pattern": pattern,
            "priority": priority,
            "variables": self._extract_variables(pattern)
        })
    
    def add_optimization_hint(self, hint: str):
        """Add an optimization hint for this template"""
        self.optimization_hints.append(hint)
    
    def add_validation_criterion(self, criterion: str):
        """Add a validation criterion for generated queries"""
        self.validation_criteria.append(criterion)
    
    def _extract_variables(self, pattern: str) -> List[str]:
        """Extract variables from a query pattern"""
        import re
        return re.findall(r'\{(\w+)\}', pattern)
    
    def generate_queries(self, variables: Dict[str, str], max_queries: int = 3) -> List[str]:
        """Generate queries from patterns with provided variables"""
        queries = []
        
        for pattern_info in self.query_patterns[:max_queries]:
            pattern = pattern_info["pattern"]
            try:
                query = pattern.format(**variables)
                queries.append(query)
            except KeyError as e:
                logger.warning(f"Missing variable {e} for pattern {pattern}")
        
        return queries

class KimiK2SearchQueryGenerator:
    """Advanced search query generator using Kimi K2 AI"""
    
    def __init__(self, kimi_client: Optional[KimiK2Client] = None):
        self.kimi_client = kimi_client or KimiK2Client()
        self.query_templates = self._initialize_templates()
        self.query_cache = {}  # Cache for repeated query generation
    
    def _initialize_templates(self) -> Dict[str, SearchQueryTemplate]:
        """Initialize domain and gap-type specific query templates"""
        templates = {}
        
        # Technology domain templates
        tech_content_template = SearchQueryTemplate("tech_content", ResearchDomain.TECHNOLOGY, GapType.CONTENT)
        tech_content_template.add_pattern("{topic} technical specifications", Priority.HIGH)
        tech_content_template.add_pattern("{topic} implementation guide", Priority.HIGH)
        tech_content_template.add_pattern("{topic} architecture overview", Priority.MEDIUM)
        tech_content_template.add_pattern("{topic} best practices", Priority.MEDIUM)
        tech_content_template.add_optimization_hint("Include technical terminology")
        tech_content_template.add_optimization_hint("Focus on implementation details")
        tech_content_template.add_validation_criterion("Contains technical keywords")
        templates["tech_content"] = tech_content_template
        
        tech_evidence_template = SearchQueryTemplate("tech_evidence", ResearchDomain.TECHNOLOGY, GapType.EVIDENCE)
        tech_evidence_template.add_pattern("{topic} case studies", Priority.HIGH)
        tech_evidence_template.add_pattern("{topic} success stories", Priority.HIGH)
        tech_evidence_template.add_pattern("{topic} benchmarks performance", Priority.MEDIUM)
        tech_evidence_template.add_pattern("{topic} real world examples", Priority.MEDIUM)
        tech_evidence_template.add_optimization_hint("Look for quantitative results")
        tech_evidence_template.add_optimization_hint("Include industry examples")
        templates["tech_evidence"] = tech_evidence_template
        
        # Science domain templates
        science_content_template = SearchQueryTemplate("science_content", ResearchDomain.SCIENCE, GapType.CONTENT)
        science_content_template.add_pattern("{topic} research methodology", Priority.HIGH)
        science_content_template.add_pattern("{topic} scientific principles", Priority.HIGH)
        science_content_template.add_pattern("{topic} theoretical framework", Priority.MEDIUM)
        science_content_template.add_pattern("{topic} experimental design", Priority.MEDIUM)
        science_content_template.add_optimization_hint("Include peer-reviewed sources")
        science_content_template.add_optimization_hint("Focus on scientific rigor")
        templates["science_content"] = science_content_template
        
        science_evidence_template = SearchQueryTemplate("science_evidence", ResearchDomain.SCIENCE, GapType.EVIDENCE)
        science_evidence_template.add_pattern("{topic} empirical studies", Priority.HIGH)
        science_evidence_template.add_pattern("{topic} experimental results", Priority.HIGH)
        science_evidence_template.add_pattern("{topic} meta analysis", Priority.MEDIUM)
        science_evidence_template.add_pattern("{topic} systematic review", Priority.MEDIUM)
        science_evidence_template.add_optimization_hint("Prioritize peer-reviewed research")
        science_evidence_template.add_optimization_hint("Include statistical significance")
        templates["science_evidence"] = science_evidence_template
        
        # Business domain templates
        business_content_template = SearchQueryTemplate("business_content", ResearchDomain.BUSINESS, GapType.CONTENT)
        business_content_template.add_pattern("{topic} market analysis", Priority.HIGH)
        business_content_template.add_pattern("{topic} business strategy", Priority.HIGH)
        business_content_template.add_pattern("{topic} competitive landscape", Priority.MEDIUM)
        business_content_template.add_pattern("{topic} industry trends", Priority.MEDIUM)
        business_content_template.add_optimization_hint("Include market data")
        business_content_template.add_optimization_hint("Focus on business impact")
        templates["business_content"] = business_content_template
        
        business_evidence_template = SearchQueryTemplate("business_evidence", ResearchDomain.BUSINESS, GapType.EVIDENCE)
        business_evidence_template.add_pattern("{topic} ROI case studies", Priority.HIGH)
        business_evidence_template.add_pattern("{topic} business outcomes", Priority.HIGH)
        business_evidence_template.add_pattern("{topic} market research data", Priority.MEDIUM)
        business_evidence_template.add_pattern("{topic} industry reports", Priority.MEDIUM)
        business_evidence_template.add_optimization_hint("Include financial metrics")
        business_evidence_template.add_optimization_hint("Focus on measurable outcomes")
        templates["business_evidence"] = business_evidence_template
        
        # Citation templates (domain-agnostic)
        citation_template = SearchQueryTemplate("citation", ResearchDomain.GENERAL, GapType.CITATION)
        citation_template.add_pattern("{topic} academic papers", Priority.HIGH)
        citation_template.add_pattern("{topic} peer reviewed research", Priority.HIGH)
        citation_template.add_pattern("{topic} scholarly articles", Priority.MEDIUM)
        citation_template.add_pattern("{topic} research publications", Priority.MEDIUM)
        citation_template.add_optimization_hint("Prioritize academic sources")
        citation_template.add_optimization_hint("Include publication dates")
        templates["citation"] = citation_template
        
        return templates
    
    async def generate_search_queries(
        self,
        gap: InformationGap,
        topic: str,
        domain: ResearchDomain,
        max_queries: int = 3
    ) -> List[SearchQuery]:
        """
        Generate optimized search queries for an information gap using Kimi K2
        
        Args:
            gap: The information gap to generate queries for
            topic: The research topic
            domain: The research domain
            max_queries: Maximum number of queries to generate
            
        Returns:
            List of optimized search queries
        """
        try:
            # Try Kimi K2 generation first
            kimi_queries = await self._generate_with_kimi_k2(gap, topic, domain, max_queries)
            
            # Optimize queries using templates and domain knowledge
            optimized_queries = await self._optimize_queries_with_templates(
                kimi_queries, gap, topic, domain
            )
            
            # Validate and refine queries
            validated_queries = await self._validate_and_refine_queries(
                optimized_queries, gap, topic, domain
            )
            
            logger.info(f"Generated {len(validated_queries)} optimized search queries for gap: {gap.id}")
            return validated_queries
            
        except Exception as e:
            logger.error(f"Failed to generate search queries: {e}")
            return self._generate_fallback_queries(gap, topic, domain, max_queries)
    
    async def _generate_with_kimi_k2(
        self,
        gap: InformationGap,
        topic: str,
        domain: ResearchDomain,
        max_queries: int
    ) -> List[SearchQuery]:
        """Generate queries using Kimi K2 AI"""
        
        prompt = self._build_kimi_query_generation_prompt(gap, topic, domain, max_queries)
        
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
                            "search_strategy": {"type": "string"},
                            "reasoning": {"type": "string"}
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
                expected_results=query_data.get("expected_results", 10),
                search_strategy=query_data.get("search_strategy", "general"),
                effectiveness_score=0.8  # Default high score for Kimi K2 generated queries
            )
            query.reasoning = query_data.get("reasoning", "")
            search_queries.append(query)
        
        return search_queries
    
    def _build_kimi_query_generation_prompt(
        self,
        gap: InformationGap,
        topic: str,
        domain: ResearchDomain,
        max_queries: int
    ) -> str:
        """Build comprehensive prompt for Kimi K2 query generation"""
        
        # Get template for this domain and gap type
        template_key = f"{domain.value}_{gap.gap_type.value}"
        if template_key not in self.query_templates:
            template_key = gap.gap_type.value  # Fallback to gap type only
        
        template = self.query_templates.get(template_key)
        optimization_hints = template.optimization_hints if template else []
        validation_criteria = template.validation_criteria if template else []
        
        return f"""
Generate effective search queries to address the following information gap:

Research Context:
- Topic: {topic}
- Domain: {domain.value}
- Gap Type: {gap.gap_type.value}
- Gap Description: {gap.description}
- Section: {gap.section_id}
- Priority: {gap.priority.value}

Additional Context:
- Specific Needs: {getattr(gap, 'specific_needs', [])}
- Suggested Sources: {getattr(gap, 'suggested_sources', [])}

Query Generation Guidelines:
1. Generate {max_queries} search queries optimized for Google Search
2. Each query should be 3-8 words for optimal search performance
3. Use natural language rather than complex boolean operators
4. Include domain-specific terminology where appropriate
5. Consider different search angles and perspectives

Domain-Specific Optimization Hints:
{chr(10).join([f"- {hint}" for hint in optimization_hints]) if optimization_hints else "- Use general search optimization principles"}

Validation Criteria:
{chr(10).join([f"- {criterion}" for criterion in validation_criteria]) if validation_criteria else "- Ensure queries are relevant and specific"}

For each query, provide:
- query: The actual search query string (optimized for Google Search)
- priority: critical, high, medium, or low (based on likelihood of finding relevant information)
- expected_results: Number of results to retrieve (5-20)
- search_strategy: Brief description of what this query is trying to find
- reasoning: Why this query is effective for addressing the gap

Focus on queries that are:
- Specific enough to return relevant results
- Broad enough to capture authoritative sources
- Optimized for the research domain and gap type
- Likely to find high-quality, credible information
"""
    
    async def _optimize_queries_with_templates(
        self,
        queries: List[SearchQuery],
        gap: InformationGap,
        topic: str,
        domain: ResearchDomain
    ) -> List[SearchQuery]:
        """Optimize queries using domain-specific templates"""
        
        # Get appropriate template
        template_key = f"{domain.value}_{gap.gap_type.value}"
        if template_key not in self.query_templates:
            # Try with abbreviated domain names
            domain_abbrev = {
                ResearchDomain.TECHNOLOGY: "tech",
                ResearchDomain.SCIENCE: "science", 
                ResearchDomain.BUSINESS: "business",
                ResearchDomain.ACADEMIC: "academic",
                ResearchDomain.GENERAL: "general"
            }
            template_key = f"{domain_abbrev.get(domain, domain.value)}_{gap.gap_type.value}"
            
            if template_key not in self.query_templates:
                template_key = gap.gap_type.value
        
        template = self.query_templates.get(template_key)
        
        if not template:
            return queries  # No optimization available
        
        # Generate additional queries from templates
        template_variables = {
            "topic": topic,
            "domain": domain.value,
            "gap_description": gap.description[:50]  # Truncated for query use
        }
        
        template_queries = template.generate_queries(template_variables, max_queries=2)
        
        # Add template-generated queries
        for template_query in template_queries:
            if not any(q.query.lower() == template_query.lower() for q in queries):
                search_query = SearchQuery(
                    query=template_query,
                    priority=Priority.MEDIUM,
                    expected_results=10,
                    search_strategy="template_generated",
                    effectiveness_score=0.7
                )
                queries.append(search_query)
        
        # Apply domain-specific optimizations
        optimized_queries = []
        for query in queries:
            optimized_query = self._apply_domain_optimization(query, domain, gap.gap_type)
            optimized_queries.append(optimized_query)
        
        return optimized_queries
    
    def _apply_domain_optimization(
        self,
        query: SearchQuery,
        domain: ResearchDomain,
        gap_type: GapType
    ) -> SearchQuery:
        """Apply domain-specific optimizations to a query"""
        
        optimized_query = query.query
        
        # Domain-specific keyword additions
        domain_keywords = {
            ResearchDomain.TECHNOLOGY: ["technical", "implementation", "system"],
            ResearchDomain.SCIENCE: ["research", "study", "scientific"],
            ResearchDomain.BUSINESS: ["business", "market", "industry"],
            ResearchDomain.ACADEMIC: ["academic", "scholarly", "peer-reviewed"]
        }
        
        # Gap-type specific optimizations
        gap_keywords = {
            GapType.EVIDENCE: ["examples", "case studies", "data"],
            GapType.CITATION: ["papers", "research", "publications"],
            GapType.ANALYSIS: ["analysis", "evaluation", "assessment"],
            GapType.CONTENT: ["overview", "guide", "information"]
        }
        
        # Add domain keywords if not present
        domain_terms = domain_keywords.get(domain, [])
        if domain_terms and not any(term in optimized_query.lower() for term in domain_terms):
            optimized_query += f" {domain_terms[0]}"
        
        # Add gap-type keywords if not present
        gap_terms = gap_keywords.get(gap_type, [])
        if gap_terms and not any(term in optimized_query.lower() for term in gap_terms):
            optimized_query += f" {gap_terms[0]}"
        
        # Update query
        query.query = optimized_query
        return query
    
    async def _validate_and_refine_queries(
        self,
        queries: List[SearchQuery],
        gap: InformationGap,
        topic: str,
        domain: ResearchDomain
    ) -> List[SearchQuery]:
        """Validate and refine queries using Kimi K2"""
        
        try:
            prompt = self._build_validation_prompt(queries, gap, topic, domain)
            
            schema = {
                "type": "object",
                "properties": {
                    "validated_queries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "original_query": {"type": "string"},
                                "refined_query": {"type": "string"},
                                "effectiveness_score": {"type": "number"},
                                "validation_notes": {"type": "string"},
                                "keep_query": {"type": "boolean"}
                            }
                        }
                    }
                }
            }
            
            response = await self.kimi_client.generate_structured_response(prompt, schema)
            
            # Apply validation results
            validated_queries = []
            validation_results = {r["original_query"]: r for r in response.get("validated_queries", [])}
            
            for query in queries:
                if query.query in validation_results:
                    result = validation_results[query.query]
                    
                    if result.get("keep_query", True):
                        query.query = result.get("refined_query", query.query)
                        query.effectiveness_score = result.get("effectiveness_score", query.effectiveness_score)
                        query.validation_notes = result.get("validation_notes", "")
                        validated_queries.append(query)
                else:
                    # Keep query if not in validation results
                    validated_queries.append(query)
            
            return validated_queries
            
        except Exception as e:
            logger.error(f"Query validation failed: {e}")
            return queries  # Return original queries if validation fails
    
    def _build_validation_prompt(
        self,
        queries: List[SearchQuery],
        gap: InformationGap,
        topic: str,
        domain: ResearchDomain
    ) -> str:
        """Build prompt for query validation"""
        
        query_list = []
        for i, query in enumerate(queries, 1):
            query_list.append(f"{i}. \"{query.query}\" (Priority: {query.priority.value})")
        
        return f"""
Validate and refine the following search queries for effectiveness:

Research Context:
- Topic: {topic}
- Domain: {domain.value}
- Information Gap: {gap.description}
- Gap Type: {gap.gap_type.value}

Current Queries:
{chr(10).join(query_list)}

Validation Criteria:
1. Query Specificity: Are queries specific enough to return relevant results?
2. Search Optimization: Are queries optimized for Google Search algorithms?
3. Domain Relevance: Do queries use appropriate domain terminology?
4. Gap Alignment: Do queries directly address the information gap?
5. Source Quality: Are queries likely to return high-quality, authoritative sources?

For each query, evaluate and provide:
- original_query: The original query text
- refined_query: Improved version of the query (or same if no improvement needed)
- effectiveness_score: 0.0 to 1.0 rating of query effectiveness
- validation_notes: Brief explanation of assessment and any improvements made
- keep_query: true/false whether this query should be kept

Guidelines for refinement:
- Remove redundant words
- Add domain-specific terminology where helpful
- Ensure queries are 3-8 words for optimal search performance
- Balance specificity with breadth
- Consider search engine optimization best practices
"""
    
    def _generate_fallback_queries(
        self,
        gap: InformationGap,
        topic: str,
        domain: ResearchDomain,
        max_queries: int
    ) -> List[SearchQuery]:
        """Generate fallback queries when Kimi K2 is unavailable"""
        
        logger.warning("Using fallback query generation")
        
        # Use templates if available
        template_key = f"{domain.value}_{gap.gap_type.value}"
        if template_key not in self.query_templates:
            template_key = gap.gap_type.value
        
        template = self.query_templates.get(template_key)
        
        if template:
            template_variables = {
                "topic": topic,
                "domain": domain.value,
                "gap_description": gap.description[:50]
            }
            
            template_queries = template.generate_queries(template_variables, max_queries)
            
            queries = []
            for query_text in template_queries:
                query = SearchQuery(
                    query=query_text,
                    priority=Priority.MEDIUM,
                    expected_results=10,
                    search_strategy="template_fallback",
                    effectiveness_score=0.6
                )
                queries.append(query)
            
            return queries
        
        # Basic fallback queries
        base_query = f"{topic} {gap.section_id.replace('_', ' ')}"
        
        queries = [
            SearchQuery(
                query=base_query,
                priority=Priority.MEDIUM,
                expected_results=10,
                search_strategy="basic_fallback"
            ),
            SearchQuery(
                query=f"{topic} {gap.gap_type.value}",
                priority=Priority.MEDIUM,
                expected_results=10,
                search_strategy="gap_type_fallback"
            )
        ]
        
        # Add gap-type specific query
        if gap.gap_type == GapType.EVIDENCE:
            queries.append(SearchQuery(
                query=f"{topic} evidence examples case study",
                priority=Priority.HIGH,
                expected_results=15,
                search_strategy="evidence_fallback"
            ))
        elif gap.gap_type == GapType.CITATION:
            queries.append(SearchQuery(
                query=f"{topic} research papers academic sources",
                priority=Priority.HIGH,
                expected_results=15,
                search_strategy="citation_fallback"
            ))
        
        return queries[:max_queries]
    
    async def batch_generate_queries(
        self,
        gaps: List[InformationGap],
        topic: str,
        domain: ResearchDomain,
        max_queries_per_gap: int = 3
    ) -> Dict[str, List[SearchQuery]]:
        """Generate queries for multiple gaps in batch"""
        
        results = {}
        
        # Process gaps in batches to avoid overwhelming the API
        batch_size = 5
        for i in range(0, len(gaps), batch_size):
            batch = gaps[i:i + batch_size]
            
            # Generate queries for each gap in the batch
            batch_tasks = []
            for gap in batch:
                task = self.generate_search_queries(gap, topic, domain, max_queries_per_gap)
                batch_tasks.append((gap.id, task))
            
            # Execute batch
            for gap_id, task in batch_tasks:
                try:
                    queries = await task
                    results[gap_id] = queries
                except Exception as e:
                    logger.error(f"Failed to generate queries for gap {gap_id}: {e}")
                    results[gap_id] = []
        
        return results
    
    def get_query_statistics(self, queries: List[SearchQuery]) -> Dict[str, Any]:
        """Get statistics about generated queries"""
        
        if not queries:
            return {"total": 0}
        
        priority_counts = {}
        for priority in Priority:
            priority_counts[priority.value] = sum(1 for q in queries if q.priority == priority)
        
        avg_effectiveness = sum(q.effectiveness_score for q in queries) / len(queries)
        
        return {
            "total": len(queries),
            "priority_distribution": priority_counts,
            "average_effectiveness": avg_effectiveness,
            "query_lengths": [len(q.query.split()) for q in queries],
            "strategies": list(set(q.search_strategy for q in queries if q.search_strategy))
        }