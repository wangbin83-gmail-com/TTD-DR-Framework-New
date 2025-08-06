"""
Domain-specific adaptation system for TTD-DR framework.
This module provides domain detection, adaptation algorithms, and configurable
research strategies for different research domains.
"""

from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import re
import json
from datetime import datetime
from pydantic import BaseModel, Field

from models.core import (
    ResearchDomain, ComplexityLevel, ResearchRequirements,
    Draft, QualityMetrics, InformationGap, SearchQuery, Priority
)
from models.research_structure import (
    DomainSpecificTemplate, EnhancedResearchStructure, 
    KimiK2PromptTemplate, get_domain_template, get_prompt_template
)
from .kimi_k2_client import KimiK2Client


class DomainConfidence(BaseModel):
    """Confidence score for domain detection"""
    domain: ResearchDomain
    confidence: float = Field(ge=0.0, le=1.0)
    indicators: List[str] = []
    keywords_matched: List[str] = []


class DomainDetectionResult(BaseModel):
    """Result of domain detection analysis"""
    primary_domain: ResearchDomain
    confidence: float = Field(ge=0.0, le=1.0)
    secondary_domains: List[DomainConfidence] = []
    detection_method: str
    keywords_found: List[str] = []
    reasoning: str = ""


class ResearchStrategy(BaseModel):
    """Configurable research strategy for specific domains"""
    domain: ResearchDomain
    strategy_name: str
    description: str
    
    # Search and retrieval configuration
    preferred_source_types: List[str] = []
    search_query_templates: List[str] = []
    max_sources_per_gap: int = 5
    credibility_weights: Dict[str, float] = {}
    
    # Quality assessment configuration
    quality_criteria: Dict[str, float] = {}
    iteration_thresholds: Dict[str, float] = {}
    
    # Content generation configuration
    writing_style: str = "academic"
    citation_format: str = "apa"
    terminology_preferences: Dict[str, str] = {}
    
    # Kimi K2 optimization
    kimi_k2_system_prompts: Dict[str, str] = {}
    kimi_k2_temperature: float = 0.7
    kimi_k2_max_tokens: int = 2000


class TerminologyHandler(BaseModel):
    """Handler for domain-specific terminology"""
    domain: ResearchDomain
    terminology_map: Dict[str, str] = {}  # general term -> domain-specific term
    abbreviations: Dict[str, str] = {}  # abbreviation -> full form
    definitions: Dict[str, str] = {}  # term -> definition
    context_rules: List[str] = []  # Rules for term usage in context


class DomainAdapter:
    """Main domain adaptation system"""
    
    def __init__(self, kimi_client: Optional[KimiK2Client] = None):
        self.kimi_client = kimi_client or KimiK2Client()
        self.domain_keywords = self._initialize_domain_keywords()
        self.research_strategies = self._initialize_research_strategies()
        self.terminology_handlers = self._initialize_terminology_handlers()
    
    def generate_content(self, prompt: str, **kwargs) -> str:
        """Synchronous wrapper for Kimi K2 content generation"""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new event loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._async_generate_content(prompt, **kwargs))
                    return future.result()
            else:
                return loop.run_until_complete(self._async_generate_content(prompt, **kwargs))
        except Exception as e:
            # Fallback for testing or when Kimi K2 is not available
            logger.warning(f"Kimi K2 generation failed, using fallback: {e}")
            return self._fallback_content_generation(prompt, **kwargs)
    
    async def _async_generate_content(self, prompt: str, **kwargs) -> str:
        """Async content generation using Kimi K2"""
        response = await self.kimi_client.generate_text(prompt, **kwargs)
        return response.content
    
    def _fallback_content_generation(self, prompt: str, **kwargs) -> str:
        """Fallback content generation when Kimi K2 is not available"""
        # Simple fallback for testing
        if "domain" in prompt.lower():
            return json.dumps({
                "TECHNOLOGY": 0.3,
                "SCIENCE": 0.3,
                "BUSINESS": 0.3,
                "ACADEMIC": 0.3,
                "GENERAL": 0.5,
                "reasoning": "Fallback domain detection"
            })
        return "Fallback response"
    
    def detect_domain(self, topic: str, content: Optional[str] = None) -> DomainDetectionResult:
        """
        Detect the research domain based on topic and optional content.
        Uses both keyword matching and Kimi K2 analysis.
        """
        # Keyword-based detection
        keyword_scores = self._calculate_keyword_scores(topic, content)
        
        # Kimi K2-based detection for more nuanced analysis
        kimi_analysis = self._kimi_domain_analysis(topic, content)
        
        # Combine results
        combined_scores = self._combine_detection_results(keyword_scores, kimi_analysis)
        
        # Determine primary domain
        primary_domain = max(combined_scores.items(), key=lambda x: x[1])[0]
        primary_confidence = combined_scores[primary_domain]
        
        # Get secondary domains
        secondary_domains = [
            DomainConfidence(
                domain=domain,
                confidence=score,
                indicators=self._get_domain_indicators(domain, topic, content),
                keywords_matched=self._get_matched_keywords(domain, topic, content)
            )
            for domain, score in combined_scores.items()
            if domain != primary_domain and score > 0.2
        ]
        
        return DomainDetectionResult(
            primary_domain=primary_domain,
            confidence=primary_confidence,
            secondary_domains=sorted(secondary_domains, key=lambda x: x.confidence, reverse=True),
            detection_method="hybrid_keyword_kimi",
            keywords_found=self._get_matched_keywords(primary_domain, topic, content),
            reasoning=f"Detected {primary_domain.value} domain with {primary_confidence:.2f} confidence based on keyword analysis and Kimi K2 semantic understanding"
        )
    
    def adapt_research_requirements(
        self, 
        requirements: ResearchRequirements, 
        domain_result: DomainDetectionResult
    ) -> ResearchRequirements:
        """Adapt research requirements based on detected domain"""
        strategy = self.research_strategies[domain_result.primary_domain]
        
        # Create adapted requirements
        adapted_requirements = ResearchRequirements(
            domain=domain_result.primary_domain,
            complexity_level=requirements.complexity_level,
            max_iterations=requirements.max_iterations,
            quality_threshold=strategy.iteration_thresholds.get('quality_threshold', requirements.quality_threshold),
            max_sources=strategy.max_sources_per_gap * 4,  # Estimate based on expected gaps
            preferred_source_types=strategy.preferred_source_types or requirements.preferred_source_types
        )
        
        return adapted_requirements
    
    def generate_domain_specific_structure(
        self, 
        topic: str, 
        domain: ResearchDomain,
        complexity_level: ComplexityLevel = ComplexityLevel.INTERMEDIATE
    ) -> EnhancedResearchStructure:
        """Generate domain-specific research structure using Kimi K2"""
        template = get_domain_template(domain)
        prompt_template = get_prompt_template("structure_generation")
        
        if not prompt_template:
            raise ValueError("Structure generation prompt template not found")
        
        # Prepare prompt variables
        target_length = self._estimate_target_length(complexity_level)
        prompt = prompt_template.format_prompt(
            topic=topic,
            domain=domain.value,
            complexity_level=complexity_level.value,
            target_length=target_length
        )
        
        # Get structure from Kimi K2
        try:
            response = self.generate_content(
                prompt=prompt,
                temperature=0.3,  # Lower temperature for structured output
                max_tokens=2000
            )
            
            structure_data = json.loads(response)
            
            # Convert to EnhancedResearchStructure
            structure = self._convert_to_enhanced_structure(
                structure_data, domain, template, complexity_level
            )
            
            return structure
            
        except Exception as e:
            # Fallback to template-based structure
            return self._create_fallback_structure(topic, domain, template, complexity_level)
    
    def adapt_search_queries(
        self, 
        gaps: List[InformationGap], 
        domain: ResearchDomain
    ) -> List[InformationGap]:
        """Adapt search queries for domain-specific optimization"""
        strategy = self.research_strategies[domain]
        terminology = self.terminology_handlers[domain]
        
        adapted_gaps = []
        for gap in gaps:
            adapted_gap = gap.copy(deep=True)
            
            # Enhance search queries with domain-specific terms
            enhanced_queries = []
            for query in gap.search_queries:
                enhanced_query = self._enhance_search_query(query, strategy, terminology)
                enhanced_queries.append(enhanced_query)
            
            # Add domain-specific query templates
            for template in strategy.search_query_templates:
                if template not in [q.query for q in enhanced_queries]:
                    domain_query = SearchQuery(
                        query=template.format(topic=gap.description),
                        domain=domain.value,
                        priority=gap.priority,
                        search_strategy="domain_template"
                    )
                    enhanced_queries.append(domain_query)
            
            adapted_gap.search_queries = enhanced_queries
            adapted_gaps.append(adapted_gap)
        
        return adapted_gaps
    
    def apply_domain_formatting(
        self, 
        content: str, 
        domain: ResearchDomain,
        section_type: str = "general"
    ) -> str:
        """Apply domain-specific formatting and terminology"""
        strategy = self.research_strategies[domain]
        terminology = self.terminology_handlers[domain]
        
        # Apply terminology mapping
        formatted_content = self._apply_terminology_mapping(content, terminology)
        
        # Apply domain-specific formatting rules
        formatted_content = self._apply_formatting_rules(formatted_content, strategy, section_type)
        
        # Apply citation formatting
        formatted_content = self._apply_citation_formatting(formatted_content, strategy.citation_format)
        
        return formatted_content
    
    def get_domain_quality_criteria(self, domain: ResearchDomain) -> Dict[str, float]:
        """Get domain-specific quality criteria"""
        strategy = self.research_strategies[domain]
        return strategy.quality_criteria
    
    def get_kimi_system_prompt(self, domain: ResearchDomain, task_type: str) -> str:
        """Get domain-specific Kimi K2 system prompt"""
        strategy = self.research_strategies[domain]
        return strategy.kimi_k2_system_prompts.get(
            task_type, 
            f"You are an expert researcher in {domain.value}. Provide accurate, well-researched content."
        )
    
    # Private helper methods
    
    def _initialize_domain_keywords(self) -> Dict[ResearchDomain, List[str]]:
        """Initialize domain-specific keywords for detection"""
        return {
            ResearchDomain.TECHNOLOGY: [
                "software", "hardware", "algorithm", "programming", "development",
                "artificial intelligence", "machine learning", "blockchain", "cloud",
                "cybersecurity", "data science", "IoT", "API", "framework", "platform",
                "mobile", "web", "database", "network", "system", "architecture",
                "innovation", "digital", "tech", "computing", "automation"
            ],
            ResearchDomain.SCIENCE: [
                "research", "experiment", "hypothesis", "methodology", "analysis",
                "data", "results", "findings", "study", "investigation", "theory",
                "empirical", "statistical", "quantitative", "qualitative", "peer-reviewed",
                "journal", "publication", "evidence", "observation", "measurement",
                "laboratory", "clinical", "trial", "sample", "population", "correlation"
            ],
            ResearchDomain.BUSINESS: [
                "market", "business", "strategy", "management", "finance", "revenue",
                "profit", "customer", "competition", "industry", "economic", "commercial",
                "enterprise", "corporate", "organization", "leadership", "operations",
                "marketing", "sales", "investment", "ROI", "KPI", "performance",
                "growth", "expansion", "merger", "acquisition", "stakeholder"
            ],
            ResearchDomain.ACADEMIC: [
                "academic", "scholarly", "university", "education", "curriculum",
                "pedagogy", "learning", "teaching", "student", "faculty", "dissertation",
                "thesis", "literature review", "theoretical", "conceptual", "framework",
                "citation", "bibliography", "peer review", "conference", "symposium",
                "journal", "publication", "research question", "methodology", "analysis"
            ],
            ResearchDomain.GENERAL: [
                "overview", "introduction", "background", "context", "information",
                "topic", "subject", "issue", "question", "problem", "solution",
                "discussion", "analysis", "review", "summary", "conclusion",
                "perspective", "viewpoint", "opinion", "fact", "detail", "aspect"
            ]
        }
    
    def _initialize_research_strategies(self) -> Dict[ResearchDomain, ResearchStrategy]:
        """Initialize domain-specific research strategies"""
        return {
            ResearchDomain.TECHNOLOGY: ResearchStrategy(
                domain=ResearchDomain.TECHNOLOGY,
                strategy_name="Technology Research Strategy",
                description="Optimized for technology and software research",
                preferred_source_types=["tech_blogs", "documentation", "github", "stackoverflow", "tech_news"],
                search_query_templates=[
                    "{topic} technical specifications",
                    "{topic} implementation guide",
                    "{topic} performance benchmarks",
                    "{topic} best practices",
                    "{topic} security considerations"
                ],
                max_sources_per_gap=7,
                credibility_weights={
                    "official_documentation": 0.9,
                    "tech_blogs": 0.7,
                    "stackoverflow": 0.6,
                    "github": 0.8,
                    "academic": 0.85
                },
                quality_criteria={
                    "technical_accuracy": 0.9,
                    "implementation_detail": 0.8,
                    "current_relevance": 0.85,
                    "practical_applicability": 0.8
                },
                iteration_thresholds={
                    "quality_threshold": 0.85,
                    "completeness_threshold": 0.8
                },
                writing_style="technical",
                citation_format="ieee",
                terminology_preferences={
                    "AI": "Artificial Intelligence",
                    "ML": "Machine Learning",
                    "API": "Application Programming Interface"
                },
                kimi_k2_system_prompts={
                    "content_generation": "You are a technical expert. Focus on accuracy, implementation details, and current best practices.",
                    "gap_analysis": "Identify technical gaps that would affect implementation or understanding.",
                    "quality_assessment": "Evaluate technical accuracy, completeness, and practical applicability."
                },
                kimi_k2_temperature=0.3,
                kimi_k2_max_tokens=2500
            ),
            
            ResearchDomain.SCIENCE: ResearchStrategy(
                domain=ResearchDomain.SCIENCE,
                strategy_name="Scientific Research Strategy",
                description="Optimized for scientific research and analysis",
                preferred_source_types=["pubmed", "arxiv", "scientific_journals", "research_institutions"],
                search_query_templates=[
                    "{topic} peer reviewed research",
                    "{topic} systematic review",
                    "{topic} meta-analysis",
                    "{topic} clinical studies",
                    "{topic} experimental results"
                ],
                max_sources_per_gap=10,
                credibility_weights={
                    "peer_reviewed": 0.95,
                    "pubmed": 0.9,
                    "arxiv": 0.8,
                    "research_institutions": 0.85,
                    "scientific_journals": 0.9
                },
                quality_criteria={
                    "scientific_rigor": 0.95,
                    "evidence_quality": 0.9,
                    "methodology_soundness": 0.9,
                    "reproducibility": 0.85
                },
                iteration_thresholds={
                    "quality_threshold": 0.9,
                    "completeness_threshold": 0.85
                },
                writing_style="scientific",
                citation_format="apa",
                terminology_preferences={
                    "p-value": "statistical significance value",
                    "CI": "Confidence Interval",
                    "RCT": "Randomized Controlled Trial"
                },
                kimi_k2_system_prompts={
                    "content_generation": "You are a scientific researcher. Emphasize evidence-based content, proper methodology, and statistical rigor.",
                    "gap_analysis": "Identify gaps in evidence, methodology, or statistical analysis.",
                    "quality_assessment": "Evaluate scientific rigor, evidence quality, and methodological soundness."
                },
                kimi_k2_temperature=0.2,
                kimi_k2_max_tokens=3000
            ),
            
            ResearchDomain.BUSINESS: ResearchStrategy(
                domain=ResearchDomain.BUSINESS,
                strategy_name="Business Analysis Strategy",
                description="Optimized for business research and market analysis",
                preferred_source_types=["market_reports", "business_news", "financial_data", "industry_analysis"],
                search_query_templates=[
                    "{topic} market analysis",
                    "{topic} industry trends",
                    "{topic} competitive landscape",
                    "{topic} financial performance",
                    "{topic} business strategy"
                ],
                max_sources_per_gap=8,
                credibility_weights={
                    "market_reports": 0.9,
                    "financial_data": 0.95,
                    "business_news": 0.7,
                    "industry_analysis": 0.85,
                    "consulting_reports": 0.8
                },
                quality_criteria={
                    "business_relevance": 0.9,
                    "data_accuracy": 0.85,
                    "market_insight": 0.8,
                    "actionability": 0.85
                },
                iteration_thresholds={
                    "quality_threshold": 0.8,
                    "completeness_threshold": 0.8
                },
                writing_style="business",
                citation_format="apa",
                terminology_preferences={
                    "ROI": "Return on Investment",
                    "KPI": "Key Performance Indicator",
                    "B2B": "Business-to-Business"
                },
                kimi_k2_system_prompts={
                    "content_generation": "You are a business analyst. Focus on market insights, data-driven analysis, and actionable recommendations.",
                    "gap_analysis": "Identify gaps in market data, competitive analysis, or business metrics.",
                    "quality_assessment": "Evaluate business relevance, data quality, and practical applicability."
                },
                kimi_k2_temperature=0.4,
                kimi_k2_max_tokens=2500
            ),
            
            ResearchDomain.ACADEMIC: ResearchStrategy(
                domain=ResearchDomain.ACADEMIC,
                strategy_name="Academic Research Strategy",
                description="Optimized for academic research and scholarly work",
                preferred_source_types=["academic_journals", "university_repositories", "scholarly_databases", "conference_proceedings"],
                search_query_templates=[
                    "{topic} academic research",
                    "{topic} scholarly articles",
                    "{topic} theoretical framework",
                    "{topic} literature review",
                    "{topic} empirical studies"
                ],
                max_sources_per_gap=12,
                credibility_weights={
                    "academic_journals": 0.95,
                    "university_repositories": 0.9,
                    "conference_proceedings": 0.85,
                    "scholarly_databases": 0.9,
                    "dissertations": 0.8
                },
                quality_criteria={
                    "academic_rigor": 0.95,
                    "theoretical_contribution": 0.9,
                    "citation_completeness": 0.95,
                    "methodological_soundness": 0.9
                },
                iteration_thresholds={
                    "quality_threshold": 0.9,
                    "completeness_threshold": 0.9
                },
                writing_style="academic",
                citation_format="apa",
                terminology_preferences={
                    "et al.": "and others",
                    "ibid.": "in the same place",
                    "cf.": "compare"
                },
                kimi_k2_system_prompts={
                    "content_generation": "You are an academic researcher. Emphasize theoretical rigor, comprehensive literature review, and scholarly contribution.",
                    "gap_analysis": "Identify gaps in theoretical framework, literature coverage, or methodological approach.",
                    "quality_assessment": "Evaluate academic rigor, theoretical contribution, and scholarly standards."
                },
                kimi_k2_temperature=0.3,
                kimi_k2_max_tokens=3500
            ),
            
            ResearchDomain.GENERAL: ResearchStrategy(
                domain=ResearchDomain.GENERAL,
                strategy_name="General Research Strategy",
                description="Balanced approach for general research topics",
                preferred_source_types=["reputable_websites", "news_sources", "encyclopedias", "government_sources"],
                search_query_templates=[
                    "{topic} overview",
                    "{topic} comprehensive guide",
                    "{topic} current information",
                    "{topic} expert analysis",
                    "{topic} recent developments"
                ],
                max_sources_per_gap=6,
                credibility_weights={
                    "government_sources": 0.9,
                    "reputable_websites": 0.7,
                    "news_sources": 0.6,
                    "encyclopedias": 0.8,
                    "expert_analysis": 0.75
                },
                quality_criteria={
                    "completeness": 0.8,
                    "clarity": 0.85,
                    "accuracy": 0.8,
                    "balance": 0.75
                },
                iteration_thresholds={
                    "quality_threshold": 0.75,
                    "completeness_threshold": 0.8
                },
                writing_style="accessible",
                citation_format="apa",
                terminology_preferences={},
                kimi_k2_system_prompts={
                    "content_generation": "You are a knowledgeable researcher. Provide clear, balanced, and accessible content.",
                    "gap_analysis": "Identify gaps in coverage, clarity, or balance of perspectives.",
                    "quality_assessment": "Evaluate completeness, clarity, and overall quality."
                },
                kimi_k2_temperature=0.5,
                kimi_k2_max_tokens=2000
            )
        }
    
    def _initialize_terminology_handlers(self) -> Dict[ResearchDomain, TerminologyHandler]:
        """Initialize domain-specific terminology handlers"""
        return {
            ResearchDomain.TECHNOLOGY: TerminologyHandler(
                domain=ResearchDomain.TECHNOLOGY,
                terminology_map={
                    "computer program": "software application",
                    "computer": "system",
                    "internet": "web",
                    "website": "web application"
                },
                abbreviations={
                    "AI": "Artificial Intelligence",
                    "ML": "Machine Learning",
                    "API": "Application Programming Interface",
                    "SDK": "Software Development Kit",
                    "IDE": "Integrated Development Environment",
                    "UI": "User Interface",
                    "UX": "User Experience",
                    "IoT": "Internet of Things",
                    "SaaS": "Software as a Service",
                    "PaaS": "Platform as a Service"
                },
                definitions={
                    "API": "A set of protocols and tools for building software applications",
                    "Framework": "A platform for developing software applications",
                    "Algorithm": "A step-by-step procedure for solving a problem"
                },
                context_rules=[
                    "Always define technical acronyms on first use",
                    "Use precise technical terminology",
                    "Include version numbers when relevant"
                ]
            ),
            
            ResearchDomain.SCIENCE: TerminologyHandler(
                domain=ResearchDomain.SCIENCE,
                terminology_map={
                    "test": "experiment",
                    "result": "finding",
                    "proof": "evidence"
                },
                abbreviations={
                    "DNA": "Deoxyribonucleic Acid",
                    "RNA": "Ribonucleic Acid",
                    "PCR": "Polymerase Chain Reaction",
                    "ELISA": "Enzyme-Linked Immunosorbent Assay",
                    "MRI": "Magnetic Resonance Imaging",
                    "CT": "Computed Tomography",
                    "WHO": "World Health Organization",
                    "FDA": "Food and Drug Administration"
                },
                definitions={
                    "hypothesis": "A testable prediction about the relationship between variables",
                    "p-value": "The probability of obtaining results at least as extreme as observed, assuming the null hypothesis is true",
                    "control group": "A group in an experiment that does not receive the treatment being tested"
                },
                context_rules=[
                    "Use precise scientific terminology",
                    "Include statistical significance levels",
                    "Specify sample sizes and methodologies"
                ]
            ),
            
            ResearchDomain.BUSINESS: TerminologyHandler(
                domain=ResearchDomain.BUSINESS,
                terminology_map={
                    "company": "organization",
                    "money": "capital",
                    "profit": "revenue"
                },
                abbreviations={
                    "ROI": "Return on Investment",
                    "KPI": "Key Performance Indicator",
                    "B2B": "Business-to-Business",
                    "B2C": "Business-to-Consumer",
                    "CEO": "Chief Executive Officer",
                    "CFO": "Chief Financial Officer",
                    "CTO": "Chief Technology Officer",
                    "IPO": "Initial Public Offering",
                    "M&A": "Mergers and Acquisitions"
                },
                definitions={
                    "ROI": "A measure of the efficiency of an investment",
                    "Market Cap": "The total value of a company's shares",
                    "EBITDA": "Earnings Before Interest, Taxes, Depreciation, and Amortization"
                },
                context_rules=[
                    "Include financial metrics and timeframes",
                    "Specify market segments and geographies",
                    "Use industry-standard terminology"
                ]
            ),
            
            ResearchDomain.ACADEMIC: TerminologyHandler(
                domain=ResearchDomain.ACADEMIC,
                terminology_map={
                    "paper": "scholarly article",
                    "study": "research investigation",
                    "book": "monograph"
                },
                abbreviations={
                    "et al.": "and others",
                    "ibid.": "in the same place",
                    "cf.": "compare",
                    "i.e.": "that is",
                    "e.g.": "for example",
                    "viz.": "namely",
                    "PhD": "Doctor of Philosophy",
                    "MA": "Master of Arts",
                    "BA": "Bachelor of Arts"
                },
                definitions={
                    "peer review": "Evaluation of scholarly work by experts in the same field",
                    "citation": "A reference to a published or unpublished source",
                    "bibliography": "A list of sources used in research"
                },
                context_rules=[
                    "Follow academic citation standards",
                    "Use formal academic language",
                    "Include proper attribution for all sources"
                ]
            ),
            
            ResearchDomain.GENERAL: TerminologyHandler(
                domain=ResearchDomain.GENERAL,
                terminology_map={},
                abbreviations={
                    "etc.": "et cetera",
                    "i.e.": "that is",
                    "e.g.": "for example"
                },
                definitions={},
                context_rules=[
                    "Use clear, accessible language",
                    "Define technical terms when used",
                    "Maintain consistent terminology throughout"
                ]
            )
        }    

    def _calculate_keyword_scores(self, topic: str, content: Optional[str] = None) -> Dict[ResearchDomain, float]:
        """Calculate domain scores based on keyword matching"""
        text = f"{topic} {content or ''}".lower()
        scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            matches = sum(1 for keyword in keywords if keyword.lower() in text)
            # Normalize by number of keywords and text length
            score = matches / len(keywords) if keywords else 0
            # Boost score if multiple keywords match
            if matches > 1:
                score *= (1 + (matches - 1) * 0.1)
            scores[domain] = min(score, 1.0)
        
        return scores
    
    def _kimi_domain_analysis(self, topic: str, content: Optional[str] = None) -> Dict[ResearchDomain, float]:
        """Use Kimi K2 for semantic domain analysis"""
        prompt = f"""
        Analyze the following research topic and determine which research domain it belongs to.
        
        Topic: {topic}
        {f"Additional content: {content[:500]}..." if content else ""}
        
        Available domains:
        - TECHNOLOGY: Software, hardware, AI, programming, digital systems
        - SCIENCE: Scientific research, experiments, empirical studies, natural sciences
        - BUSINESS: Market analysis, business strategy, finance, management
        - ACADEMIC: Scholarly research, theoretical frameworks, educational content
        - GENERAL: General topics that don't fit specific domains
        
        Provide confidence scores (0.0 to 1.0) for each domain in JSON format:
        {{
            "TECHNOLOGY": score,
            "SCIENCE": score,
            "BUSINESS": score,
            "ACADEMIC": score,
            "GENERAL": score,
            "reasoning": "explanation of the analysis"
        }}
        """
        
        try:
            response = self.generate_content(
                prompt=prompt,
                temperature=0.3,
                max_tokens=500
            )
            
            analysis = json.loads(response)
            
            # Convert to ResearchDomain enum keys
            domain_scores = {}
            for domain_str, score in analysis.items():
                if domain_str != "reasoning":
                    try:
                        domain = ResearchDomain(domain_str.lower())
                        domain_scores[domain] = float(score)
                    except (ValueError, TypeError):
                        continue
            
            return domain_scores
            
        except Exception as e:
            # Fallback to equal distribution if Kimi analysis fails
            return {domain: 0.2 for domain in ResearchDomain}
    
    def _combine_detection_results(
        self, 
        keyword_scores: Dict[ResearchDomain, float],
        kimi_scores: Dict[ResearchDomain, float]
    ) -> Dict[ResearchDomain, float]:
        """Combine keyword and Kimi analysis results"""
        combined_scores = {}
        
        for domain in ResearchDomain:
            keyword_score = keyword_scores.get(domain, 0.0)
            kimi_score = kimi_scores.get(domain, 0.0)
            
            # Weight Kimi analysis more heavily (70%) than keyword matching (30%)
            combined_score = (keyword_score * 0.3) + (kimi_score * 0.7)
            combined_scores[domain] = combined_score
        
        return combined_scores
    
    def _get_domain_indicators(self, domain: ResearchDomain, topic: str, content: Optional[str] = None) -> List[str]:
        """Get specific indicators that suggest this domain"""
        indicators = []
        text = f"{topic} {content or ''}".lower()
        
        domain_indicators = {
            ResearchDomain.TECHNOLOGY: [
                "Contains technical terminology",
                "Mentions software or hardware",
                "Discusses implementation details",
                "References programming concepts"
            ],
            ResearchDomain.SCIENCE: [
                "Uses scientific methodology terms",
                "Mentions experiments or studies",
                "Contains statistical references",
                "Discusses empirical evidence"
            ],
            ResearchDomain.BUSINESS: [
                "Contains business terminology",
                "Mentions market or financial concepts",
                "Discusses strategy or management",
                "References commercial activities"
            ],
            ResearchDomain.ACADEMIC: [
                "Uses scholarly language",
                "Mentions academic institutions",
                "Contains theoretical references",
                "Discusses educational content"
            ],
            ResearchDomain.GENERAL: [
                "Broad topic coverage",
                "Accessible language",
                "General interest subject",
                "Non-specialized content"
            ]
        }
        
        # Check for specific patterns
        for indicator in domain_indicators.get(domain, []):
            if self._check_indicator_pattern(indicator, text):
                indicators.append(indicator)
        
        return indicators
    
    def _get_matched_keywords(self, domain: ResearchDomain, topic: str, content: Optional[str] = None) -> List[str]:
        """Get keywords that matched for this domain"""
        text = f"{topic} {content or ''}".lower()
        keywords = self.domain_keywords.get(domain, [])
        
        matched = [keyword for keyword in keywords if keyword.lower() in text]
        return matched[:10]  # Limit to top 10 matches
    
    def _check_indicator_pattern(self, indicator: str, text: str) -> bool:
        """Check if an indicator pattern is present in the text"""
        # Simple pattern matching - could be enhanced with more sophisticated NLP
        indicator_patterns = {
            "Contains technical terminology": ["algorithm", "software", "system", "technology"],
            "Mentions software or hardware": ["software", "hardware", "application", "device"],
            "Uses scientific methodology terms": ["experiment", "hypothesis", "methodology", "analysis"],
            "Contains business terminology": ["market", "business", "strategy", "revenue"],
            "Uses scholarly language": ["research", "study", "academic", "theoretical"]
        }
        
        patterns = indicator_patterns.get(indicator, [])
        return any(pattern in text for pattern in patterns)
    
    def _estimate_target_length(self, complexity_level: ComplexityLevel) -> int:
        """Estimate target length based on complexity level"""
        length_map = {
            ComplexityLevel.BASIC: 1500,
            ComplexityLevel.INTERMEDIATE: 3000,
            ComplexityLevel.ADVANCED: 5000,
            ComplexityLevel.EXPERT: 8000
        }
        return length_map.get(complexity_level, 3000)
    
    def _convert_to_enhanced_structure(
        self, 
        structure_data: Dict[str, Any], 
        domain: ResearchDomain,
        template: DomainSpecificTemplate,
        complexity_level: ComplexityLevel
    ) -> EnhancedResearchStructure:
        """Convert Kimi K2 structure data to EnhancedResearchStructure"""
        from ..models.research_structure import (
            EnhancedSection, ContentPlaceholder, ResearchSectionType,
            ContentPlaceholderType, EnhancedResearchStructure
        )
        
        sections = []
        for section_data in structure_data.get("sections", []):
            # Convert placeholders
            placeholders = []
            for placeholder_data in section_data.get("content_placeholders", []):
                placeholder = ContentPlaceholder(
                    id=placeholder_data.get("id", f"placeholder_{len(placeholders)}"),
                    placeholder_type=ContentPlaceholderType(
                        placeholder_data.get("placeholder_type", "introduction")
                    ),
                    title=placeholder_data.get("title", ""),
                    description=placeholder_data.get("description", ""),
                    estimated_word_count=placeholder_data.get("estimated_word_count", 200),
                    priority=Priority(placeholder_data.get("priority", "medium")),
                    kimi_k2_prompt_hints=placeholder_data.get("kimi_k2_prompt_hints", [])
                )
                placeholders.append(placeholder)
            
            # Create enhanced section
            section = EnhancedSection(
                id=section_data.get("id", f"section_{len(sections)}"),
                title=section_data.get("title", ""),
                content="",
                estimated_length=section_data.get("estimated_length", 500),
                section_type=ResearchSectionType(
                    section_data.get("section_type", "introduction")
                ),
                content_placeholders=placeholders,
                dependencies=section_data.get("dependencies", [])
            )
            sections.append(section)
        
        # Create enhanced research structure
        structure = EnhancedResearchStructure(
            sections=sections,
            relationships=[],
            estimated_length=structure_data.get("estimated_total_length", 3000),
            complexity_level=complexity_level,
            domain=domain,
            domain_template=template,
            kimi_k2_generation_strategy=structure_data.get("generation_strategy", {})
        )
        
        return structure
    
    def _create_fallback_structure(
        self, 
        topic: str, 
        domain: ResearchDomain,
        template: DomainSpecificTemplate,
        complexity_level: ComplexityLevel
    ) -> EnhancedResearchStructure:
        """Create fallback structure when Kimi K2 fails"""
        from ..models.research_structure import (
            EnhancedSection, ContentPlaceholder, ResearchSectionType,
            ContentPlaceholderType, EnhancedResearchStructure
        )
        
        # Use template default sections
        sections = []
        for i, section_template in enumerate(template.default_sections):
            placeholders = []
            for j, placeholder_data in enumerate(section_template.get("content_placeholders", [])):
                placeholder = ContentPlaceholder(
                    id=f"placeholder_{i}_{j}",
                    placeholder_type=ContentPlaceholderType(
                        placeholder_data.get("placeholder_type", "introduction")
                    ),
                    title=placeholder_data.get("title", ""),
                    description=placeholder_data.get("description", ""),
                    estimated_word_count=placeholder_data.get("estimated_word_count", 200),
                    kimi_k2_prompt_hints=placeholder_data.get("kimi_k2_prompt_hints", [])
                )
                placeholders.append(placeholder)
            
            section = EnhancedSection(
                id=f"section_{i}",
                title=section_template.get("title", f"Section {i+1}"),
                content="",
                estimated_length=section_template.get("estimated_length", 500),
                section_type=ResearchSectionType(
                    section_template.get("section_type", "introduction")
                ),
                content_placeholders=placeholders
            )
            sections.append(section)
        
        structure = EnhancedResearchStructure(
            sections=sections,
            relationships=[],
            estimated_length=sum(s.estimated_length for s in sections),
            complexity_level=complexity_level,
            domain=domain,
            domain_template=template
        )
        
        return structure
    
    def _enhance_search_query(
        self, 
        query: SearchQuery, 
        strategy: ResearchStrategy,
        terminology: TerminologyHandler
    ) -> SearchQuery:
        """Enhance search query with domain-specific terms"""
        enhanced_query = query.copy(deep=True)
        
        # Apply terminology mapping
        query_text = query.query
        for general_term, domain_term in terminology.terminology_map.items():
            query_text = query_text.replace(general_term, domain_term)
        
        # Add domain-specific search operators
        if strategy.domain == ResearchDomain.SCIENCE:
            query_text += " peer-reviewed"
        elif strategy.domain == ResearchDomain.TECHNOLOGY:
            query_text += " documentation OR tutorial"
        elif strategy.domain == ResearchDomain.BUSINESS:
            query_text += " market analysis OR industry report"
        elif strategy.domain == ResearchDomain.ACADEMIC:
            query_text += " scholarly article OR academic paper"
        
        enhanced_query.query = query_text
        enhanced_query.domain = strategy.domain.value
        enhanced_query.search_strategy = "domain_enhanced"
        
        return enhanced_query
    
    def _apply_terminology_mapping(self, content: str, terminology: TerminologyHandler) -> str:
        """Apply domain-specific terminology mapping"""
        formatted_content = content
        
        # Apply terminology mapping
        for general_term, domain_term in terminology.terminology_map.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(general_term) + r'\b'
            formatted_content = re.sub(pattern, domain_term, formatted_content, flags=re.IGNORECASE)
        
        # Expand abbreviations on first use
        for abbrev, full_form in terminology.abbreviations.items():
            # Find first occurrence and expand it
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            if re.search(pattern, formatted_content):
                formatted_content = re.sub(
                    pattern, 
                    f"{full_form} ({abbrev})", 
                    formatted_content, 
                    count=1
                )
        
        return formatted_content
    
    def _apply_formatting_rules(self, content: str, strategy: ResearchStrategy, section_type: str) -> str:
        """Apply domain-specific formatting rules"""
        formatted_content = content
        
        # Apply writing style adjustments
        if strategy.writing_style == "technical":
            # Add more precise technical language
            formatted_content = self._apply_technical_formatting(formatted_content)
        elif strategy.writing_style == "scientific":
            # Add scientific rigor formatting
            formatted_content = self._apply_scientific_formatting(formatted_content)
        elif strategy.writing_style == "business":
            # Add business-oriented formatting
            formatted_content = self._apply_business_formatting(formatted_content)
        elif strategy.writing_style == "academic":
            # Add academic formatting
            formatted_content = self._apply_academic_formatting(formatted_content)
        
        return formatted_content
    
    def _apply_citation_formatting(self, content: str, citation_format: str) -> str:
        """Apply domain-specific citation formatting"""
        # This is a simplified implementation
        # In practice, you'd want a more sophisticated citation parser
        
        if citation_format == "apa":
            # Apply APA formatting rules
            content = re.sub(r'\(([^)]+), (\d{4})\)', r'(\1, \2)', content)
        elif citation_format == "ieee":
            # Apply IEEE formatting rules
            content = re.sub(r'\(([^)]+), (\d{4})\)', r'[\1, \2]', content)
        
        return content
    
    def _apply_technical_formatting(self, content: str) -> str:
        """Apply technical writing formatting"""
        # Add code formatting for technical terms
        technical_terms = ["API", "SDK", "framework", "algorithm", "database"]
        for term in technical_terms:
            pattern = r'\b' + re.escape(term) + r'\b'
            content = re.sub(pattern, f"`{term}`", content, flags=re.IGNORECASE)
        
        return content
    
    def _apply_scientific_formatting(self, content: str) -> str:
        """Apply scientific writing formatting"""
        # Ensure proper statistical notation
        content = re.sub(r'p\s*=\s*([0-9.]+)', r'*p* = \1', content)
        content = re.sub(r'n\s*=\s*([0-9]+)', r'*n* = \1', content)
        
        return content
    
    def _apply_business_formatting(self, content: str) -> str:
        """Apply business writing formatting"""
        # Format financial figures
        content = re.sub(r'\$([0-9,]+)', r'$\1', content)
        content = re.sub(r'([0-9]+)%', r'\1%', content)
        
        return content
    
    def _apply_academic_formatting(self, content: str) -> str:
        """Apply academic writing formatting"""
        # Ensure proper academic language
        content = re.sub(r'\bet al\b', 'et al.', content)
        content = re.sub(r'\bibid\b', 'ibid.', content)
        
        return content


class DomainAdaptationError(Exception):
    """Custom exception for domain adaptation errors"""
    pass


class DomainMetrics:
    """Metrics for evaluating domain adaptation effectiveness"""
    
    def __init__(self):
        self.detection_accuracy = 0.0
        self.adaptation_effectiveness = 0.0
        self.terminology_consistency = 0.0
        self.format_compliance = 0.0
    
    def calculate_overall_score(self) -> float:
        """Calculate overall domain adaptation score"""
        metrics = [
            self.detection_accuracy,
            self.adaptation_effectiveness,
            self.terminology_consistency,
            self.format_compliance
        ]
        return sum(metrics) / len(metrics) if metrics else 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary"""
        return {
            "detection_accuracy": self.detection_accuracy,
            "adaptation_effectiveness": self.adaptation_effectiveness,
            "terminology_consistency": self.terminology_consistency,
            "format_compliance": self.format_compliance,
            "overall_score": self.calculate_overall_score()
        }