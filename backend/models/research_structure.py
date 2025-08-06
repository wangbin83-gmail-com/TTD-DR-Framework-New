"""
Research structure and content models optimized for Kimi K2 integration.
This module provides enhanced data models and utilities for generating
research structures using Kimi K2 AI capabilities.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import json

from .core import (
    ResearchStructure, Section, ResearchDomain, ComplexityLevel,
    Draft, QualityMetrics, Priority
)

class ContentPlaceholderType(str, Enum):
    """Types of content placeholders"""
    INTRODUCTION = "introduction"
    BACKGROUND = "background"
    METHODOLOGY = "methodology"
    ANALYSIS = "analysis"
    FINDINGS = "findings"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    APPENDIX = "appendix"

class ResearchSectionType(str, Enum):
    """Types of research sections"""
    TITLE = "title"
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    LITERATURE_REVIEW = "literature_review"
    METHODOLOGY = "methodology"
    ANALYSIS = "analysis"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    APPENDIX = "appendix"

class ContentPlaceholder(BaseModel):
    """Represents a content placeholder with Kimi K2 generation hints"""
    id: str
    placeholder_type: ContentPlaceholderType
    title: str
    description: str
    estimated_word_count: int = Field(ge=0)
    priority: Priority = Priority.MEDIUM
    kimi_k2_prompt_hints: List[str] = []  # Hints for Kimi K2 content generation
    required_elements: List[str] = []  # Required elements for this section
    
    class Config:
        use_enum_values = True

class EnhancedSection(Section):
    """Enhanced section model with Kimi K2 optimization features"""
    section_type: ResearchSectionType = ResearchSectionType.INTRODUCTION
    content_placeholders: List[ContentPlaceholder] = []
    kimi_k2_generation_context: Dict[str, Any] = {}  # Context for Kimi K2 generation
    quality_requirements: Dict[str, float] = {}  # Quality thresholds for this section
    dependencies: List[str] = []  # Section IDs this section depends on
    
    @validator('content_placeholders')
    def validate_placeholders(cls, v):
        """Ensure placeholder IDs are unique"""
        ids = [p.id for p in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Content placeholder IDs must be unique")
        return v
    
    class Config:
        use_enum_values = True

class DomainSpecificTemplate(BaseModel):
    """Template for domain-specific research structures"""
    domain: ResearchDomain
    template_name: str
    description: str
    default_sections: List[Dict[str, Any]]  # Section templates
    kimi_k2_domain_prompts: Dict[str, str]  # Domain-specific prompts
    quality_criteria: Dict[str, float]  # Domain-specific quality criteria
    estimated_completion_time: int = Field(ge=0)  # Minutes
    
    class Config:
        use_enum_values = True

class KimiK2PromptTemplate(BaseModel):
    """Template for Kimi K2 prompts with domain optimization"""
    template_id: str
    domain: ResearchDomain
    section_type: ResearchSectionType
    prompt_template: str
    variables: List[str] = []  # Variables to be filled in the template
    expected_output_format: str = "text"  # text, json, markdown
    quality_indicators: List[str] = []  # What makes a good response
    
    def format_prompt(self, **kwargs) -> str:
        """Format the prompt template with provided variables"""
        try:
            return self.prompt_template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required variable for prompt template: {e}")
    
    class Config:
        use_enum_values = True

class EnhancedResearchStructure(ResearchStructure):
    """Enhanced research structure with Kimi K2 optimization"""
    sections: List[EnhancedSection]
    domain_template: Optional[DomainSpecificTemplate] = None
    kimi_k2_generation_strategy: Dict[str, Any] = {}
    quality_targets: QualityMetrics = Field(default_factory=lambda: QualityMetrics(
        completeness=0.8,
        coherence=0.8,
        accuracy=0.8,
        citation_quality=0.8
    ))
    generation_metadata: Dict[str, Any] = {}  # Metadata about how structure was generated
    
    @validator('sections')
    def validate_enhanced_sections(cls, v):
        """Validate that all sections are EnhancedSection instances"""
        for section in v:
            if not isinstance(section, EnhancedSection):
                raise ValueError("All sections must be EnhancedSection instances")
        return v
    
    def get_section_by_type(self, section_type: ResearchSectionType) -> Optional[EnhancedSection]:
        """Get section by type"""
        for section in self.sections:
            if section.section_type == section_type:
                return section
        return None
    
    def get_sections_by_dependency(self, section_id: str) -> List[EnhancedSection]:
        """Get sections that depend on the given section"""
        return [s for s in self.sections if section_id in s.dependencies]
    
    def calculate_total_placeholders(self) -> int:
        """Calculate total number of content placeholders"""
        return sum(len(section.content_placeholders) for section in self.sections)
    
    def get_high_priority_placeholders(self) -> List[ContentPlaceholder]:
        """Get all high priority content placeholders"""
        placeholders = []
        for section in self.sections:
            placeholders.extend([
                p for p in section.content_placeholders 
                if p.priority in [Priority.HIGH, Priority.CRITICAL]
            ])
        return placeholders

class QualityAssessmentCriteria(BaseModel):
    """Criteria for assessing research quality using Kimi K2"""
    criteria_id: str
    name: str
    description: str
    weight: float = Field(ge=0.0, le=1.0)
    kimi_k2_evaluation_prompt: str
    expected_score_range: tuple[float, float] = (0.0, 1.0)
    
    @validator('expected_score_range')
    def validate_score_range(cls, v):
        """Validate score range"""
        if v[0] >= v[1] or v[0] < 0 or v[1] > 1:
            raise ValueError("Score range must be (min, max) where 0 <= min < max <= 1")
        return v

class KimiK2QualityAssessment(BaseModel):
    """Quality assessment results from Kimi K2"""
    assessment_id: str
    draft_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    criteria_scores: Dict[str, float] = {}  # criteria_id -> score
    overall_score: float = Field(ge=0.0, le=1.0)
    detailed_feedback: str = ""
    improvement_suggestions: List[str] = []
    kimi_k2_confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    
    @validator('overall_score')
    def validate_overall_score(cls, v, values):
        """Calculate overall score if not provided"""
        if 'criteria_scores' in values and values['criteria_scores']:
            scores = list(values['criteria_scores'].values())
            calculated_score = sum(scores) / len(scores) if scores else 0.0
            # Allow some tolerance for manual override
            if abs(v - calculated_score) > 0.1:
                raise ValueError(f"Overall score {v} doesn't match calculated score {calculated_score}")
        return v

# Domain-specific templates
DOMAIN_TEMPLATES = {
    ResearchDomain.GENERAL: DomainSpecificTemplate(
        domain=ResearchDomain.GENERAL,
        template_name="General Research",
        description="Template for general research reports",
        default_sections=[
            {
                "section_type": "introduction",
                "title": "Introduction",
                "estimated_length": 500,
                "content_placeholders": [
                    {
                        "placeholder_type": "introduction",
                        "title": "Topic Overview",
                        "description": "Overview of the research topic",
                        "estimated_word_count": 200,
                        "kimi_k2_prompt_hints": [
                            "Provide clear context and background",
                            "Define key terms and concepts"
                        ]
                    }
                ]
            },
            {
                "section_type": "analysis",
                "title": "Analysis",
                "estimated_length": 1000,
                "content_placeholders": [
                    {
                        "placeholder_type": "analysis",
                        "title": "Main Analysis",
                        "description": "Core analysis of the topic",
                        "estimated_word_count": 500,
                        "kimi_k2_prompt_hints": [
                            "Present evidence and reasoning",
                            "Include multiple perspectives"
                        ]
                    }
                ]
            },
            {
                "section_type": "conclusion",
                "title": "Conclusion",
                "estimated_length": 300,
                "content_placeholders": [
                    {
                        "placeholder_type": "conclusion",
                        "title": "Summary and Implications",
                        "description": "Summary of findings and implications",
                        "estimated_word_count": 300,
                        "kimi_k2_prompt_hints": [
                            "Summarize key findings",
                            "Discuss implications and future directions"
                        ]
                    }
                ]
            }
        ],
        kimi_k2_domain_prompts={
            "introduction": "Provide clear and accessible introduction",
            "analysis": "Present balanced analysis with evidence",
            "conclusion": "Summarize findings and implications"
        },
        quality_criteria={
            "clarity": 0.8,
            "completeness": 0.8,
            "evidence_quality": 0.7
        },
        estimated_completion_time=90
    ),
    
    ResearchDomain.TECHNOLOGY: DomainSpecificTemplate(
        domain=ResearchDomain.TECHNOLOGY,
        template_name="Technology Research",
        description="Template for technology-focused research reports",
        default_sections=[
            {
                "section_type": "introduction",
                "title": "Introduction",
                "estimated_length": 500,
                "content_placeholders": [
                    {
                        "placeholder_type": "introduction",
                        "title": "Technology Overview",
                        "description": "Overview of the technology being researched",
                        "estimated_word_count": 200,
                        "kimi_k2_prompt_hints": [
                            "Focus on technical specifications",
                            "Include current market adoption",
                            "Mention key players and competitors"
                        ]
                    }
                ]
            },
            {
                "section_type": "methodology",
                "title": "Technical Analysis",
                "estimated_length": 800,
                "content_placeholders": [
                    {
                        "placeholder_type": "methodology",
                        "title": "Technical Evaluation Criteria",
                        "description": "Criteria used to evaluate the technology",
                        "estimated_word_count": 300,
                        "kimi_k2_prompt_hints": [
                            "Include performance metrics",
                            "Consider scalability factors",
                            "Address security considerations"
                        ]
                    }
                ]
            }
        ],
        kimi_k2_domain_prompts={
            "introduction": "Focus on technical innovation and market impact",
            "methodology": "Emphasize quantitative analysis and benchmarking",
            "results": "Present data-driven findings with technical depth"
        },
        quality_criteria={
            "technical_accuracy": 0.9,
            "market_relevance": 0.8,
            "innovation_assessment": 0.8
        },
        estimated_completion_time=120
    ),
    
    ResearchDomain.SCIENCE: DomainSpecificTemplate(
        domain=ResearchDomain.SCIENCE,
        template_name="Scientific Research",
        description="Template for scientific research reports",
        default_sections=[
            {
                "section_type": "abstract",
                "title": "Abstract",
                "estimated_length": 250,
                "content_placeholders": [
                    {
                        "placeholder_type": "introduction",
                        "title": "Research Summary",
                        "description": "Concise summary of research objectives and findings",
                        "estimated_word_count": 250,
                        "kimi_k2_prompt_hints": [
                            "Include hypothesis and methodology",
                            "Summarize key findings",
                            "State implications and significance"
                        ]
                    }
                ]
            },
            {
                "section_type": "methodology",
                "title": "Methods",
                "estimated_length": 1000,
                "content_placeholders": [
                    {
                        "placeholder_type": "methodology",
                        "title": "Experimental Design",
                        "description": "Detailed description of experimental methodology",
                        "estimated_word_count": 500,
                        "kimi_k2_prompt_hints": [
                            "Include sample size and selection criteria",
                            "Describe data collection procedures",
                            "Explain statistical analysis methods"
                        ]
                    }
                ]
            }
        ],
        kimi_k2_domain_prompts={
            "abstract": "Provide concise scientific summary with key findings",
            "methodology": "Detail experimental procedures and statistical methods",
            "results": "Present findings with appropriate statistical analysis"
        },
        quality_criteria={
            "scientific_rigor": 0.95,
            "reproducibility": 0.9,
            "statistical_validity": 0.9
        },
        estimated_completion_time=180
    ),
    
    ResearchDomain.BUSINESS: DomainSpecificTemplate(
        domain=ResearchDomain.BUSINESS,
        template_name="Business Analysis",
        description="Template for business research and analysis",
        default_sections=[
            {
                "section_type": "introduction",
                "title": "Executive Summary",
                "estimated_length": 400,
                "content_placeholders": [
                    {
                        "placeholder_type": "introduction",
                        "title": "Business Context",
                        "description": "Overview of business context and objectives",
                        "estimated_word_count": 200,
                        "kimi_k2_prompt_hints": [
                            "Include market context and business drivers",
                            "State key business questions",
                            "Outline expected outcomes"
                        ]
                    }
                ]
            },
            {
                "section_type": "analysis",
                "title": "Market Analysis",
                "estimated_length": 1200,
                "content_placeholders": [
                    {
                        "placeholder_type": "analysis",
                        "title": "Competitive Landscape",
                        "description": "Analysis of competitive environment",
                        "estimated_word_count": 600,
                        "kimi_k2_prompt_hints": [
                            "Include competitor analysis",
                            "Assess market positioning",
                            "Identify competitive advantages"
                        ]
                    }
                ]
            }
        ],
        kimi_k2_domain_prompts={
            "introduction": "Focus on business value and strategic importance",
            "analysis": "Provide data-driven business insights and recommendations",
            "conclusion": "Include actionable business recommendations"
        },
        quality_criteria={
            "business_relevance": 0.9,
            "data_quality": 0.8,
            "actionability": 0.85
        },
        estimated_completion_time=150
    ),
    
    ResearchDomain.ACADEMIC: DomainSpecificTemplate(
        domain=ResearchDomain.ACADEMIC,
        template_name="Academic Research",
        description="Template for academic research papers",
        default_sections=[
            {
                "section_type": "abstract",
                "title": "Abstract",
                "estimated_length": 300,
                "content_placeholders": [
                    {
                        "placeholder_type": "introduction",
                        "title": "Research Abstract",
                        "description": "Comprehensive abstract of the research",
                        "estimated_word_count": 300,
                        "kimi_k2_prompt_hints": [
                            "Include research question and methodology",
                            "Summarize main findings and contributions",
                            "State theoretical and practical implications"
                        ]
                    }
                ]
            },
            {
                "section_type": "literature_review",
                "title": "Literature Review",
                "estimated_length": 1500,
                "content_placeholders": [
                    {
                        "placeholder_type": "background",
                        "title": "Theoretical Framework",
                        "description": "Review of relevant literature and theoretical foundation",
                        "estimated_word_count": 750,
                        "kimi_k2_prompt_hints": [
                            "Synthesize existing research",
                            "Identify gaps in current knowledge",
                            "Establish theoretical foundation"
                        ]
                    }
                ]
            },
            {
                "section_type": "methodology",
                "title": "Methodology",
                "estimated_length": 1000,
                "content_placeholders": [
                    {
                        "placeholder_type": "methodology",
                        "title": "Research Design",
                        "description": "Detailed research methodology and approach",
                        "estimated_word_count": 500,
                        "kimi_k2_prompt_hints": [
                            "Justify methodological choices",
                            "Describe data collection and analysis",
                            "Address validity and reliability"
                        ]
                    }
                ]
            }
        ],
        kimi_k2_domain_prompts={
            "abstract": "Provide scholarly abstract with clear contributions",
            "literature_review": "Synthesize existing research and identify gaps",
            "methodology": "Detail rigorous research methodology",
            "results": "Present findings with academic rigor",
            "discussion": "Interpret results within theoretical framework"
        },
        quality_criteria={
            "academic_rigor": 0.95,
            "theoretical_contribution": 0.9,
            "methodological_soundness": 0.9,
            "citation_completeness": 0.95
        },
        estimated_completion_time=240
    )
}

# Kimi K2 prompt templates for different sections
KIMI_K2_PROMPT_TEMPLATES = {
    "structure_generation": KimiK2PromptTemplate(
        template_id="structure_gen",
        domain=ResearchDomain.GENERAL,
        section_type=ResearchSectionType.TITLE,
        prompt_template="""
Generate a comprehensive research structure for the topic: "{topic}"

Domain: {domain}
Complexity Level: {complexity_level}
Target Length: {target_length} words

Please create a detailed research structure with the following requirements:
1. Generate 5-8 main sections appropriate for this domain
2. For each section, provide:
   - Section title
   - Section type (introduction, methodology, analysis, etc.)
   - Estimated word count
   - 2-3 content placeholders with descriptions
   - Dependencies on other sections

Return the response as a JSON object with the following structure:
{{
    "sections": [
        {{
            "id": "section_id",
            "title": "Section Title",
            "section_type": "section_type",
            "estimated_length": word_count,
            "content_placeholders": [
                {{
                    "id": "placeholder_id",
                    "placeholder_type": "placeholder_type",
                    "title": "Placeholder Title",
                    "description": "Detailed description",
                    "estimated_word_count": word_count,
                    "priority": "medium",
                    "kimi_k2_prompt_hints": ["hint1", "hint2"]
                }}
            ],
            "dependencies": ["dependent_section_id"]
        }}
    ],
    "estimated_total_length": total_word_count,
    "generation_strategy": "strategy_description"
}}
""",
        variables=["topic", "domain", "complexity_level", "target_length"],
        expected_output_format="json",
        quality_indicators=[
            "Logical section progression",
            "Appropriate depth for complexity level",
            "Domain-specific section types",
            "Realistic word count estimates"
        ]
    ),
    
    "content_placeholder": KimiK2PromptTemplate(
        template_id="content_placeholder",
        domain=ResearchDomain.GENERAL,
        section_type=ResearchSectionType.INTRODUCTION,
        prompt_template="""
Generate content for the following research section placeholder:

Section: {section_title}
Placeholder: {placeholder_title}
Description: {placeholder_description}
Target Word Count: {target_word_count}
Domain: {domain}
Research Topic: {research_topic}

Context from previous sections:
{previous_context}

Requirements:
{requirements}

Please generate high-quality content that:
1. Fits the specified word count (±10%)
2. Maintains academic/professional tone appropriate for {domain}
3. Integrates well with the overall research structure
4. Includes relevant examples and evidence where appropriate
5. Uses proper citations format when referencing sources

Generate the content in markdown format.
""",
        variables=[
            "section_title", "placeholder_title", "placeholder_description",
            "target_word_count", "domain", "research_topic", "previous_context",
            "requirements"
        ],
        expected_output_format="markdown",
        quality_indicators=[
            "Appropriate word count",
            "Professional tone",
            "Logical flow",
            "Relevant examples",
            "Proper citations"
        ]
    ),
    
    "quality_assessment": KimiK2PromptTemplate(
        template_id="quality_assessment",
        domain=ResearchDomain.GENERAL,
        section_type=ResearchSectionType.TITLE,
        prompt_template="""
Assess the quality of this research draft:

Topic: {topic}
Domain: {domain}
Target Quality Metrics:
- Completeness: {target_completeness}
- Coherence: {target_coherence}
- Accuracy: {target_accuracy}
- Citation Quality: {target_citation_quality}

Draft Content:
{draft_content}

Please evaluate the draft and provide:
1. Scores for each quality metric (0.0 to 1.0)
2. Overall quality score
3. Detailed feedback on strengths and weaknesses
4. Specific improvement suggestions
5. Confidence level in your assessment

Return the assessment as a JSON object:
{{
    "criteria_scores": {{
        "completeness": score,
        "coherence": score,
        "accuracy": score,
        "citation_quality": score
    }},
    "overall_score": score,
    "detailed_feedback": "detailed_feedback_text",
    "improvement_suggestions": ["suggestion1", "suggestion2"],
    "confidence": confidence_score
}}
""",
        variables=[
            "topic", "domain", "target_completeness", "target_coherence",
            "target_accuracy", "target_citation_quality", "draft_content"
        ],
        expected_output_format="json",
        quality_indicators=[
            "Accurate quality scoring",
            "Constructive feedback",
            "Specific improvement suggestions",
            "Appropriate confidence level"
        ]
    )
}

def get_domain_template(domain: ResearchDomain) -> DomainSpecificTemplate:
    """Get domain-specific template"""
    return DOMAIN_TEMPLATES.get(domain, DOMAIN_TEMPLATES[ResearchDomain.GENERAL])

def get_prompt_template(template_id: str) -> Optional[KimiK2PromptTemplate]:
    """Get Kimi K2 prompt template by ID"""
    return KIMI_K2_PROMPT_TEMPLATES.get(template_id)