"""
Kimi K2 Research Structure Generator Service.
This service provides functionality to generate research structures,
content placeholders, and quality assessments using Kimi K2 AI.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.core import (
    ResearchDomain, ComplexityLevel, Draft, QualityMetrics
)
from models.research_structure import (
    EnhancedResearchStructure, EnhancedSection, ContentPlaceholder,
    DomainSpecificTemplate, KimiK2PromptTemplate, KimiK2QualityAssessment,
    QualityAssessmentCriteria, ResearchSectionType, ContentPlaceholderType,
    Priority, get_domain_template, get_prompt_template, KIMI_K2_PROMPT_TEMPLATES
)
from services.kimi_k2_client import KimiK2Client, KimiK2Error

logger = logging.getLogger(__name__)

class KimiK2ResearchStructureGenerator:
    """Generator for research structures using Kimi K2 AI"""
    
    def __init__(self, kimi_client: Optional[KimiK2Client] = None):
        self.kimi_client = kimi_client or KimiK2Client()
        self.generation_cache = {}  # Simple cache for repeated requests
    
    async def generate_research_structure(
        self,
        topic: str,
        domain: ResearchDomain = ResearchDomain.GENERAL,
        complexity_level: ComplexityLevel = ComplexityLevel.INTERMEDIATE,
        target_length: int = 3000
    ) -> EnhancedResearchStructure:
        """Generate a complete research structure using Kimi K2"""
        
        try:
            # Get domain template for baseline structure
            domain_template = get_domain_template(domain)
            
            # Generate structure using Kimi K2
            structure_data = await self._generate_structure_with_kimi(
                topic, domain, complexity_level, target_length
            )
            
            # Create enhanced sections
            sections = []
            for section_data in structure_data.get("sections", []):
                section = await self._create_enhanced_section(section_data, domain)
                sections.append(section)
            
            # Create enhanced research structure
            structure = EnhancedResearchStructure(
                sections=sections,
                estimated_length=structure_data.get("estimated_total_length", target_length),
                complexity_level=complexity_level,
                domain=domain,
                domain_template=domain_template,
                kimi_k2_generation_strategy=structure_data.get("generation_strategy", {}),
                generation_metadata={
                    "generated_at": datetime.now().isoformat(),
                    "kimi_k2_model": self.kimi_client.model,
                    "topic": topic,
                    "generation_method": "kimi_k2_structured"
                }
            )
            
            logger.info(f"Generated research structure for topic '{topic}' with {len(sections)} sections")
            return structure
            
        except Exception as e:
            logger.error(f"Failed to generate research structure: {e}")
            # Fallback to template-based generation
            return await self._generate_fallback_structure(topic, domain, complexity_level, target_length)
    
    async def _generate_structure_with_kimi(
        self,
        topic: str,
        domain: ResearchDomain,
        complexity_level: ComplexityLevel,
        target_length: int
    ) -> Dict[str, Any]:
        """Generate structure using Kimi K2 API"""
        
        template = get_prompt_template("structure_generation")
        if not template:
            raise ValueError("Structure generation template not found")
        
        prompt = template.format_prompt(
            topic=topic,
            domain=domain.value,
            complexity_level=complexity_level.value,
            target_length=target_length
        )
        
        try:
            # Use structured response for JSON output
            schema = {
                "type": "object",
                "properties": {
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "title": {"type": "string"},
                                "section_type": {"type": "string"},
                                "estimated_length": {"type": "integer"},
                                "content_placeholders": {"type": "array"},
                                "dependencies": {"type": "array"}
                            }
                        }
                    },
                    "estimated_total_length": {"type": "integer"},
                    "generation_strategy": {"type": "string"}
                }
            }
            
            response = await self.kimi_client.generate_structured_response(prompt, schema)
            return response
            
        except KimiK2Error as e:
            logger.error(f"Kimi K2 API error during structure generation: {e}")
            raise
    
    async def _create_enhanced_section(
        self,
        section_data: Dict[str, Any],
        domain: ResearchDomain
    ) -> EnhancedSection:
        """Create an enhanced section from generated data"""
        
        # Create content placeholders
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
                kimi_k2_prompt_hints=placeholder_data.get("kimi_k2_prompt_hints", []),
                required_elements=placeholder_data.get("required_elements", [])
            )
            placeholders.append(placeholder)
        
        # Create enhanced section
        section = EnhancedSection(
            id=section_data.get("id", f"section_{datetime.now().timestamp()}"),
            title=section_data.get("title", ""),
            estimated_length=section_data.get("estimated_length", 500),
            section_type=ResearchSectionType(
                section_data.get("section_type", "introduction")
            ),
            content_placeholders=placeholders,
            kimi_k2_generation_context={
                "domain": domain.value,
                "generation_timestamp": datetime.now().isoformat()
            },
            quality_requirements={
                "min_completeness": 0.7,
                "min_coherence": 0.8,
                "min_accuracy": 0.8
            },
            dependencies=section_data.get("dependencies", [])
        )
        
        return section
    
    async def _generate_fallback_structure(
        self,
        topic: str,
        domain: ResearchDomain,
        complexity_level: ComplexityLevel,
        target_length: int
    ) -> EnhancedResearchStructure:
        """Generate fallback structure when Kimi K2 is unavailable"""
        
        logger.warning("Using fallback structure generation")
        
        domain_template = get_domain_template(domain)
        
        # Create basic sections based on domain template
        sections = []
        for i, section_template in enumerate(domain_template.default_sections):
            section_data = section_template.copy()
            section_data["id"] = f"section_{i+1}"
            section = await self._create_enhanced_section(section_data, domain)
            sections.append(section)
        
        return EnhancedResearchStructure(
            sections=sections,
            estimated_length=target_length,
            complexity_level=complexity_level,
            domain=domain,
            domain_template=domain_template,
            generation_metadata={
                "generated_at": datetime.now().isoformat(),
                "generation_method": "fallback_template",
                "topic": topic
            }
        )

class KimiK2ContentGenerator:
    """Generator for content placeholders using Kimi K2"""
    
    def __init__(self, kimi_client: Optional[KimiK2Client] = None):
        self.kimi_client = kimi_client or KimiK2Client()
    
    async def generate_placeholder_content(
        self,
        placeholder: ContentPlaceholder,
        section: EnhancedSection,
        research_topic: str,
        domain: ResearchDomain,
        previous_context: str = ""
    ) -> str:
        """Generate content for a specific placeholder"""
        
        try:
            template = get_prompt_template("content_placeholder")
            if not template:
                raise ValueError("Content placeholder template not found")
            
            # Prepare requirements from placeholder
            requirements = []
            if placeholder.required_elements:
                requirements.extend([f"- Include: {elem}" for elem in placeholder.required_elements])
            if placeholder.kimi_k2_prompt_hints:
                requirements.extend([f"- {hint}" for hint in placeholder.kimi_k2_prompt_hints])
            
            prompt = template.format_prompt(
                section_title=section.title,
                placeholder_title=placeholder.title,
                placeholder_description=placeholder.description,
                target_word_count=placeholder.estimated_word_count,
                domain=domain.value,
                research_topic=research_topic,
                previous_context=previous_context[:1000],  # Limit context length
                requirements="\n".join(requirements) if requirements else "No specific requirements"
            )
            
            response = await self.kimi_client.generate_text(prompt)
            
            logger.info(f"Generated content for placeholder '{placeholder.title}' ({len(response.content)} chars)")
            return response.content
            
        except Exception as e:
            logger.error(f"Failed to generate placeholder content: {e}")
            return self._generate_fallback_content(placeholder, section, research_topic)
    
    def _generate_fallback_content(
        self,
        placeholder: ContentPlaceholder,
        section: EnhancedSection,
        research_topic: str
    ) -> str:
        """Generate fallback content when Kimi K2 is unavailable"""
        
        fallback_content = f"""
## {placeholder.title}

[This section discusses {placeholder.description.lower()} related to {research_topic}.]

{placeholder.description}

Key points to address:
"""
        
        if placeholder.required_elements:
            for element in placeholder.required_elements:
                fallback_content += f"\n- {element}"
        
        if placeholder.kimi_k2_prompt_hints:
            fallback_content += "\n\nAdditional considerations:"
            for hint in placeholder.kimi_k2_prompt_hints:
                fallback_content += f"\n- {hint}"
        
        fallback_content += f"\n\n[Content placeholder - approximately {placeholder.estimated_word_count} words needed]"
        
        return fallback_content
    
    async def generate_section_content(
        self,
        section: EnhancedSection,
        research_topic: str,
        domain: ResearchDomain,
        previous_sections_content: Dict[str, str] = None
    ) -> str:
        """Generate complete content for a section"""
        
        previous_sections_content = previous_sections_content or {}
        
        # Build context from previous sections
        context_parts = []
        for dep_id in section.dependencies:
            if dep_id in previous_sections_content:
                context_parts.append(f"From {dep_id}: {previous_sections_content[dep_id][:200]}...")
        
        previous_context = "\n".join(context_parts)
        
        # Generate content for each placeholder
        section_content_parts = [f"# {section.title}\n"]
        
        for placeholder in section.content_placeholders:
            placeholder_content = await self.generate_placeholder_content(
                placeholder, section, research_topic, domain, previous_context
            )
            section_content_parts.append(placeholder_content)
            section_content_parts.append("")  # Add spacing
        
        return "\n".join(section_content_parts)

class KimiK2QualityAssessor:
    """Quality assessor using Kimi K2 AI"""
    
    def __init__(self, kimi_client: Optional[KimiK2Client] = None):
        self.kimi_client = kimi_client or KimiK2Client()
        self.assessment_criteria = self._initialize_criteria()
    
    def _initialize_criteria(self) -> List[QualityAssessmentCriteria]:
        """Initialize quality assessment criteria"""
        return [
            QualityAssessmentCriteria(
                criteria_id="completeness",
                name="Content Completeness",
                description="How complete and comprehensive the content is",
                weight=0.25,
                kimi_k2_evaluation_prompt="Assess how well the content covers all necessary aspects of the topic"
            ),
            QualityAssessmentCriteria(
                criteria_id="coherence",
                name="Logical Coherence",
                description="How well the content flows logically and maintains consistency",
                weight=0.25,
                kimi_k2_evaluation_prompt="Evaluate the logical flow and consistency of arguments"
            ),
            QualityAssessmentCriteria(
                criteria_id="accuracy",
                name="Factual Accuracy",
                description="How accurate and reliable the information appears to be",
                weight=0.25,
                kimi_k2_evaluation_prompt="Assess the factual accuracy and reliability of claims made"
            ),
            QualityAssessmentCriteria(
                criteria_id="citation_quality",
                name="Citation Quality",
                description="Quality and appropriateness of citations and references",
                weight=0.25,
                kimi_k2_evaluation_prompt="Evaluate the quality and appropriateness of citations and references"
            )
        ]
    
    async def assess_draft_quality(
        self,
        draft: Draft,
        target_metrics: Optional[QualityMetrics] = None
    ) -> KimiK2QualityAssessment:
        """Assess the quality of a research draft using Kimi K2"""
        
        target_metrics = target_metrics or QualityMetrics(
            completeness=0.8,
            coherence=0.8,
            accuracy=0.8,
            citation_quality=0.8
        )
        
        try:
            # Prepare draft content for assessment
            draft_content = self._prepare_draft_content(draft)
            
            # Get quality assessment from Kimi K2
            assessment_data = await self._get_kimi_assessment(
                draft.topic,
                draft.structure.domain,
                draft_content,
                target_metrics
            )
            
            # Create quality assessment object
            assessment = KimiK2QualityAssessment(
                assessment_id=f"assessment_{datetime.now().timestamp()}",
                draft_id=draft.id,
                criteria_scores=assessment_data.get("criteria_scores", {}),
                overall_score=assessment_data.get("overall_score", 0.0),
                detailed_feedback=assessment_data.get("detailed_feedback", ""),
                improvement_suggestions=assessment_data.get("improvement_suggestions", []),
                kimi_k2_confidence=assessment_data.get("confidence", 0.8)
            )
            
            logger.info(f"Assessed draft quality: {assessment.overall_score:.2f}")
            return assessment
            
        except Exception as e:
            logger.error(f"Failed to assess draft quality: {e}")
            return self._generate_fallback_assessment(draft)
    
    def _prepare_draft_content(self, draft: Draft) -> str:
        """Prepare draft content for assessment"""
        content_parts = [f"Topic: {draft.topic}\n"]
        
        for section in draft.structure.sections:
            section_content = draft.content.get(section.id, "[No content]")
            content_parts.append(f"## {section.title}")
            content_parts.append(section_content)
            content_parts.append("")
        
        return "\n".join(content_parts)
    
    async def _get_kimi_assessment(
        self,
        topic: str,
        domain: ResearchDomain,
        draft_content: str,
        target_metrics: QualityMetrics
    ) -> Dict[str, Any]:
        """Get quality assessment from Kimi K2"""
        
        template = get_prompt_template("quality_assessment")
        if not template:
            raise ValueError("Quality assessment template not found")
        
        prompt = template.format_prompt(
            topic=topic,
            domain=domain.value,
            target_completeness=target_metrics.completeness,
            target_coherence=target_metrics.coherence,
            target_accuracy=target_metrics.accuracy,
            target_citation_quality=target_metrics.citation_quality,
            draft_content=draft_content[:4000]  # Limit content length
        )
        
        schema = {
            "type": "object",
            "properties": {
                "criteria_scores": {
                    "type": "object",
                    "properties": {
                        "completeness": {"type": "number", "minimum": 0, "maximum": 1},
                        "coherence": {"type": "number", "minimum": 0, "maximum": 1},
                        "accuracy": {"type": "number", "minimum": 0, "maximum": 1},
                        "citation_quality": {"type": "number", "minimum": 0, "maximum": 1}
                    }
                },
                "overall_score": {"type": "number", "minimum": 0, "maximum": 1},
                "detailed_feedback": {"type": "string"},
                "improvement_suggestions": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1}
            }
        }
        
        response = await self.kimi_client.generate_structured_response(prompt, schema)
        return response
    
    def _generate_fallback_assessment(self, draft: Draft) -> KimiK2QualityAssessment:
        """Generate fallback assessment when Kimi K2 is unavailable"""
        
        # Simple heuristic-based assessment
        total_content_length = sum(len(content) for content in draft.content.values())
        expected_length = draft.structure.estimated_length
        
        completeness = min(1.0, total_content_length / max(expected_length, 1))
        coherence = 0.7  # Default assumption
        accuracy = 0.7   # Default assumption
        citation_quality = 0.6  # Conservative estimate
        
        overall_score = (completeness + coherence + accuracy + citation_quality) / 4
        
        return KimiK2QualityAssessment(
            assessment_id=f"fallback_assessment_{datetime.now().timestamp()}",
            draft_id=draft.id,
            criteria_scores={
                "completeness": completeness,
                "coherence": coherence,
                "accuracy": accuracy,
                "citation_quality": citation_quality
            },
            overall_score=overall_score,
            detailed_feedback="Fallback assessment - Kimi K2 unavailable",
            improvement_suggestions=[
                "Expand content to meet target length",
                "Add more citations and references",
                "Review logical flow between sections"
            ],
            kimi_k2_confidence=0.5
        )
    
    async def suggest_improvements(
        self,
        assessment: KimiK2QualityAssessment,
        draft: Draft
    ) -> List[str]:
        """Generate specific improvement suggestions based on assessment"""
        
        suggestions = list(assessment.improvement_suggestions)
        
        # Add specific suggestions based on scores
        if assessment.criteria_scores.get("completeness", 0) < 0.7:
            suggestions.append("Add more detailed content to underdeveloped sections")
        
        if assessment.criteria_scores.get("coherence", 0) < 0.7:
            suggestions.append("Improve transitions between sections and logical flow")
        
        if assessment.criteria_scores.get("accuracy", 0) < 0.7:
            suggestions.append("Verify factual claims and add supporting evidence")
        
        if assessment.criteria_scores.get("citation_quality", 0) < 0.7:
            suggestions.append("Add more high-quality citations and improve reference formatting")
        
        return suggestions