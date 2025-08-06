"""
Draft Generator Node Implementation for TTD-DR Framework

This module implements the initial draft generation functionality using Kimi K2 model
for intelligent topic analysis and research skeleton creation.
"""

import logging
import uuid
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

from models.core import (
    TTDRState, Draft, ResearchStructure, Section, DraftMetadata,
    ResearchRequirements, ComplexityLevel, ResearchDomain, GapType, Priority
)
from services.kimi_k2_client import KimiK2Client, KimiK2Error

logger = logging.getLogger(__name__)

class DraftGenerationError(Exception):
    """Custom exception for draft generation errors"""
    pass

class KimiK2DraftGenerator:
    """
    Draft generator using Kimi K2 model for intelligent content creation
    """
    
    def __init__(self):
        """Initialize the draft generator with Kimi K2 client"""
        self.kimi_client = KimiK2Client()
        
        # Domain-specific section templates
        self.domain_templates = {
            ResearchDomain.TECHNOLOGY: [
                {"id": "introduction", "title": "Introduction", "weight": 0.15},
                {"id": "background", "title": "Background and Related Work", "weight": 0.20},
                {"id": "technical_overview", "title": "Technical Overview", "weight": 0.25},
                {"id": "implementation", "title": "Implementation and Applications", "weight": 0.20},
                {"id": "challenges", "title": "Challenges and Limitations", "weight": 0.10},
                {"id": "conclusion", "title": "Conclusion and Future Directions", "weight": 0.10}
            ],
            ResearchDomain.SCIENCE: [
                {"id": "introduction", "title": "Introduction", "weight": 0.15},
                {"id": "literature_review", "title": "Literature Review", "weight": 0.20},
                {"id": "methodology", "title": "Methodology", "weight": 0.20},
                {"id": "results", "title": "Results and Analysis", "weight": 0.25},
                {"id": "discussion", "title": "Discussion", "weight": 0.10},
                {"id": "conclusion", "title": "Conclusion", "weight": 0.10}
            ],
            ResearchDomain.BUSINESS: [
                {"id": "executive_summary", "title": "Executive Summary", "weight": 0.15},
                {"id": "market_analysis", "title": "Market Analysis", "weight": 0.25},
                {"id": "competitive_landscape", "title": "Competitive Landscape", "weight": 0.20},
                {"id": "strategic_recommendations", "title": "Strategic Recommendations", "weight": 0.25},
                {"id": "implementation", "title": "Implementation Plan", "weight": 0.10},
                {"id": "conclusion", "title": "Conclusion", "weight": 0.05}
            ],
            ResearchDomain.ACADEMIC: [
                {"id": "abstract", "title": "Abstract", "weight": 0.10},
                {"id": "introduction", "title": "Introduction", "weight": 0.15},
                {"id": "literature_review", "title": "Literature Review", "weight": 0.25},
                {"id": "methodology", "title": "Methodology", "weight": 0.20},
                {"id": "findings", "title": "Findings and Analysis", "weight": 0.20},
                {"id": "conclusion", "title": "Conclusion", "weight": 0.10}
            ],
            ResearchDomain.GENERAL: [
                {"id": "introduction", "title": "Introduction", "weight": 0.20},
                {"id": "background", "title": "Background", "weight": 0.25},
                {"id": "main_content", "title": "Main Analysis", "weight": 0.35},
                {"id": "implications", "title": "Implications", "weight": 0.10},
                {"id": "conclusion", "title": "Conclusion", "weight": 0.10}
            ]
        }
        
        # Complexity-based length multipliers
        self.complexity_multipliers = {
            ComplexityLevel.BASIC: 0.7,
            ComplexityLevel.INTERMEDIATE: 1.0,
            ComplexityLevel.ADVANCED: 1.3,
            ComplexityLevel.EXPERT: 1.6
        }
    
    async def generate_initial_draft(self, topic: str, requirements: ResearchRequirements) -> Draft:
        """
        Generate initial research draft using Kimi K2 model
        
        Args:
            topic: Research topic
            requirements: Research requirements and constraints
            
        Returns:
            Initial draft with structure and placeholder content
            
        Raises:
            DraftGenerationError: If draft generation fails
        """
        try:
            logger.info(f"Generating initial draft for topic: {topic}")
            
            # Step 1: Analyze topic and create research structure
            structure = await self._create_research_structure(topic, requirements)
            
            # Step 2: Generate initial content for each section
            content = await self._generate_section_content(topic, structure, requirements)
            
            # Step 3: Create draft object
            draft = Draft(
                id=str(uuid.uuid4()),
                topic=topic,
                structure=structure,
                content=content,
                metadata=DraftMetadata(
                    word_count=sum(len(text.split()) for text in content.values())
                ),
                quality_score=0.3,  # Initial low quality score
                iteration=0
            )
            
            logger.info(f"Successfully generated initial draft with {len(content)} sections")
            return draft
            
        except Exception as e:
            logger.error(f"Failed to generate initial draft: {str(e)}")
            raise DraftGenerationError(f"Draft generation failed: {str(e)}") from e
    
    async def _create_research_structure(self, topic: str, requirements: ResearchRequirements) -> ResearchStructure:
        """
        Create research structure using Kimi K2 topic analysis
        
        Args:
            topic: Research topic
            requirements: Research requirements
            
        Returns:
            Research structure with sections and metadata
        """
        try:
            # Build prompt for structure generation
            structure_prompt = self._build_structure_prompt(topic, requirements)
            
            # Define expected response schema
            structure_schema = {
                "type": "object",
                "properties": {
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "title": {"type": "string"},
                                "description": {"type": "string"},
                                "estimated_length": {"type": "integer"}
                            },
                            "required": ["id", "title", "description", "estimated_length"]
                        }
                    },
                    "total_estimated_length": {"type": "integer"},
                    "key_themes": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["sections", "total_estimated_length", "key_themes"]
            }
            
            # Generate structure using Kimi K2
            response = await self.kimi_client.generate_structured_response(
                structure_prompt, 
                structure_schema,
                temperature=0.3  # Lower temperature for more structured output
            )
            
            # Convert response to Section objects
            sections = []
            for section_data in response["sections"]:
                section = Section(
                    id=section_data["id"],
                    title=section_data["title"],
                    content="",  # Will be filled later
                    estimated_length=section_data["estimated_length"]
                )
                sections.append(section)
            
            # Create research structure
            structure = ResearchStructure(
                sections=sections,
                estimated_length=response["total_estimated_length"],
                complexity_level=requirements.complexity_level,
                domain=requirements.domain
            )
            
            logger.info(f"Created research structure with {len(sections)} sections")
            return structure
            
        except KimiK2Error as e:
            logger.error(f"Kimi K2 error in structure creation: {str(e)}")
            # Fallback to template-based structure
            return self._create_fallback_structure(topic, requirements)
        except Exception as e:
            logger.error(f"Unexpected error in structure creation: {str(e)}")
            # Also fallback for other errors to ensure robustness
            return self._create_fallback_structure(topic, requirements)
    
    def _build_structure_prompt(self, topic: str, requirements: ResearchRequirements) -> str:
        """
        Build prompt for research structure generation
        
        Args:
            topic: Research topic
            requirements: Research requirements
            
        Returns:
            Formatted prompt for Kimi K2
        """
        domain_guidance = {
            ResearchDomain.TECHNOLOGY: "Focus on technical specifications, implementation details, and practical applications.",
            ResearchDomain.SCIENCE: "Emphasize methodology, experimental design, and empirical findings.",
            ResearchDomain.BUSINESS: "Include market analysis, competitive landscape, and strategic implications.",
            ResearchDomain.ACADEMIC: "Follow academic structure with literature review and theoretical framework.",
            ResearchDomain.GENERAL: "Provide comprehensive coverage suitable for general audience."
        }
        
        complexity_guidance = {
            ComplexityLevel.BASIC: "Keep sections simple and accessible, avoid technical jargon.",
            ComplexityLevel.INTERMEDIATE: "Balance accessibility with depth, include moderate technical detail.",
            ComplexityLevel.ADVANCED: "Include detailed analysis and technical depth for expert audience.",
            ComplexityLevel.EXPERT: "Provide comprehensive, highly technical analysis with specialized terminology."
        }
        
        base_length = 3000
        target_length = int(base_length * self.complexity_multipliers[requirements.complexity_level])
        
        prompt = f"""
You are an expert research analyst tasked with creating a comprehensive research structure for the topic: "{topic}"

Research Parameters:
- Domain: {requirements.domain.value}
- Complexity Level: {requirements.complexity_level.value}
- Target Length: {target_length} words
- Domain Guidance: {domain_guidance.get(requirements.domain, "")}
- Complexity Guidance: {complexity_guidance.get(requirements.complexity_level, "")}

Create a detailed research structure that includes:
1. Logical section breakdown appropriate for the domain
2. Clear section titles that reflect the content focus
3. Brief descriptions of what each section should cover
4. Estimated word count for each section (totaling approximately {target_length} words)

Ensure the structure:
- Follows best practices for {requirements.domain.value} research
- Maintains logical flow from introduction to conclusion
- Balances depth with readability for {requirements.complexity_level.value} level
- Includes all essential components for comprehensive coverage

Generate section IDs using lowercase with underscores (e.g., "literature_review").
"""
        
        return prompt.strip()
    
    async def _generate_section_content(self, topic: str, structure: ResearchStructure, 
                                      requirements: ResearchRequirements) -> Dict[str, str]:
        """
        Generate initial content for each section using Kimi K2
        
        Args:
            topic: Research topic
            structure: Research structure
            requirements: Research requirements
            
        Returns:
            Dictionary mapping section IDs to initial content
        """
        content = {}
        
        for section in structure.sections:
            try:
                # Generate content for this section
                section_content = await self._generate_single_section_content(
                    topic, section, structure, requirements
                )
                content[section.id] = section_content
                
                logger.debug(f"Generated content for section: {section.id}")
                
            except Exception as e:
                logger.warning(f"Failed to generate content for section {section.id}: {str(e)}")
                # Fallback to placeholder content
                content[section.id] = self._create_placeholder_content(section, topic)
        
        return content
    
    async def _generate_single_section_content(self, topic: str, section: Section, 
                                             structure: ResearchStructure, 
                                             requirements: ResearchRequirements) -> str:
        """
        Generate content for a single section using Kimi K2
        
        Args:
            topic: Research topic
            section: Section to generate content for
            structure: Overall research structure for context
            requirements: Research requirements
            
        Returns:
            Generated content for the section
        """
        # Build context-aware prompt
        content_prompt = self._build_section_content_prompt(
            topic, section, structure, requirements
        )
        
        try:
            response = await self.kimi_client.generate_text(
                content_prompt,
                max_tokens=min(section.estimated_length * 2, 1500),  # Rough token estimation
                temperature=0.6  # Balanced creativity and consistency
            )
            
            return response.content.strip()
            
        except KimiK2Error as e:
            logger.warning(f"Kimi K2 error generating section content: {str(e)}")
            return self._create_placeholder_content(section, topic)
    
    def _build_section_content_prompt(self, topic: str, section: Section, 
                                    structure: ResearchStructure, 
                                    requirements: ResearchRequirements) -> str:
        """
        Build prompt for section content generation
        
        Args:
            topic: Research topic
            section: Section to generate content for
            structure: Overall research structure
            requirements: Research requirements
            
        Returns:
            Formatted prompt for content generation
        """
        # Get context from other sections
        section_context = []
        for s in structure.sections:
            if s.id != section.id:
                section_context.append(f"- {s.title}")
        
        context_text = "\n".join(section_context) if section_context else "None"
        
        prompt = f"""
You are writing the "{section.title}" section for a research report on "{topic}".

Research Context:
- Domain: {requirements.domain.value}
- Complexity Level: {requirements.complexity_level.value}
- Target Length: {section.estimated_length} words
- Other Sections: 
{context_text}

Section Requirements:
Write a comprehensive "{section.title}" section that:
1. Directly addresses the research topic "{topic}"
2. Maintains appropriate depth for {requirements.complexity_level.value} level
3. Uses terminology suitable for {requirements.domain.value} domain
4. Provides substantive content (not just placeholders)
5. Includes specific details and examples where relevant
6. Maintains professional academic tone

Important Guidelines:
- Write actual content, not outlines or bullet points
- Include specific information about {topic}
- Use proper markdown formatting with headers and subheaders
- Aim for approximately {section.estimated_length} words
- Ensure content flows logically and is well-structured
- Include relevant details that demonstrate understanding of the topic

Begin writing the section content now:
"""
        
        return prompt.strip()
    
    def _create_placeholder_content(self, section: Section, topic: str) -> str:
        """
        Create placeholder content when Kimi K2 generation fails
        
        Args:
            section: Section to create placeholder for
            topic: Research topic
            
        Returns:
            Placeholder content for the section
        """
        placeholder_templates = {
            "introduction": f"""# {section.title}

This research examines {topic}, exploring its key aspects and implications. The following analysis provides a comprehensive overview of the current state of knowledge and identifies areas for further investigation.

## Research Scope

The scope of this research encompasses the fundamental concepts, current developments, and future directions related to {topic}. This investigation aims to provide valuable insights for researchers, practitioners, and stakeholders in the field.

## Research Objectives

The primary objectives of this research include:
- Understanding the core principles and concepts
- Analyzing current trends and developments
- Identifying challenges and opportunities
- Providing actionable insights and recommendations

*[This section requires additional research and development to provide comprehensive coverage of the topic.]*
""",
            "background": f"""# {section.title}

The background of {topic} involves multiple interconnected factors that have shaped its current state. Understanding this context is essential for comprehensive analysis.

## Historical Context

The development of {topic} has evolved through several key phases, each contributing to our current understanding and application of the concepts involved.

## Current State

Today, {topic} represents a significant area of interest with ongoing developments and emerging trends that continue to shape the field.

*[This section requires detailed research to provide specific historical information and current developments.]*
""",
            "methodology": f"""# {section.title}

The research methodology for investigating {topic} involves a systematic approach to data collection, analysis, and interpretation.

## Research Approach

This investigation employs a comprehensive methodology designed to ensure thorough coverage of all relevant aspects of {topic}.

## Data Collection

The data collection process involves gathering information from multiple sources to provide a complete picture of the current state of knowledge.

*[This section requires specific methodological details and research design information.]*
""",
            "conclusion": f"""# {section.title}

This research on {topic} has provided valuable insights into the current state of knowledge and identified key areas for future development.

## Key Findings

The investigation has revealed several important aspects of {topic} that contribute to our understanding of the field.

## Future Directions

Based on the analysis conducted, several opportunities for future research and development have been identified.

*[This section requires synthesis of findings from the complete research to provide meaningful conclusions.]*
"""
        }
        
        # Try to match section ID to template
        for template_key, template_content in placeholder_templates.items():
            if template_key in section.id.lower():
                return template_content
        
        # Default placeholder
        return f"""# {section.title}

This section focuses on {section.title.lower()} aspects of {topic}. The analysis in this section provides important insights into the research topic.

## Overview

{section.title} represents a crucial component of understanding {topic}. This section examines the relevant factors and considerations.

## Key Points

The key points covered in this section include:
- Fundamental concepts and principles
- Current developments and trends
- Implications and significance
- Areas for further investigation

*[This section requires additional research and development to provide comprehensive coverage of {section.title.lower()} related to {topic}.]*

## Summary

The {section.title.lower()} analysis contributes to the overall understanding of {topic} and provides foundation for further research and development.
"""
    
    def _create_fallback_structure(self, topic: str, requirements: ResearchRequirements) -> ResearchStructure:
        """
        Create fallback research structure when Kimi K2 fails
        
        Args:
            topic: Research topic
            requirements: Research requirements
            
        Returns:
            Template-based research structure
        """
        logger.info("Using fallback template-based structure generation")
        
        # Get template for domain
        template = self.domain_templates.get(requirements.domain, self.domain_templates[ResearchDomain.GENERAL])
        
        # Calculate base length and apply complexity multiplier
        base_length = 3000
        total_length = int(base_length * self.complexity_multipliers[requirements.complexity_level])
        
        # Create sections from template
        sections = []
        for section_template in template:
            estimated_length = int(total_length * section_template["weight"])
            
            section = Section(
                id=section_template["id"],
                title=section_template["title"],
                content="",
                estimated_length=estimated_length
            )
            sections.append(section)
        
        return ResearchStructure(
            sections=sections,
            estimated_length=total_length,
            complexity_level=requirements.complexity_level,
            domain=requirements.domain
        )

async def draft_generator_node(state: TTDRState) -> TTDRState:
    """
    LangGraph node function for generating initial research draft
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with initial draft
    """
    logger.info("Executing draft_generator_node with Kimi K2 integration")
    
    try:
        # Initialize draft generator
        generator = KimiK2DraftGenerator()
        
        # Generate initial draft
        draft = await generator.generate_initial_draft(
            state["topic"], 
            state["requirements"]
        )
        
        logger.info(f"Successfully generated initial draft for topic: {state['topic']}")
        
        return {
            **state,
            "current_draft": draft,
            "iteration_count": 0,
            "error_log": state.get("error_log", [])
        }
        
    except Exception as e:
        error_msg = f"Draft generation failed: {str(e)}"
        logger.error(error_msg)
        
        # Add error to log but don't fail the workflow
        error_log = state.get("error_log", [])
        error_log.append(f"[{datetime.now()}] {error_msg}")
        
        return {
            **state,
            "current_draft": None,
            "iteration_count": 0,
            "error_log": error_log
        }